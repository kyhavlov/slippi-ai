import collections
import dataclasses
import enum
import itertools
import math
import random
import threading
import uuid
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from melee import Character

from slippi_ai import nametags
from slippi_ai import data as data_lib

CHARACTER_ALIASES: Dict[str, Character] = {
    'ganondor': Character.GANONDORF,
    'ganon': Character.GANONDORF,
    'captainfalcon': Character.CPTFALCON,
    'captfalcon': Character.CPTFALCON,
    'cpt falcon': Character.CPTFALCON,
    'cptfalcon': Character.CPTFALCON,
}


class AssignmentStatus(enum.Enum):
  COMPLETE = 'complete'
  ABORTED = 'aborted'


@dataclasses.dataclass(frozen=True)
class SlotSpec:
  """Identifies a single (env, port, name) slot."""
  env_id: int
  port_index: int  # 0-3 ordering to match GameState players
  name: str


@dataclasses.dataclass(frozen=True)
class Assignment:
  assignment_id: str
  env_id: int
  characters: Tuple[Character, Character, Character, Character]
  names: Tuple[str, str, str, str]


def _normalize_character_key(raw: str) -> Optional[Character]:
  key = raw.strip().lower()
  if not key:
    return None
  try:
    return data_lib.name_to_character[key]
  except KeyError as exc:
    alias = CHARACTER_ALIASES.get(key)
    if alias is not None:
      return alias
    raise ValueError(f'Unknown character label "{raw}".') from exc


def parse_name_allowlist(spec: str) -> Mapping[Character, set[str]]:
  """Parses allowlist strings like 'ALL:Master Player,Fox:Cody,Dragunov,...'."""
  if not spec:
    raise ValueError('Empty name allowlist specification.')

  entries: Dict[str, List[str]] = {}
  current_key: Optional[str] = None
  tokens = [token.strip() for token in spec.split(',')]
  for token in tokens:
    if not token:
      continue
    if ':' in token:
      key, remainder = token.split(':', 1)
      current_key = key.strip()
      entries[current_key] = []
      remainder = remainder.strip()
      if remainder:
        entries[current_key].append(remainder)
    else:
      if current_key is None:
        raise ValueError(f'Allowlist token "{token}" missing a preceding character tag.')
      entries[current_key].append(token)

  global_names = [
      nametags.normalize_name(name.strip())
      for name in entries.get('ALL', [])
      if name.strip()
  ]

  char_map: Dict[Character, set[str]] = {}
  for key, names in entries.items():
    if key == 'ALL':
      continue
    char = _normalize_character_key(key)
    if char is None:
      continue
    normalized_names = {
        nametags.normalize_name(name.strip())
        for name in names
        if name.strip()
    }
    allowed_names = set(normalized_names)
    allowed_names.update(global_names)
    if not allowed_names:
      raise ValueError(f'Character {key} has no allowed names.')
    char_map[char] = allowed_names

  if not char_map:
    raise ValueError('Parsed allowlist contains no usable character->name entries.')
  return char_map


def _canonical_matchup(characters: Sequence[Character]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
  team1 = tuple(sorted((characters[0].value, characters[1].value)))
  team2 = tuple(sorted((characters[2].value, characters[3].value)))
  return tuple(sorted((team1, team2)))


def _canonical_team(char_a: Character, char_b: Character) -> Tuple[int, int]:
  values = sorted((char_a.value, char_b.value))
  return (values[0], values[1])


@dataclasses.dataclass
class _ScoringStats:
  char_counts: collections.Counter
  matchup_counts: collections.Counter
  team_counts: collections.Counter
  char_name_counts: collections.Counter
  mean_char: float
  mean_matchup: float
  mean_team: float
  matchup_variance: float
  team_variance: float
  per_char_name_mean: Dict[Character, float]


class CharacterScheduler:
  """Balances character/matchup/name usage across many environments."""

  def __init__(
      self,
      allowlist: Mapping[Character, set[str]],
      slot_specs: Sequence[SlotSpec],
      *,
      char_weight: float = 1.0,
      matchup_weight: float = 1.0,
      team_weight: float = 1.0,
      char_name_weight: float = 0.2,
      rng_seed: int = 0,
      max_candidates: int = 1024,
      max_slot_options: int = 4,
  ):
    if not slot_specs:
      raise ValueError('CharacterScheduler requires at least one slot.')
    self._char_weight = char_weight
    self._matchup_weight = matchup_weight
    self._team_weight = team_weight
    self._char_name_weight = char_name_weight
    self._rng = random.Random(rng_seed)
    self._max_candidates = max_candidates
    self._max_slot_options = max_slot_options
    self._lock = threading.RLock()

    self._allowlist = {char: set(names) for char, names in allowlist.items()}
    self._names_by_char: Dict[Character, set[str]] = {
        char: set(names) for char, names in allowlist.items()
    }
    self._slot_specs: Dict[Tuple[int, int], SlotSpec] = {}
    self._slot_normalized_names: Dict[Tuple[int, int], str] = {}
    self._slot_characters: Dict[Tuple[int, int], Tuple[Character, ...]] = {}
    self._display_names: Dict[str, str] = {}
    self._name_slot_counts: collections.Counter[str] = collections.Counter()
    self._slot_char_lists: Dict[Tuple[int, int], list[Character]] = {}
    for spec in slot_specs:
      key = (spec.env_id, spec.port_index)
      if key in self._slot_specs:
        raise ValueError(f'Duplicate slot registration for env={spec.env_id} port={spec.port_index}')
      normalized_name = nametags.normalize_name(spec.name)
      allowed_chars = tuple(sorted(
          [char for char, names in allowlist.items() if normalized_name in names],
          key=lambda c: c.value,
      ))
      if not allowed_chars:
        raise ValueError(f'Name "{spec.name}" has no allowed characters in the provided allowlist.')
      self._slot_specs[key] = spec
      self._slot_normalized_names[key] = normalized_name
      self._display_names.setdefault(normalized_name, spec.name)
      self._slot_characters[key] = allowed_chars
      self._name_slot_counts[normalized_name] += 1
      self._slot_char_lists[key] = list(allowed_chars)

    self._char_capacity: collections.Counter[Character] = collections.Counter()
    self._char_name_capacity: collections.Counter[Tuple[Character, str]] = collections.Counter()
    for slot_key, allowed_chars in self._slot_characters.items():
      normalized_name = self._slot_normalized_names[slot_key]
      for char in allowed_chars:
        self._char_capacity[char] += 1
        self._char_name_capacity[(char, normalized_name)] += 1

    for char in self._allowlist:
      if self._char_capacity[char] == 0:
        raise ValueError(
            f'Character {char.name} is allowed in the config but no slot uses an allowed name.')

    self._env_slots: Dict[int, Tuple[SlotSpec, SlotSpec, SlotSpec, SlotSpec]] = {}
    for env_id in {spec.env_id for spec in slot_specs}:
      env_slots = [self._slot_specs[(env_id, port)] for port in range(4)]
      if len(env_slots) != 4:
        raise ValueError(f'Env {env_id} missing slot definitions; expected 4.')
      self._env_slots[env_id] = tuple(env_slots)  # type: ignore[arg-type]

    self._char_completed: collections.Counter[Character] = collections.Counter()
    self._char_pending: collections.Counter[Character] = collections.Counter()
    self._matchup_completed: collections.Counter = collections.Counter()
    self._matchup_pending: collections.Counter = collections.Counter()
    self._team_completed: collections.Counter = collections.Counter()
    self._team_pending: collections.Counter = collections.Counter()
    self._char_name_completed: collections.Counter = collections.Counter()
    self._char_name_pending: collections.Counter = collections.Counter()
    self._active_assignments: Dict[int, Assignment] = {}
    self._pending_assignments: Dict[str, Assignment] = {}
    self._char_allowed_names: Dict[Character, frozenset[str]] = {
        char: frozenset(names)
        for char, names in self._allowlist.items()
    }

  # Public API --------------------------------------------------------------

  def request_assignment(self, env_id: int) -> Assignment:
    with self._lock:
      if env_id in self._active_assignments:
        raise ValueError(f'Env {env_id} already has an active assignment.')
      slots = self._env_slots[env_id]
      characters = self._choose_characters(env_id, slots)
      assignment_id = uuid.uuid4().hex
      assignment = Assignment(
          assignment_id=assignment_id,
          env_id=env_id,
          characters=characters,
          names=tuple(slot.name for slot in slots),
      )
      self._apply_pending_counts(assignment)
      self._active_assignments[env_id] = assignment
      self._pending_assignments[assignment_id] = assignment
      return assignment

  def report_outcome(self, env_id: int, assignment_id: str, status: AssignmentStatus):
    with self._lock:
      assignment = self._pending_assignments.pop(assignment_id, None)
      if assignment is None:
        raise ValueError(f'Assignment {assignment_id} not found.')
      if env_id not in self._active_assignments:
        raise ValueError(f'Env {env_id} has no recorded assignment.')
      del self._active_assignments[env_id]
      if status is AssignmentStatus.ABORTED:
        self._revert_pending_counts(assignment)
        return
      self._commit_assignment(assignment)

  def stats_snapshot(self) -> dict:
    with self._lock:
      char_counts = self._combined_counter(self._char_completed, self._char_pending)
      matchup_counts = self._combined_counter(self._matchup_completed, self._matchup_pending)
      char_name_counts = self._combined_counter(self._char_name_completed, self._char_name_pending)
      team_counts = self._combined_counter(self._team_completed, self._team_pending)
      readable_char_names = {}
      for (char, normalized_name), value in char_name_counts.items():
        display_name = self._display_names.get(normalized_name, normalized_name)
        readable_char_names[(char, display_name)] = value
      return {
          'character': char_counts,
          'matchup': matchup_counts,
          'team': team_counts,
          'char_name': readable_char_names,
      }

  def capacity_snapshot(self) -> Dict[Character, int]:
    with self._lock:
      return dict(self._char_capacity)

  def name_capacity_snapshot(self) -> Dict[str, int]:
    with self._lock:
      return dict(self._name_slot_counts)

  def allowed_name_sets(self) -> Dict[Character, frozenset[str]]:
    with self._lock:
      return dict(self._char_allowed_names)

  def display_name(self, normalized_name: str) -> str:
    return self._display_names.get(normalized_name, normalized_name)

  # Internal helpers -------------------------------------------------------

  def _combined_counter(self, complete: collections.Counter, pending: collections.Counter):
    result = complete.copy()
    result.update(pending)
    return result

  def _build_scoring_stats(self) -> _ScoringStats:
    char_counts = self._combined_counter(self._char_completed, self._char_pending)
    matchup_counts = self._combined_counter(self._matchup_completed, self._matchup_pending)
    char_name_counts = self._combined_counter(self._char_name_completed, self._char_name_pending)
    team_counts = self._combined_counter(self._team_completed, self._team_pending)
    total_char_counts = sum(char_counts.values())
    mean_char = (
        total_char_counts / len(self._allowlist)
        if self._allowlist else 0.0
    )
    per_char_name_mean: Dict[Character, float] = {}
    for char, names in self._names_by_char.items():
      if not names:
        per_char_name_mean[char] = 0.0
        continue
      total_for_char = sum(char_name_counts.get((char, name), 0) for name in names)
      per_char_name_mean[char] = total_for_char / len(names)
    total_matchup_counts = sum(matchup_counts.values())
    mean_matchup = (
        total_matchup_counts / len(matchup_counts)
        if matchup_counts else 0.0
    )
    matchup_var = 0.0
    if matchup_counts:
      matchup_var = sum(
          (count - mean_matchup) ** 2
          for count in matchup_counts.values()
      ) / len(matchup_counts)
    total_team_counts = sum(team_counts.values())
    mean_team = (
        total_team_counts / len(team_counts)
        if team_counts else 0.0
    )
    team_var = 0.0
    if team_counts:
      team_var = sum(
          (count - mean_team) ** 2
          for count in team_counts.values()
      ) / len(team_counts)
    return _ScoringStats(
        char_counts=char_counts,
        matchup_counts=matchup_counts,
        team_counts=team_counts,
        char_name_counts=char_name_counts,
        mean_char=mean_char,
        mean_matchup=mean_matchup,
        mean_team=mean_team,
        matchup_variance=matchup_var,
        team_variance=team_var,
        per_char_name_mean=per_char_name_mean,
    )

  def _apply_pending_counts(self, assignment: Assignment):
    slots = self._env_slots[assignment.env_id]
    for char in assignment.characters:
      self._char_pending[char] += 1
    matchup = _canonical_matchup(assignment.characters)
    self._matchup_pending[matchup] += 1
    team1 = _canonical_team(assignment.characters[0], assignment.characters[1])
    team2 = _canonical_team(assignment.characters[2], assignment.characters[3])
    self._team_pending[team1] += 1
    self._team_pending[team2] += 1
    for char, slot in zip(assignment.characters, slots):
      key = (slot.env_id, slot.port_index)
      normalized = self._slot_normalized_names[key]
      self._char_name_pending[(char, normalized)] += 1

  def _revert_pending_counts(self, assignment: Assignment):
    slots = self._env_slots[assignment.env_id]
    for char in assignment.characters:
      self._char_pending[char] -= 1
      if self._char_pending[char] <= 0:
        del self._char_pending[char]
    matchup = _canonical_matchup(assignment.characters)
    self._matchup_pending[matchup] -= 1
    if self._matchup_pending[matchup] <= 0:
      del self._matchup_pending[matchup]
    team1 = _canonical_team(assignment.characters[0], assignment.characters[1])
    team2 = _canonical_team(assignment.characters[2], assignment.characters[3])
    for team in (team1, team2):
      self._team_pending[team] -= 1
      if self._team_pending[team] <= 0:
        del self._team_pending[team]
    for char, slot in zip(assignment.characters, slots):
      slot_key = (slot.env_id, slot.port_index)
      normalized = self._slot_normalized_names[slot_key]
      key = (char, normalized)
      self._char_name_pending[key] -= 1
      if self._char_name_pending[key] <= 0:
        del self._char_name_pending[key]

  def _commit_assignment(self, assignment: Assignment):
    slots = self._env_slots[assignment.env_id]
    for char in assignment.characters:
      self._char_pending[char] -= 1
      if self._char_pending[char] <= 0:
        del self._char_pending[char]
      self._char_completed[char] += 1
    matchup = _canonical_matchup(assignment.characters)
    self._matchup_pending[matchup] -= 1
    if self._matchup_pending[matchup] <= 0:
      del self._matchup_pending[matchup]
    self._matchup_completed[matchup] += 1
    team1 = _canonical_team(assignment.characters[0], assignment.characters[1])
    team2 = _canonical_team(assignment.characters[2], assignment.characters[3])
    for team in (team1, team2):
      self._team_pending[team] -= 1
      if self._team_pending[team] <= 0:
        del self._team_pending[team]
      self._team_completed[team] += 1

    for char, slot in zip(assignment.characters, slots):
      slot_key = (slot.env_id, slot.port_index)
      normalized = self._slot_normalized_names[slot_key]
      key = (char, normalized)
      self._char_name_pending[key] -= 1
      if self._char_name_pending[key] <= 0:
        del self._char_name_pending[key]
      self._char_name_completed[key] += 1

  def _choose_characters(self, env_id: int, slots: Sequence[SlotSpec]) -> Tuple[Character, ...]:
    option_lists = [
        self._slot_characters[(env_id, slot.port_index)]
        for slot in slots
    ]
    total_combos = 1
    for options in option_lists:
      total_combos *= len(options)
    if total_combos == 0:
      raise ValueError(f'Env {env_id} has a slot with no legal characters.')

    combined_char_counts = self._combined_counter(self._char_completed, self._char_pending)
    sorted_options = []
    for options in option_lists:
      ranked = list(options)
      self._rng.shuffle(ranked)
      ranked.sort(key=lambda c: combined_char_counts.get(c, 0))
      sorted_options.append(tuple(ranked))

    truncated_options = []
    for opts in sorted_options:
      if len(opts) > self._max_slot_options > 0:
        truncated_options.append(opts[:self._max_slot_options])
      else:
        truncated_options.append(opts)
    truncated_total = math.prod(len(opts) for opts in truncated_options)

    if truncated_total == 0:
      raise ValueError(f'No valid character assignments available for env {env_id}.')

    best_score = None
    best_combos: list[Tuple[Character, ...]] = []
    epsilon = 1e-6

    stats = self._build_scoring_stats()

    if truncated_total <= self._max_candidates:
      candidates = itertools.product(*truncated_options)
    else:
      candidates = self._beam_search(truncated_options, slots, stats)

    for combo in candidates:
      score = self._score_combo(combo, slots, stats)
      if best_score is None or score < best_score - epsilon:
        best_score = score
        best_combos = [combo]
      elif abs(score - best_score) <= epsilon:
        best_combos.append(combo)

    if not best_combos:
      raise ValueError(f'No valid character assignments available for env {env_id}.')
    return tuple(self._rng.choice(best_combos))

  def _beam_search(self, option_lists, slots, stats: _ScoringStats):
    beam_width = self._max_candidates // max(1, len(option_lists[0]))
    beam_width = max(8, beam_width)

    State = tuple[list[Character], collections.Counter, collections.Counter, collections.Counter, float]
    beam: list[State] = [([], collections.Counter(), collections.Counter(), collections.Counter(), 0.0)]

    for slot_index, slot in enumerate(slots):
      normalized_name = self._slot_normalized_names[(slot.env_id, slot.port_index)]
      options = option_lists[slot_index]
      new_beam: list[State] = []
      for chars, char_ctr, name_ctr, team_ctr, partial_score in beam:
        for char in options:
          char_repeats = char_ctr[char]
          char_diff = self._char_weight * (
              stats.char_counts.get(char, 0) + char_repeats + 1 - stats.mean_char)
          key = (char, normalized_name)
          name_repeats = name_ctr[key]
          name_mean = stats.per_char_name_mean.get(char, 0.0)
          name_diff = self._char_name_weight * (
              stats.char_name_counts.get(key, 0) + name_repeats + 1 - name_mean)
          new_char_ctr = char_ctr.copy()
          new_char_ctr[char] += 1
          new_name_ctr = name_ctr.copy()
          new_name_ctr[key] += 1
          new_chars = chars + [char]
          new_team_ctr = team_ctr.copy()
          new_score = partial_score + char_diff + name_diff
          if slot_index % 2 == 1:
            team_key = _canonical_team(new_chars[slot_index - 1], new_chars[slot_index])
            repeats = new_team_ctr[team_key]
            existing = stats.team_counts.get(team_key, 0)
            team_variance = stats.team_variance or 1.0
            new_score += self._team_weight * (
                (existing + repeats + 1) - stats.mean_team) / math.sqrt(team_variance)
            new_team_ctr[team_key] += 1
          new_beam.append((new_chars, new_char_ctr, new_name_ctr, new_team_ctr, new_score))

      if not new_beam:
        break
      new_beam.sort(key=lambda state: state[4])
      beam = new_beam[:beam_width]

    return [tuple(state[0]) for state in beam if len(state[0]) == 4]

  def _score_combo(self, combo: Tuple[Character, ...], slots: Sequence[SlotSpec], stats: Optional[_ScoringStats] = None) -> float:
    if stats is None:
      stats = self._build_scoring_stats()

    score = 0.0
    combo_counter = collections.Counter(combo)
    for char, repeats in combo_counter.items():
      existing = stats.char_counts.get(char, 0)
      for i in range(repeats):
        projected = existing + i + 1
        score += self._char_weight * (projected - stats.mean_char)

    matchup_key = _canonical_matchup(combo)
    matchup_existing = stats.matchup_counts.get(matchup_key, 0)
    matchup_variance = stats.matchup_variance or 1.0
    score += self._matchup_weight * ((matchup_existing + 1) - stats.mean_matchup) / math.sqrt(matchup_variance)

    team_variance = stats.team_variance or 1.0
    team1 = _canonical_team(combo[0], combo[1])
    team2 = _canonical_team(combo[2], combo[3])
    for team in (team1, team2):
      existing = stats.team_counts.get(team, 0)
      score += self._team_weight * ((existing + 1) - stats.mean_team) / math.sqrt(team_variance)

    for char, slot in zip(combo, slots):
      normalized_name = self._slot_normalized_names[(slot.env_id, slot.port_index)]
      existing = stats.char_name_counts.get((char, normalized_name), 0)
      mean_value = stats.per_char_name_mean.get(char, 0.0)
      score += self._char_name_weight * ((existing + 1) - mean_value)

    score += self._rng.random() * 1e-4
    return score
