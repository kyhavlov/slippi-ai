import collections
import dataclasses
import enum
import itertools
import math
import random
import threading
import uuid
from multiprocessing.managers import SyncManager
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
  port_index: int  # 0..slots_per_env-1 ordering to match GameState players
  name: str


@dataclasses.dataclass(frozen=True)
class Assignment:
  assignment_id: str
  env_id: int
  characters: Tuple[Character, ...]
  names: Tuple[str, ...]


# In doubles we treat ports 0 & 3 as one team vs ports 1 & 2.
DEFAULT_TEAM_PORTS: tuple[tuple[int, int], ...] = ((0, 3), (1, 2))

def _build_teammate_map(team_ports: Sequence[tuple[int, int]]) -> dict[int, int]:
  return {a: b for pair in team_ports for a, b in (pair, (pair[1], pair[0]))}


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


def normalize_name_list(display_names: Sequence[str]) -> list[tuple[str, str]]:
  names: list[tuple[str, str]] = []
  for raw in display_names:
    display = raw.strip()
    if not display:
      continue
    normalized = nametags.normalize_name(display)
    names.append((normalized, display))
  if not names:
    raise ValueError('At least one non-empty name is required to build the layout.')
  return names


def parse_name_csv(raw: str) -> list[tuple[str, str]]:
  tokens = [token.strip() for token in raw.split(',')]
  return normalize_name_list(tokens)


def allocate_name_slots(
    names: list[tuple[str, str]],
    allowlist: Mapping[Character, set[str]],
    num_envs: int,
    layout_seed: int,
    *,
    slots_per_env: int = 4,
) -> list[str]:
  normalized_to_display: dict[str, str] = {}
  for normalized, display in names:
    normalized_to_display.setdefault(normalized, display)

  referenced_names = set().union(*allowlist.values())
  missing = referenced_names - set(normalized_to_display)
  if missing:
    raise ValueError(
        'Names referenced by the allowlist are missing from the provided list: '
        + ', '.join(sorted(missing)))

  if slots_per_env <= 0:
    raise ValueError('slots_per_env must be > 0.')
  total_slots = num_envs * slots_per_env
  weights: dict[str, float] = {name: 0.0 for name in referenced_names}
  for allowed_names in allowlist.values():
    if not allowed_names:
      continue
    share = 1.0 / len(allowed_names)
    for name in allowed_names:
      weights[name] += share

  # If there are more distinct names than slots, select a weighted subset so
  # small smoke runs (e.g. few envs) can still schedule valid layouts.
  if len(weights) > total_slots:
    rng = random.Random(layout_seed)
    pool = list(weights.keys())
    pool_weights = [weights[name] for name in pool]
    chosen: list[str] = []
    for _ in range(total_slots):
      total_weight = float(sum(pool_weights))
      if total_weight <= 0:
        idx = rng.randrange(len(pool))
      else:
        r = rng.random() * total_weight
        acc = 0.0
        idx = 0
        for idx, w in enumerate(pool_weights):
          acc += float(w)
          if acc >= r:
            break
      chosen.append(pool.pop(idx))
      pool_weights.pop(idx)
    weights = {name: weights[name] for name in chosen}

  total_weight = sum(weights.values())
  if total_weight == 0:
    raise ValueError('Allowlist does not assign any names to characters.')

  ideal_counts = {
      name: weights[name] / total_weight * total_slots
      for name in weights
  }
  slot_counts = {name: int(count) for name, count in ideal_counts.items()}
  remainders = {name: ideal_counts[name] - slot_counts[name] for name in weights}

  for name in weights:
    if slot_counts[name] == 0:
      slot_counts[name] = 1
      remainders[name] = 0.0

  current_total = sum(slot_counts.values())
  if current_total > total_slots:
    surplus = current_total - total_slots
    ordered = sorted(
        slot_counts.items(),
        key=lambda item: (slot_counts[item[0]], -remainders[item[0]]),
        reverse=True,
    )
    idx = 0
    while surplus > 0 and idx < len(ordered):
      name, _ = ordered[idx]
      if slot_counts[name] > 1:
        slot_counts[name] -= 1
        surplus -= 1
      else:
        idx += 1

  current_total = sum(slot_counts.values())
  while current_total < total_slots:
    best = max(weights.keys(), key=lambda n: (remainders[n], weights[n]))
    slot_counts[best] += 1
    current_total += 1

  slots: list[str] = []
  ordered_names = sorted(
      slot_counts.keys(),
      key=lambda name: (-slot_counts[name], name),
  )
  for name in ordered_names:
    display = normalized_to_display[name]
    slots.extend([display] * slot_counts[name])

  rng = random.Random(layout_seed)
  rng.shuffle(slots)
  return slots


def build_slot_specs(
    layout: Sequence[str],
    num_envs: int,
    *,
    slots_per_env: int = 4,
) -> list['SlotSpec']:
  if slots_per_env <= 0:
    raise ValueError('slots_per_env must be > 0.')
  if len(layout) != num_envs * slots_per_env:
    raise ValueError('Layout length must be exactly num_envs * slots_per_env.')
  specs: list[SlotSpec] = []
  for idx, name in enumerate(layout):
    env_id = idx // slots_per_env
    port_index = idx % slots_per_env
    specs.append(SlotSpec(env_id=env_id, port_index=port_index, name=name))
  return specs


def _canonical_matchup(
    characters: Sequence[Character],
    team_ports: Sequence[tuple[int, int]],
) -> Tuple[Tuple[int, ...], ...]:
  if team_ports:
    teams = tuple(
        _canonical_team(characters[a], characters[b])
        for (a, b) in team_ports
    )
    return tuple(sorted(teams))

  values = sorted(char.value for char in characters)
  return (tuple(values),)


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
      slots_per_env: int = 4,
      team_ports: Sequence[tuple[int, int]] | None = None,
      char_weight: float = 1.0,
      matchup_weight: float = 1.0,
      team_weight: float = 1.0,
      char_name_weight: float = 0.2,
      rng_seed: int = 0,
      max_candidates: int = 1024,
      max_slot_options: int = 4,
      max_outstanding_per_env: int = 2,
  ):
    if not slot_specs:
      raise ValueError('CharacterScheduler requires at least one slot.')
    if slots_per_env <= 0:
      raise ValueError('slots_per_env must be > 0.')
    if team_ports is None:
      team_ports = DEFAULT_TEAM_PORTS if slots_per_env == 4 else ()
    team_ports = tuple(tuple(p) for p in team_ports)
    for a, b in team_ports:
      if a == b:
        raise ValueError('team_ports pairs must reference distinct ports.')
      if a < 0 or b < 0 or a >= slots_per_env or b >= slots_per_env:
        raise ValueError('team_ports indices must be within slots_per_env.')

    self._slots_per_env = int(slots_per_env)
    self._team_ports = team_ports
    self._teammate_port = _build_teammate_map(team_ports)
    self._char_weight = char_weight
    self._matchup_weight = matchup_weight
    self._team_weight = team_weight
    self._char_name_weight = char_name_weight
    self._rng = random.Random(rng_seed)
    self._max_candidates = max_candidates
    self._max_slot_options = max_slot_options
    self._max_outstanding_per_env = max(1, max_outstanding_per_env)
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

    self._env_slots: Dict[int, Tuple[SlotSpec, ...]] = {}
    for env_id in {spec.env_id for spec in slot_specs}:
      env_slots = [self._slot_specs[(env_id, port)] for port in range(self._slots_per_env)]
      if len(env_slots) != self._slots_per_env:
        raise ValueError(
            f'Env {env_id} missing slot definitions; expected {self._slots_per_env}.')
      self._env_slots[env_id] = tuple(env_slots)

    self._char_completed: collections.Counter[Character] = collections.Counter()
    self._char_pending: collections.Counter[Character] = collections.Counter()
    self._matchup_completed: collections.Counter = collections.Counter()
    self._matchup_pending: collections.Counter = collections.Counter()
    self._team_completed: collections.Counter = collections.Counter()
    self._team_pending: collections.Counter = collections.Counter()
    self._char_name_completed: collections.Counter = collections.Counter()
    self._char_name_pending: collections.Counter = collections.Counter()
    self._pending_assignments: Dict[str, Assignment] = {}
    self._env_active_counts: collections.Counter[int] = collections.Counter()
    self._env_assignment_ids: Dict[int, set[str]] = collections.defaultdict(set)
    self._char_allowed_names: Dict[Character, frozenset[str]] = {
        char: frozenset(names)
        for char, names in self._allowlist.items()
    }

  # Public API --------------------------------------------------------------

  def request_assignment(self, env_id: int) -> Assignment:
    with self._lock:
      if self._env_active_counts[env_id] >= self._max_outstanding_per_env:
        raise ValueError(
            f'Env {env_id} has {self._env_active_counts[env_id]} outstanding assignments; '
            f'max {self._max_outstanding_per_env}.')
      slots = self._env_slots[env_id]
    stats = self._build_scoring_stats()
    characters = self._choose_characters(env_id, slots, stats)
    assignment_id = uuid.uuid4().hex
    assignment = Assignment(
        assignment_id=assignment_id,
        env_id=env_id,
        characters=characters,
        names=tuple(slot.name for slot in slots),
    )
    with self._lock:
      if self._env_active_counts[env_id] >= self._max_outstanding_per_env:
        raise ValueError(
            f'Env {env_id} exceeded outstanding assignments while scheduling.')
      self._apply_pending_counts(assignment)
      self._pending_assignments[assignment.assignment_id] = assignment
      self._env_active_counts[env_id] += 1
      self._env_assignment_ids[env_id].add(assignment.assignment_id)
    return assignment

  def report_outcome(self, env_id: int, assignment_id: str, status: AssignmentStatus):
    with self._lock:
      assignment = self._pending_assignments.pop(assignment_id, None)
      if assignment is None:
        raise ValueError(f'Assignment {assignment_id} not found.')
      if assignment_id not in self._env_assignment_ids.get(env_id, set()):
        raise ValueError(f'Env {env_id} not tracking assignment {assignment_id}.')
      self._env_assignment_ids[env_id].discard(assignment_id)
      self._env_active_counts[env_id] -= 1
      if self._env_active_counts[env_id] <= 0:
        del self._env_active_counts[env_id]
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
    with self._lock:
      char_completed = self._char_completed.copy()
      char_pending = self._char_pending.copy()
      matchup_completed = self._matchup_completed.copy()
      matchup_pending = self._matchup_pending.copy()
      char_name_completed = self._char_name_completed.copy()
      char_name_pending = self._char_name_pending.copy()
      team_completed = self._team_completed.copy()
      team_pending = self._team_pending.copy()
      allowlist = self._allowlist
      names_by_char = self._names_by_char

    char_counts = self._combined_counter(char_completed, char_pending)
    matchup_counts = self._combined_counter(matchup_completed, matchup_pending)
    char_name_counts = self._combined_counter(char_name_completed, char_name_pending)
    team_counts = self._combined_counter(team_completed, team_pending)
    total_char_counts = sum(char_counts.values())
    mean_char = (
        total_char_counts / len(allowlist)
        if allowlist else 0.0
    )
    per_char_name_mean: Dict[Character, float] = {}
    for char, names in names_by_char.items():
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
    matchup = _canonical_matchup(assignment.characters, self._team_ports)
    self._matchup_pending[matchup] += 1
    for idx_a, idx_b in self._team_ports:
      team = _canonical_team(assignment.characters[idx_a], assignment.characters[idx_b])
      self._team_pending[team] += 1
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
    matchup = _canonical_matchup(assignment.characters, self._team_ports)
    self._matchup_pending[matchup] -= 1
    if self._matchup_pending[matchup] <= 0:
      del self._matchup_pending[matchup]
    for idx_a, idx_b in self._team_ports:
      team = _canonical_team(assignment.characters[idx_a], assignment.characters[idx_b])
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
    matchup = _canonical_matchup(assignment.characters, self._team_ports)
    self._matchup_pending[matchup] -= 1
    if self._matchup_pending[matchup] <= 0:
      del self._matchup_pending[matchup]
    self._matchup_completed[matchup] += 1
    for idx_a, idx_b in self._team_ports:
      team = _canonical_team(assignment.characters[idx_a], assignment.characters[idx_b])
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

  def _choose_characters(self, env_id: int, slots: Sequence[SlotSpec], stats: _ScoringStats) -> Tuple[Character, ...]:
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
          teammate_port = self._teammate_port.get(slot.port_index)
          if teammate_port is not None and teammate_port < len(new_chars):
            teammate_char = new_chars[teammate_port]
            team_key = _canonical_team(teammate_char, new_chars[slot_index])
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

    return [tuple(state[0]) for state in beam if len(state[0]) == self._slots_per_env]

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

    matchup_key = _canonical_matchup(combo, self._team_ports)
    matchup_existing = stats.matchup_counts.get(matchup_key, 0)
    matchup_variance = stats.matchup_variance or 1.0
    score += self._matchup_weight * ((matchup_existing + 1) - stats.mean_matchup) / math.sqrt(matchup_variance)

    team_variance = stats.team_variance or 1.0
    for idx_a, idx_b in self._team_ports:
      team = _canonical_team(combo[idx_a], combo[idx_b])
      existing = stats.team_counts.get(team, 0)
      score += self._team_weight * ((existing + 1) - stats.mean_team) / math.sqrt(team_variance)

    for char, slot in zip(combo, slots):
      normalized_name = self._slot_normalized_names[(slot.env_id, slot.port_index)]
      existing = stats.char_name_counts.get((char, normalized_name), 0)
      mean_value = stats.per_char_name_mean.get(char, 0.0)
      score += self._char_name_weight * ((existing + 1) - mean_value)

    score += self._rng.random() * 1e-4
    return score


class _SchedulerManager(SyncManager):
  pass


_SchedulerManager.register('CharacterScheduler', CharacterScheduler)


def start_scheduler_manager(**scheduler_kwargs):
  manager = _SchedulerManager()
  manager.start()
  scheduler = manager.CharacterScheduler(**scheduler_kwargs)
  return manager, scheduler
