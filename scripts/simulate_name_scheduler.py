#!/usr/bin/env python3

import argparse
import collections
import random
from typing import Iterable, Mapping, Tuple

from melee import Character

from slippi_ai.rl.character_scheduler import (
    AssignmentStatus,
    CharacterScheduler,
    SlotSpec,
    parse_name_allowlist,
)
from slippi_ai import nametags


def _parse_names(raw: str) -> list[tuple[str, str]]:
  names = []
  for token in raw.split(','):
    display = token.strip()
    if not display:
      continue
    normalized = nametags.normalize_name(display)
    names.append((normalized, display))
  if not names:
    raise ValueError('You must provide at least one name via --names.')
  return names


def _allocate_name_slots(
    names: list[tuple[str, str]],
    allowlist: Mapping[Character, set[str]],
    num_envs: int,
    layout_seed: int,
) -> list[str]:
  normalized_to_display: dict[str, str] = {}
  for normalized, display in names:
    normalized_to_display.setdefault(normalized, display)

  referenced_names = set().union(*allowlist.values())
  missing = referenced_names - set(normalized_to_display)
  if missing:
    raise ValueError(
        'The following names appear in the allowlist but not in --names: '
        + ', '.join(sorted(missing)))

  total_slots = num_envs * 4
  weights: dict[str, float] = {name: 0.0 for name in referenced_names}
  for allowed_names in allowlist.values():
    if not allowed_names:
      continue
    for name in allowed_names:
      weights[name] += 1.0 / len(allowed_names)

  total_weight = sum(weights.values())
  if total_weight == 0:
    raise ValueError('Allowlist does not assign any names to characters.')

  ideal_counts = {name: weights[name] / total_weight * total_slots for name in weights}
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
    best = max(
        weights.keys(),
        key=lambda n: (remainders[n], weights[n]),
    )
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


def _build_slot_specs(layout: list[str], num_envs: int) -> list[SlotSpec]:
  if len(layout) != num_envs * 4:
    raise ValueError('Layout length must equal num_envs * 4.')
  specs: list[SlotSpec] = []
  for idx, name in enumerate(layout):
    env_id = idx // 4
    port = idx % 4
    specs.append(SlotSpec(env_id=env_id, port_index=port, name=name))
  return specs


def _spread(values: Iterable[int]) -> Tuple[int, int, int]:
  data = list(values)
  if not data:
    return (0, 0, 0)
  return (min(data), max(data), max(data) - min(data))


def _format_matchup(key: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> str:
  team_a = '/'.join(Character(v).name for v in key[0])
  team_b = '/'.join(Character(v).name for v in key[1])
  return f'{team_a} vs {team_b}'


def _format_team(team: Tuple[int, ...]) -> str:
  return '/'.join(Character(v).name for v in team)


def _summarize(counter, formatter, list_all=False, top_k=5):
  if not counter:
    print('  (no data)')
    return
  min_val, max_val, spread = _spread(counter.values())
  print(f'  min={min_val} max={max_val} spread={spread}')
  sorted_items = sorted(counter.items(), key=lambda item: item[1])
  if list_all:
    for key, value in sorted_items:
      print(f'    {formatter(key)} -> {value}')
  else:
    print('  Lowest entries:')
    for key, value in sorted_items[:top_k]:
      print(f'    {formatter(key)} -> {value}')
    print('  Highest entries:')
    for key, value in sorted_items[-top_k:]:
      print(f'    {formatter(key)} -> {value}')


def main():
  parser = argparse.ArgumentParser(
      description='Simulate the character/name scheduler to inspect distribution stats.')
  parser.add_argument('--names', required=True,
                      help='Comma-separated list of nametags assigned to ports.')
  parser.add_argument('--name-allowlist', required=True,
                      help='Allowlist string (e.g., "ALL:Master Player,Fox:Cody,Dragunov,...").')
  parser.add_argument('--num-envs', type=int, required=True,
                      help='Number of parallel environments to simulate.')
  parser.add_argument('--num-games', type=int, default=10000,
                      help='Total number of games to simulate.')
  parser.add_argument('--seed', type=int, default=0,
                      help='Random seed for tie-breaking and env sampling.')
  parser.add_argument('--abort-prob', type=float, default=0.0,
                      help='Probability that a simulated game crashes (to test reset handling).')
  parser.add_argument('--log-interval', type=int, default=2000,
                      help='Print intermediate statistics every N games.')
  parser.add_argument('--max-candidates', type=int, default=1024,
                      help='Max character combinations evaluated per assignment (sampled if exceeded).')
  parser.add_argument('--max-slot-options', type=int, default=4,
                      help='Limit on how many characters per slot are considered each assignment.')
  parser.add_argument('--layout-seed', type=int, default=0,
                      help='Random seed for shuffling the computed nametag layout.')

  args = parser.parse_args()
  names = _parse_names(args.names)
  allowlist = parse_name_allowlist(args.name_allowlist)
  layout = _allocate_name_slots(names, allowlist, args.num_envs, args.layout_seed)
  slots = _build_slot_specs(layout, args.num_envs)
  scheduler = CharacterScheduler(
      allowlist=allowlist,
      slot_specs=slots,
      rng_seed=args.seed,
      max_candidates=args.max_candidates,
      max_slot_options=args.max_slot_options,
  )

  rng = random.Random(args.seed + 123)
  completed = 0
  aborted = 0

  for game in range(1, args.num_games + 1):
    env_id = rng.randrange(args.num_envs)
    assignment = scheduler.request_assignment(env_id)
    if rng.random() < args.abort_prob:
      scheduler.report_outcome(env_id, assignment.assignment_id, AssignmentStatus.ABORTED)
      aborted += 1
    else:
      scheduler.report_outcome(env_id, assignment.assignment_id, AssignmentStatus.COMPLETE)
      completed += 1

    if args.log_interval and game % args.log_interval == 0:
      print(f'Completed {game} games (good={completed}, aborted={aborted})')

  stats = scheduler.stats_snapshot()
  char_counts = stats['character']
  matchup_counts = stats['matchup']
  char_name_counts = stats['char_name']
  capacities = scheduler.capacity_snapshot()
  name_capacities = scheduler.name_capacity_snapshot()
  allowed_sets = scheduler.allowed_name_sets()

  total_slots = args.num_envs * 4
  total_assignments = args.num_games * 4
  equal_target = total_assignments / max(1, len(capacities))

  impossible: list[tuple[str, float, float]] = []
  for char, capacity in capacities.items():
    max_possible = capacity / total_slots * total_assignments
    if equal_target - max_possible > 1e-6:
      impossible.append((Character(char).name, max_possible, equal_target))

  group_warnings = []
  group_map: dict[frozenset[str], list[Character]] = {}
  for char, names in allowed_sets.items():
    group_map.setdefault(names, []).append(char)
  for name_set, chars in group_map.items():
    if not name_set:
      continue
    capacity_slots = sum(name_capacities.get(name, 0) for name in name_set)
    max_possible = capacity_slots / total_slots * total_assignments
    target_group = equal_target * len(chars)
    if target_group - max_possible > 1e-6:
      display_names = [scheduler.display_name(name) for name in name_set]
      char_names = [Character(c).name for c in chars]
      group_warnings.append((char_names, display_names, max_possible, target_group))

  print('\nCharacter distribution:')
  _summarize(char_counts, lambda c: Character(c).name)
  print(f'  Target per character if perfectly even: {equal_target:.1f}')
  if impossible:
    print('  WARNING: The following characters cannot reach the target with the current names/allowlist:')
    for name, max_possible, target in impossible:
      print(f'    {name}: max feasible {max_possible:.1f} vs target {target:.1f}')
  if group_warnings:
    print('  WARNING: Combined capacity limits hit for these character/name groups:')
    for chars, names, max_possible, target in group_warnings:
      print(f'    Characters {chars} share names {names}; max feasible {max_possible:.1f} vs target {target:.1f}')
  print('\nMatchup distribution:')
  _summarize(matchup_counts, _format_matchup, top_k=50)
  team_counts = collections.Counter()
  for matchup, count in matchup_counts.items():
    for team in matchup:
      team_counts[team] += count
  print('\nTeam composition distribution (per side):')
  _summarize(team_counts, _format_team, top_k=50)
  print('\nCharacter + name distribution:')
  _summarize(char_name_counts, lambda k: f'{Character(k[0]).name} as {k[1]}', list_all=True)
  total_recorded = sum(char_counts.values())
  print(f'\nTotal successful games counted: {total_recorded // 4}')
  print(f'Total aborted games: {aborted}')


if __name__ == '__main__':
  main()
