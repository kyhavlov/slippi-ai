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
    allocate_name_slots,
    build_slot_specs,
    parse_name_allowlist,
    parse_name_csv,
)


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
  names = parse_name_csv(args.names)
  allowlist = parse_name_allowlist(args.name_allowlist)
  layout = allocate_name_slots(names, allowlist, args.num_envs, args.layout_seed)
  slots = build_slot_specs(layout, args.num_envs)
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
