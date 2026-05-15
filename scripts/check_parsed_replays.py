#!/usr/bin/env python3
"""Inspect parsed Slippi parquet logs for singles/doubles sanity.

Reads a random sample of hashed parquet files (default: from ``data/Parsed``)
and reports whether singles games contain meaningful action or look idle. For
each singles replay we compute:

* Whether auxiliary slots (p1 / p3) are empty as expected.
* Total per-player damage (approximate) and number of stocks lost.
* Number of frames with attack buttons pressed.
* Number of frames with significant stick movement.

Games that fall below configurable thresholds for damage, attack frames, or
KO count are flagged so you can spot warm-up / idle matches quickly.

Example usage::

    ./scripts/check_parsed_replays.py --count 100 \
        --min-damage 50 --min-attack-frames 20

    ./scripts/check_parsed_replays.py --count 10000 --seed 42

No files are modified; output is printed to stdout.
"""

from __future__ import annotations

import argparse
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from slippi_ai import types


ATTACK_BUTTONS = ('A', 'B', 'X', 'Y', 'Z')
STICK_THRESHOLD = 0.15  # approximate dead-zone for movement detection


@dataclass
class PlayerStats:
  port: str
  character: int
  attack_frames: int
  move_frames: int
  damage: float
  kos_taken: int
  total_frames: int

  @property
  def is_present(self) -> bool:
    return self.character != 0


def _read_table(path: Path) -> pq.Table:
  data = path.read_bytes()
  try:
    data = zlib.decompress(data)
  except zlib.error:
    pass
  return pq.read_table(pa.BufferReader(data))


def _player_stats(port_name: str, player: types.Player) -> PlayerStats:
  character = int(player.character[0]) if player.character.size else 0
  total_frames = player.percent.size

  if character == 0:
    return PlayerStats(port_name, 0, 0, 0, 0.0, 0, total_frames)

  percent = player.percent.astype(np.int32)
  # Positive deltas approximate damage taken on this port
  deltas = np.diff(percent, prepend=percent[:1])
  positive = np.clip(deltas, 0, None)
  damage = float(np.sum(positive))

  stocks = player.stocks_left.astype(np.int32)
  if stocks.size:
    kos_taken = int(stocks[0] - stocks[-1])
    damage += 100.0 * kos_taken  # approximate percent reset after KO
  else:
    kos_taken = 0

  buttons = player.controller.buttons
  attack_frames = int(sum(np.count_nonzero(getattr(buttons, btn))
                          for btn in ATTACK_BUTTONS))

  main_x = player.controller.main_stick.x.astype(np.float32)
  main_y = player.controller.main_stick.y.astype(np.float32)
  mag = np.sqrt(main_x ** 2 + main_y ** 2)
  move_frames = int(np.count_nonzero(mag > STICK_THRESHOLD))

  return PlayerStats(
      port=port_name,
      character=character,
      attack_frames=attack_frames,
      move_frames=move_frames,
      damage=damage,
      kos_taken=kos_taken,
      total_frames=total_frames,
  )


def _summarize_game(game_struct: pa.StructArray) -> tuple[bool, list[PlayerStats]]:
  game = types.game_array_to_nt(game_struct)
  stats = [
      _player_stats('p0', game.p0),
      _player_stats('p1', game.p1),
      _player_stats('p2', game.p2),
      _player_stats('p3', game.p3),
  ]
  is_teams = bool(game.is_teams[0])
  return is_teams, stats


def _iter_hash_files(root: Path) -> Iterable[Path]:
  return sorted(
      path for path in root.rglob('*')
      if path.is_file() and len(path.name) == 32
  )


def analyze(root: Path, count: int, seed: int | None,
            min_damage: float, min_attack_frames: int,
            min_kos: int) -> None:
  files = _iter_hash_files(root)
  if not files:
    raise SystemExit(f'No hash-named parquet files found under {root}.')

  if seed is not None:
    random.Random(seed).shuffle(files)

  sample = files[:count]

  singles = 0
  doubles = 0
  p1_should_be_empty = 0
  p3_should_be_empty = 0
  suspicious: list[tuple[Path, float, int, int]] = []

  for path in sample:
    table = _read_table(path)
    game_struct = table['root'].combine_chunks()
    is_teams, players = _summarize_game(game_struct)

    if is_teams:
      doubles += 1
      continue

    singles += 1

    # Raw parquet keeps original ports; expect p1/p3 empty in singles slices
    if not players[1].is_present:
      p1_should_be_empty += 1
    if not players[3].is_present:
      p3_should_be_empty += 1

    present_players = [p for p in players if p.is_present]
    total_damage = sum(p.damage for p in present_players)
    total_attacks = sum(p.attack_frames for p in present_players)
    total_kos = sum(p.kos_taken for p in present_players)

    if (total_damage < min_damage or
        total_attacks < min_attack_frames or
        total_kos < min_kos):
      suspicious.append((path, total_damage, total_attacks, total_kos))

  print(f'Scanned {len(sample)} files: singles={singles}, doubles={doubles}')
  if singles:
    print(f'  singles p1 empty : {p1_should_be_empty}/{singles} '
          f'({p1_should_be_empty / singles:.1%})')
    print(f'  singles p3 empty : {p3_should_be_empty}/{singles} '
          f'({p3_should_be_empty / singles:.1%})')

  if suspicious:
    print('\nPotentially low-action singles replays (below thresholds):')
    for path, dmg, attacks, kos in suspicious[:20]:
      print(f'  {path}: damage={dmg:.1f}, attack_frames={attacks}, KOs={kos}')
    if len(suspicious) > 20:
      print(f'  ... and {len(suspicious) - 20} more')
  else:
    print('\nNo singles replays fell below the specified thresholds.')


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root', type=Path, default=Path('data/Parsed'),
                      help='Root directory containing hashed parquet files.')
  parser.add_argument('--count', type=int, default=100,
                      help='Number of files to inspect.')
  parser.add_argument('--seed', type=int, default=0,
                      help='Shuffle seed (set to None for deterministic order).')
  parser.add_argument('--min-damage', type=float, default=75.0,
                      help='Minimum total damage (across both players) before flagging.')
  parser.add_argument('--min-attack-frames', type=int, default=25,
                      help='Minimum total attack-button frames before flagging.')
  parser.add_argument('--min-kos', type=int, default=1,
                      help='Minimum total KOs before flagging.')
  args = parser.parse_args()

  analyze(args.root, args.count, args.seed,
          args.min_damage, args.min_attack_frames, args.min_kos)


if __name__ == '__main__':
  main()
