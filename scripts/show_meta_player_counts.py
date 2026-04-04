"""Print normalized player counts from an existing meta.json.

By default this reports replay counts per player, deduplicated within a replay.
Use --count_mode=trajectory to count emitted main-player trajectories instead.
"""

import collections
import json

from absl import app, flags

from slippi_ai import nametags


META_PATH = flags.DEFINE_string(
    'meta_path',
    None,
    'Path to the meta.json file to summarize.',
    required=True,
)

COUNT_MODE = flags.DEFINE_enum(
    'count_mode',
    'replay',
    ['replay', 'trajectory'],
    'Whether to count unique replays per player or main-player trajectories.',
)


def _player_names_for_row(row: dict) -> list[str]:
  players = row.get('players', [])
  allowed_indices = row.get('allowed_main_player_indices')
  if allowed_indices is None:
    allowed_indices = range(len(players))

  raw = row.get('raw')
  names: list[str] = []
  for idx in allowed_indices:
    idx = int(idx)
    if idx < 0 or idx >= len(players):
      raise ValueError(
          f'allowed_main_player_indices contains invalid index {idx} '
          f'for replay {row.get("name", row.get("slp_md5", "<unknown>"))}')
    player = players[idx]
    names.append(nametags.normalize_name(
        nametags.name_from_metadata(player, raw=raw)))
  return names


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with open(META_PATH.value) as f:
    rows = json.load(f)

  counts = collections.Counter()
  for row in rows:
    names = _player_names_for_row(row)
    if COUNT_MODE.value == 'replay':
      counts.update(set(names))
    else:
      counts.update(names)

  print(f'Rows: {len(rows)}')
  print(f'Count mode: {COUNT_MODE.value}')
  for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
    print(f'{count:6d}  {name}')


if __name__ == '__main__':
  app.run(main)
