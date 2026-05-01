"""The final step of dataset creation.

python slippi_db/scripts/make_local_dataset.py --root=Root \
  --singles_blacklist=singles_blacklist.txt \
  --doubles_whitelist=doubles_whitelist.txt
"""

from __future__ import annotations

import collections
import os
import pickle
import json
import tarfile
from typing import Iterable, Optional

import tqdm
from slippi_ai import nametags
from melee import Character

from absl import app, flags
from slippi_db import file_layout

DEFAULT_SUMMARY_LIMIT = 300

ROOT = flags.DEFINE_string('root', None, 'root directory', required=True)
WINNER_ONLY = flags.DEFINE_boolean(
  'winner_only', True, 'only keep games that have a winner')

MAKE_TAR = flags.DEFINE_boolean('tar', False, 'Create dataset tar archive')

DOUBLES_ONLY = flags.DEFINE_boolean(
  'doubles_only', True, 'only use doubles games')

SINGLES_BLACKLIST = flags.DEFINE_string(
    'singles_blacklist',
    None,
    'Optional path to newline-separated player names to exclude from singles.',
)

DOUBLES_WHITELIST = flags.DEFINE_string(
    'doubles_whitelist',
    None,
    'Optional path to newline-separated player names to include for doubles.',
)

SUMMARY_LIMIT = flags.DEFINE_integer(
    'summary_limit',
    DEFAULT_SUMMARY_LIMIT,
    'Maximum player/character pairings to print in metadata summaries.',
)


_EXCLUDED_CHARACTERS = {
    Character.WIREFRAME_MALE,
    Character.WIREFRAME_FEMALE,
    Character.GIGA_BOWSER,
    Character.SANDBAG,
    Character.UNKNOWN_CHARACTER,
}


def _normalize_player_names(
    names: Optional[Iterable[str]],
) -> Optional[set[str]]:
  if names is None:
    return None

  return {nametags.normalize_name(name.strip()) for name in names if name.strip()}


def _load_player_names_file(
    path: Optional[str],
) -> Optional[set[str]]:
  if not path:
    return None

  with open(path) as f:
    return _normalize_player_names(
        line for line in f
        if line.strip() and not line.strip().startswith('#'))


def _metadata_player_name(row: dict, player: dict) -> str:
  code = nametags.name_from_metadata(player, raw=row.get('raw'))
  return nametags.normalize_name(code)


def _selected_main_player_indices(
    row: dict,
    *,
    singles_blacklist: Optional[set[str]],
    doubles_whitelist: Optional[set[str]],
) -> list[int]:
  if not row.get('is_teams') and singles_blacklist is None:
    return list(range(len(row['players'])))

  if row.get('is_teams') and doubles_whitelist is None:
    return list(range(len(row['players'])))

  matching_indices: list[int] = []
  for i, player in enumerate(row['players']):
    name = _metadata_player_name(row, player)
    if row.get('is_teams'):
      if name in doubles_whitelist:
        matching_indices.append(i)
    elif name not in singles_blacklist:
      matching_indices.append(i)

  return matching_indices


def is_valid_replay(
    row: dict,
    *,
    doubles_only: bool,
    singles_blacklist: Optional[set[str]],
    doubles_whitelist: Optional[set[str]],
) -> bool:
  if not row.get('is_training'):
    return False

  if doubles_only and not row.get('is_teams'):
    return False

  for player in row['players']:
    if Character(player['character']) in _EXCLUDED_CHARACTERS:
      return False

  return bool(_selected_main_player_indices(
      row,
      singles_blacklist=singles_blacklist,
      doubles_whitelist=doubles_whitelist,
  ))

def _summarize(rows: list[dict], *, summary_limit: int) -> None:
  _summarize_rows(
      rows,
      only_allowed_main_players=False,
      summary_limit=summary_limit)


def _iter_summary_players(
    row: dict,
    *,
    only_allowed_main_players: bool,
):
  players = row['players']
  if only_allowed_main_players:
    indices = row.get('allowed_main_player_indices')
    if indices is None:
      indices = range(len(players))
    for i in indices:
      yield players[int(i)]
  else:
    for player in players:
      yield player


def _summarize_rows(
    rows: list[dict],
    *,
    only_allowed_main_players: bool,
    summary_limit: int,
) -> None:
  player_character_counts = collections.Counter()
  character_counts = collections.Counter()
  trajectory_count = 0

  for row in rows:
    if len(row['players']) != 4:
      continue

    raw = row.get('raw')
    for player in _iter_summary_players(
        row,
        only_allowed_main_players=only_allowed_main_players,
    ):
      code = nametags.name_from_metadata(player, raw=raw)
      name = nametags.normalize_name(code)

      trajectory_count += 1
      player_character_counts[(name, Character(player['character']))] += 1
      character_counts[Character(player['character'])] += 1

  print(f"Found {trajectory_count} trajectories.")
  print("Character trajectories:")
  for character, count in character_counts.most_common():
    print(f"({character}): {count}")
  print("")
  print("Player/Character Pairings:")
  for (name, character), count in player_character_counts.most_common(
      n=summary_limit):
    print(f"{name} ({character}): {count}")


def _summarize_singles(rows: list[dict], *, summary_limit: int) -> None:
  _summarize_singles_rows(
      rows,
      only_allowed_main_players=False,
      summary_limit=summary_limit)


def _summarize_singles_rows(
    rows: list[dict],
    *,
    only_allowed_main_players: bool,
    summary_limit: int,
) -> None:
  singles_player_character_counts = collections.Counter()
  singles_character_counts = collections.Counter()
  trajectory_count = 0

  for row in rows:
    if row.get('is_teams'):
      continue

    raw = row.get('raw')
    for player in _iter_summary_players(
        row,
        only_allowed_main_players=only_allowed_main_players,
    ):
      code = nametags.name_from_metadata(player, raw=raw)
      name = nametags.normalize_name(code)
      singles_player_character_counts[(name, Character(player['character']))] += 1
      singles_character_counts[Character(player['character'])] += 1
      trajectory_count += 1

  print(f"Found total {trajectory_count} singles trajectories.")
  print("Singles character trajectories:")
  for character, count in singles_character_counts.most_common():
    print(f"({character}): {count}")
  print("")
  print("Singles player/character pairings:")
  for (name, character), count in singles_player_character_counts.most_common(
      n=summary_limit):
    print(f"{name} ({character}): {count}")


def _summarize_selected_players(
    rows: list[dict],
    *,
    configured_players: Optional[Iterable[str]],
) -> None:
  replay_counts = collections.Counter()
  if configured_players is not None:
    replay_counts.update({name: 0 for name in sorted(configured_players)})

  for row in rows:
    allowed_indices = row.get('allowed_main_player_indices')
    if allowed_indices is None:
      continue
    raw = row.get('raw')
    for i in allowed_indices:
      player = row['players'][i]
      code = nametags.name_from_metadata(player, raw=raw)
      name = nametags.normalize_name(code)
      replay_counts[name] += 1

  print("")
  print("Selected main-player replay counts:")
  for name, count in replay_counts.items():
    print(f"{name}: {count}")


def build_meta(
    root: str | os.PathLike[str],
    *,
    doubles_only: Optional[bool] = None,
    winner_only: Optional[bool] = None,
    make_tar: Optional[bool] = None,
    singles_blacklist: Optional[Iterable[str]] = None,
    doubles_whitelist: Optional[Iterable[str]] = None,
    singles_blacklist_file: Optional[str] = None,
    doubles_whitelist_file: Optional[str] = None,
    summary_limit: Optional[int] = None,
    quiet: bool = False,
) -> list[dict]:
  root_path = os.fspath(root)

  with open(os.path.join(root_path, 'parsed.pkl'), 'rb') as f:
    rows = pickle.load(f)

  doubles_only = DOUBLES_ONLY.value if doubles_only is None else doubles_only
  winner_only = WINNER_ONLY.value if winner_only is None else winner_only
  make_tar = MAKE_TAR.value if make_tar is None else make_tar
  if summary_limit is None:
    summary_limit = (
        SUMMARY_LIMIT.value if flags.FLAGS.is_parsed()
        else DEFAULT_SUMMARY_LIMIT)

  if singles_blacklist_file is None:
    singles_blacklist_file = SINGLES_BLACKLIST.value
  if doubles_whitelist_file is None:
    doubles_whitelist_file = DOUBLES_WHITELIST.value

  file_singles_blacklist = _load_player_names_file(singles_blacklist_file)
  file_doubles_whitelist = _load_player_names_file(doubles_whitelist_file)
  singles_blacklist = _normalize_player_names(singles_blacklist)
  doubles_whitelist = _normalize_player_names(doubles_whitelist)
  if file_singles_blacklist is not None:
    singles_blacklist = file_singles_blacklist
  if file_doubles_whitelist is not None:
    doubles_whitelist = file_doubles_whitelist

  has_player_filter = (
      singles_blacklist is not None or doubles_whitelist is not None)

  total_rows = len(rows)
  filtered_rows: list[dict] = []
  for row in rows:
    if not is_valid_replay(
        row,
        doubles_only=doubles_only,
        singles_blacklist=singles_blacklist,
        doubles_whitelist=doubles_whitelist,
    ):
      continue

    row = dict(row)
    if has_player_filter:
      row['allowed_main_player_indices'] = _selected_main_player_indices(
          row,
          singles_blacklist=singles_blacklist,
          doubles_whitelist=doubles_whitelist,
      )
    else:
      row.pop('allowed_main_player_indices', None)
    filtered_rows.append(row)
  rows = filtered_rows

  if not quiet:
    print(f"Found {total_rows} replays.")
    print(f"{len(rows)} replays after filtering.")

  if winner_only:
    rows = [row for row in rows if row.get('winner') is not None]
    if not quiet:
      print(f"Filtered to {len(rows)} games with a winner.")

  doubles_games = sum(1 for row in rows if row.get('is_teams'))
  singles_games = len(rows) - doubles_games
  if not quiet:
    print(f"Doubles games: {doubles_games}, Singles games: {singles_games}")

  if not quiet:
    if has_player_filter:
      _summarize_rows(
          rows,
          only_allowed_main_players=True,
          summary_limit=summary_limit)
      if not doubles_only:
        _summarize_singles_rows(
            rows,
            only_allowed_main_players=True,
            summary_limit=summary_limit)
      _summarize_selected_players(
          rows,
          configured_players=doubles_whitelist,
      )
    else:
      _summarize(rows, summary_limit=summary_limit)
      if not doubles_only:
        _summarize_singles(rows, summary_limit=summary_limit)

  tar = None
  if make_tar:
    tar = tarfile.open(os.path.join(root_path, 'training.tar'), 'w')

  missing = collections.Counter()
  iterator = rows if quiet else tqdm.tqdm(rows, smoothing=0, unit='slp')
  for row in iterator:
    md5 = row['slp_md5']
    parsed_dir = os.path.join(root_path, 'Parsed')
    parsed_path = file_layout.resolve_parquet_path(parsed_dir, md5)
    if not os.path.isfile(parsed_path):
      missing[row.get('raw', row['name'])] += 1

    if tar is not None:
      tar.add(parsed_path, arcname='games/' + md5)

  if not quiet:
    print(f"Missing: {missing}")

  meta_path = os.path.join(root_path, 'meta.json')
  with open(meta_path, 'w') as f:
    json.dump(rows, f, indent=2)

  if tar is not None:
    tar.add(meta_path, arcname='meta.json')
    tar.close()

  return rows


def main(_):
  build_meta(ROOT.value)

if __name__ == '__main__':
  app.run(main)
