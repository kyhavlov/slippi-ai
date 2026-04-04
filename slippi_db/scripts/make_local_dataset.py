"""The final step of dataset creation.

python slippi_db/scripts/make_local_dataset.py --root=Root
"""

from __future__ import annotations

import collections
import os
import pickle
import json
import tarfile
from typing import Dict, Iterable, Optional

import tqdm
from slippi_ai import nametags
from melee import Character

from absl import app, flags
from slippi_db import file_layout

ROOT = flags.DEFINE_string('root', None, 'root directory', required=True)
WINNER_ONLY = flags.DEFINE_boolean(
  'winner_only', True, 'only keep games that have a winner')

MAKE_TAR = flags.DEFINE_boolean('tar', False, 'Create dataset tar archive')

DOUBLES_ONLY = flags.DEFINE_boolean(
  'doubles_only', True, 'only use doubles games')

ALLOWED_PLAYERS_FILE = flags.DEFINE_string(
    'allowed_players_file',
    None,
    'Optional path to a newline-separated whitelist of allowed players.',
)


_EXCLUDED_CHARACTERS = {
    Character.WIREFRAME_MALE,
    Character.WIREFRAME_FEMALE,
    Character.GIGA_BOWSER,
    Character.SANDBAG,
    Character.UNKNOWN_CHARACTER,
}


def _normalize_allowed_players(
    allowed_players: Optional[Dict[str, Optional[Iterable[Character]]]],
) -> Optional[Dict[str, Optional[set[Character]]]]:
  if allowed_players is None:
    return None

  normalised: Dict[str, Optional[set[Character]]] = {}
  for name, characters in allowed_players.items():
    if characters is None:
      normalised[name] = None
    else:
      normalised[name] = set(characters)
  return normalised


def _load_allowed_players_file(
    path: Optional[str],
) -> Optional[Dict[str, Optional[set[Character]]]]:
  if not path:
    return None

  allowed_players: Dict[str, Optional[set[Character]]] = {}
  with open(path) as f:
    for raw_line in f:
      line = raw_line.strip()
      if not line or line.startswith('#'):
        continue
      allowed_players[nametags.normalize_name(line)] = None
  return allowed_players


def _matching_allowed_player_indices(
    row: dict,
    *,
    allowed_players: Optional[Dict[str, Optional[set[Character]]]],
) -> list[int]:
  if allowed_players is None:
    return list(range(len(row['players'])))

  matching_indices: list[int] = []
  raw = row.get('raw')
  for i, player in enumerate(row['players']):
    code = nametags.name_from_metadata(player, raw=raw)
    name = nametags.normalize_name(code)
    if name not in allowed_players:
      continue

    allowed_chars = allowed_players[name]
    character = Character(player['character'])
    if allowed_chars is None or character in allowed_chars:
      matching_indices.append(i)

  return matching_indices


def is_valid_replay(
    row: dict,
    *,
    doubles_only: bool,
    allowed_players: Optional[Dict[str, Optional[set[Character]]]],
) -> bool:
  if not row.get('is_training'):
    return False

  if doubles_only and not row.get('is_teams'):
    return False

  for player in row['players']:
    if Character(player['character']) in _EXCLUDED_CHARACTERS:
      return False

  if allowed_players is None:
    return True

  return bool(_matching_allowed_player_indices(
      row,
      allowed_players=allowed_players,
  ))

def _summarize(rows: list[dict]) -> None:
  _summarize_rows(rows, only_allowed_main_players=False)


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
  for (name, character), count in player_character_counts.most_common(n=300):
    print(f"{name} ({character}): {count}")


def _summarize_singles(rows: list[dict]) -> None:
  _summarize_singles_rows(rows, only_allowed_main_players=False)


def _summarize_singles_rows(
    rows: list[dict],
    *,
    only_allowed_main_players: bool,
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
  for (name, character), count in singles_player_character_counts.most_common(n=300):
    print(f"{name} ({character}): {count}")


def _summarize_selected_players(
    rows: list[dict],
    *,
    allowed_players: Dict[str, Optional[set[Character]]],
) -> None:
  replay_counts = collections.Counter({
      name: 0 for name in sorted(allowed_players)
  })

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
  print("Selected player replay counts:")
  for name, count in replay_counts.items():
    print(f"{name}: {count}")


def build_meta(
    root: str | os.PathLike[str],
    *,
    doubles_only: Optional[bool] = None,
    winner_only: Optional[bool] = None,
    make_tar: Optional[bool] = None,
    allowed_players_file: Optional[str] = None,
    allowed_players: Optional[Dict[str, Optional[Iterable[Character]]]] = None,
    quiet: bool = False,
) -> list[dict]:
  root_path = os.fspath(root)

  with open(os.path.join(root_path, 'parsed.pkl'), 'rb') as f:
    rows = pickle.load(f)

  doubles_only = DOUBLES_ONLY.value if doubles_only is None else doubles_only
  winner_only = WINNER_ONLY.value if winner_only is None else winner_only
  make_tar = MAKE_TAR.value if make_tar is None else make_tar

  if allowed_players_file is None:
    allowed_players_file = ALLOWED_PLAYERS_FILE.value

  file_allowed_players = _load_allowed_players_file(allowed_players_file)
  allowed_players = _normalize_allowed_players(allowed_players)
  if file_allowed_players is not None:
    allowed_players = file_allowed_players
  elif allowed_players is None and hasattr(nametags, 'ALLOWED_PLAYERS'):
    allowed_players = _normalize_allowed_players(nametags.ALLOWED_PLAYERS)

  total_rows = len(rows)
  filtered_rows: list[dict] = []
  for row in rows:
    if not is_valid_replay(
        row,
        doubles_only=doubles_only,
        allowed_players=allowed_players,
    ):
      continue

    row = dict(row)
    if allowed_players is None:
      row.pop('allowed_main_player_indices', None)
    else:
      row['allowed_main_player_indices'] = _matching_allowed_player_indices(
          row,
          allowed_players=allowed_players,
      )
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
    if allowed_players is not None:
      _summarize_rows(rows, only_allowed_main_players=True)
      if not doubles_only:
        _summarize_singles_rows(rows, only_allowed_main_players=True)
      _summarize_selected_players(rows, allowed_players=allowed_players)
    else:
      _summarize(rows)
      if not doubles_only:
        _summarize_singles(rows)

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
