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

ROOT = flags.DEFINE_string('root', None, 'root directory', required=True)
WINNER_ONLY = flags.DEFINE_boolean(
  'winner_only', True, 'only keep games that have a winner')

MAKE_TAR = flags.DEFINE_boolean('tar', False, 'Create dataset tar archive')

DOUBLES_ONLY = flags.DEFINE_boolean(
  'doubles_only', True, 'only use doubles games')


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

  for player in row['players']:
    code = nametags.name_from_metadata(player)
    name = nametags.normalize_name(code)
    if name in allowed_players:
      allowed_chars = allowed_players[name]
      character = Character(player['character'])
      if allowed_chars is None or character in allowed_chars:
        return True

  return False

def _summarize(rows: list[dict]) -> None:
  player_character_counts = collections.Counter()
  character_counts = collections.Counter()
  trajectory_count = 0

  for row in rows:
    if len(row['players']) != 4:
      continue

    for player in row['players']:
      code = nametags.name_from_metadata(player)
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
  for (name, character), count in player_character_counts.most_common(n=100):
    print(f"{name} ({character}): {count}")


def _summarize_singles(rows: list[dict]) -> None:
  singles_player_character_counts = collections.Counter()
  singles_character_counts = collections.Counter()
  trajectory_count = 0

  for row in rows:
    if row.get('is_teams'):
      continue

    for player in row['players']:
      code = nametags.name_from_metadata(player)
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
  for (name, character), count in singles_player_character_counts.most_common(n=100):
    print(f"{name} ({character}): {count}")


def build_meta(
    root: str | os.PathLike[str],
    *,
    doubles_only: Optional[bool] = None,
    winner_only: Optional[bool] = None,
    make_tar: Optional[bool] = None,
    allowed_players: Optional[Dict[str, Optional[Iterable[Character]]]] = None,
    quiet: bool = False,
) -> list[dict]:
  root_path = os.fspath(root)

  with open(os.path.join(root_path, 'parsed.pkl'), 'rb') as f:
    rows = pickle.load(f)

  doubles_only = DOUBLES_ONLY.value if doubles_only is None else doubles_only
  winner_only = WINNER_ONLY.value if winner_only is None else winner_only
  make_tar = MAKE_TAR.value if make_tar is None else make_tar

  allowed_players = _normalize_allowed_players(allowed_players)
  if allowed_players is None and hasattr(nametags, 'ALLOWED_PLAYERS'):
    allowed_players = _normalize_allowed_players(nametags.ALLOWED_PLAYERS)

  total_rows = len(rows)
  rows = [
      row for row in rows
      if is_valid_replay(
          row,
          doubles_only=doubles_only,
          allowed_players=allowed_players,
      )
  ]

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
    parsed_path = os.path.join(root_path, 'Parsed', md5)
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
