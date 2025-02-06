"""The final step of dataset creation.

python slippi_db/scripts/make_local_dataset.py --root=Root
"""

import collections
import os
import pickle
import json
import tarfile
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

def is_valid_replay(row: dict):
  #print(row)
  if not row.get('is_training'):
    return False

  if DOUBLES_ONLY.value and not row.get('is_teams'):
    return False

  for player in row['players']:
    if Character(player['character']) in [Character.WIREFRAME_MALE,
                                          Character.WIREFRAME_FEMALE,
                                          Character.GIGA_BOWSER,
                                          Character.SANDBAG,
                                          Character.UNKNOWN_CHARACTER]:
      return False

  return True#row.get('is_teams')

  '''found_required_players = 0
  team = -1
  for codes, character in nametags.REQUIRED_PLAYERS:
    #print(codes, character)
    for player in row['players']:
      code = nametags.name_from_metadata(player)
      #print(code, codes, Character(player['character']), character)
      if code in codes and Character(player['character']) == character:
        found_required_players += 1
        if team == -1:
          team = player['team']
        elif team != player['team']:
          return False'''
  
  for player in row['players']:
    code = nametags.name_from_metadata(player)
    name = nametags.normalize_name(code)
    if name in nametags.ALLOWED_PLAYERS and (nametags.ALLOWED_PLAYERS[name] is None or Character(player['character']) in nametags.ALLOWED_PLAYERS[name]):
      return True

  #print(found_required_players == len(nametags.REQUIRED_PLAYERS))
  #return found_required_players == len(nametags.REQUIRED_PLAYERS)
  return False

def main(_):
  with open(os.path.join(ROOT.value, 'parsed.pkl'), 'rb') as f:
    rows = pickle.load(f)

  # keep only training replays
  print(f"Found {len(rows)} replays.")
  rows = [row for row in rows if is_valid_replay(row)]
  print(f"{len(rows)} replays with is_training = True.")

  # keep only games with a winner
  # TODO: this throws away games that have a salty runback

  total_doubles_games = 0.0
  doubles_winner = 0.0
  for row in rows:
    if len(row['players']) == 4:
      total_doubles_games += 1
      if row.get('winner') is not None:
        doubles_winner += 1

  print(f"Found {len(rows)} training replays.")

  # aggregate the player/character pairings by frequency
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
      '''if not name in nametags.ALLOWED_PLAYERS:
        player_character_counts[("Unrecognized", Character(player['character']))] += 1
      else:'''
      player_character_counts[(name, Character(player['character']))] += 1
      character_counts[Character(player['character'])] += 1
      '''if not name in nametags.ALLOWED_PLAYERS:
        continue
      if nametags.ALLOWED_PLAYERS[name] is not None and Character(player['character']) not in nametags.ALLOWED_PLAYERS[name]:
        continue
      player_character_counts[(name, Character(player['character']))] += 1
      trajectory_count += 1'''

  # pretty print the player/character pairings
  print(f"Found {trajectory_count} trajectories.")
  print("Doubles games with winner: ", total_doubles_games, doubles_winner)
  print("Character trajectories:")
  for (character), count in character_counts.most_common():
    print(f"({character}): {count}")
  print("")
  print("Player/Character Pairings:")
  for (name, character), count in player_character_counts.most_common(n=200):
    print(f"{name} ({character}): {count}")

  print("\n")

  singles_player_character_counts = collections.Counter()
  singles_character_counts = collections.Counter()

  total_singles_games = 0.0
  singles_winner = 0.0
  for row in rows:
    if not row.get('is_teams'):
      total_singles_games += 1
      if row.get('winner') is not None:
        singles_winner += 1

  for row in rows:
    if row.get('is_teams'):
      continue

    for player in row['players']:
      code = nametags.name_from_metadata(player)
      name = nametags.normalize_name(code)

      trajectory_count += 1
      '''if not name in nametags.ALLOWED_PLAYERS:
        player_character_counts[("Unrecognized", Character(player['character']))] += 1
      else:'''
      singles_player_character_counts[(name, Character(player['character']))] += 1
      singles_character_counts[Character(player['character'])] += 1
      '''if not name in nametags.ALLOWED_PLAYERS:
        continue
      if nametags.ALLOWED_PLAYERS[name] is not None and Character(player['character']) not in nametags.ALLOWED_PLAYERS[name]:
        continue
      player_character_counts[(name, Character(player['character']))] += 1
      trajectory_count += 1'''

  # pretty print the player/character pairings
  print(f"Found total {trajectory_count} trajectories.")
  print("Singles games with winner: ", total_singles_games, singles_winner)
  print("Character trajectories:")
  for (character), count in singles_character_counts.most_common():
    print(f"({character}): {count}")
  print("")
  print("Player/Character Pairings:")
  for (name, character), count in singles_player_character_counts.most_common(n=100):
    print(f"{name} ({character}): {count}")

  if WINNER_ONLY.value:
    rows = [row for row in rows if row.get('winner') is not None]
    print(f"Filtered to {len(rows)} games with a winner.")

  return

  make_tar = MAKE_TAR.value

  if make_tar:
    tar = tarfile.open(os.path.join(ROOT.value, 'training.tar'), 'w')

  missing = collections.Counter()

  for row in tqdm.tqdm(rows, smoothing=0, unit='slp'):
    md5 = row['slp_md5']
    parsed_path = os.path.join(ROOT.value, 'Parsed', md5)
    if not os.path.isfile(parsed_path):
      missing[row['raw']] += 1

    if make_tar:
      tar.add(parsed_path, arcname='games/' + md5)

  print(f"Missing: {missing}")

  # write metadata
  meta_path = os.path.join(ROOT.value, 'meta.json')
  with open(meta_path, 'w') as f:
    json.dump(rows, f, indent=2)

  if make_tar:
    tar.add(meta_path, arcname='meta.json')
    tar.close()

if __name__ == '__main__':
  app.run(main)
