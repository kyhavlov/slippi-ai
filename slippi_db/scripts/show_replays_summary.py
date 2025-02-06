import os
import time
from dataclasses import dataclass
import peppi_py
import melee
from operator import attrgetter

@dataclass
class PlayerStats():
  game_count: int
  character_games: dict[melee.Character, int]
  teammate_counts: dict[str, int]

def walk_directory(directory):
  game_count = 0
  player_aggregate_stats: dict[str, PlayerStats] = {}

  for dirpath, dirnames, filenames in os.walk(directory):
    print("Directory:", dirpath)
    for filename in filenames:
      try:
        game = peppi_py.read_slippi(os.path.join(dirpath, filename), skip_frames=True)
      except Exception as e:
        print("Error reading file:", os.path.join(dirpath, filename))
        print(e)
        continue

      

      if game.start['is_teams'] == False:
        continue

      game_count += 1

      for player in game.start['players']:
        code = player['netplay']['code']
        port = player['port']

        # look up player stats entry or create it
        p = player_aggregate_stats[code] if code in player_aggregate_stats else PlayerStats(0, {}, {})
        p.game_count += 1
        #character = game.frames[0]['ports'][port]['leader']['post']['character'].as_py()
        character = player['character']

        # increment character games
        if character in p.character_games:
          p.character_games[character] += 1
        else:
          p.character_games[character] = 1

        # update teammate stats
        for other_player in game.start['players']:
          if other_player['netplay']['code'] == code:
            continue
          if other_player['team']['color'] == player['team']['color']:
            teammate_code = other_player['netplay']['code']
            if teammate_code in p.teammate_counts:
              p.teammate_counts[teammate_code] += 1
            else:
              p.teammate_counts[teammate_code] = 1
        
        player_aggregate_stats[code] = p

      # print some stats
      if game_count % 500 == 0:
        print("Game count:", game_count)
        print("Player stats:")
        for code, stats in sorted(player_aggregate_stats.items(), reverse=True, key=lambda item: getattr(item[1], "game_count")):
          if stats.game_count < 100:
            continue
          print("Code:", code)
          print("Games played:", stats.game_count)
          print("Character games:", stats.character_games)
          print("Teammate counts:", sorted(stats.teammate_counts.items(), reverse=True, key=lambda item: item[1]))
        print("")

      '''if game_count == 20:
        print(game.start['players'])
        print(game.frames[0]['ports']['P2']['leader']['post']['character'])
        time.sleep(100)'''
      #for player in game.metadata.players:
      #  print(player)

walk_directory("data/replays/enzyme")