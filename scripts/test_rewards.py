import json
import os
import numpy as np
from absl import app

from slippi_ai.data import ReplayMeta, ReplayInfo, TrajectoryManager, swap_players
from slippi_ai import reward
from slippi_db import file_layout

def main(_):
    data_dir = '/media/kyle/Windows/Users/kyleh/git/slippi-ai/data/'
    with open(data_dir+"meta.json") as f:
        meta_rows: list[dict] = json.load(f)

    first_game = None

    count = 0

    for row in meta_rows:
        if row['is_teams'] and row['is_training']:
            first_game = row
            print("\n")
            print(row)
            #break

            replay_meta = ReplayMeta.from_metadata(first_game)
            parsed_dir = os.path.join(data_dir, 'Parsed')
            replay_path = file_layout.resolve_parquet_path(parsed_dir, replay_meta.slp_md5)

            def make_replay_info(port: int):
                teammate_index = 1
                other_ports = tuple(i for i in range(4) if i not in (port, teammate_index))
                return ReplayInfo(
                    replay_path,
                    port,
                    teammate_index,
                    replay_meta.p0.name,
                    replay_meta,
                    other_ports,
                )

            info_list = [
                ReplayInfo(
                    replay_path,
                    0,
                    1,
                    replay_meta.p0.name,
                    replay_meta,
                    tuple(i for i in range(4) if i not in (0, 1)),
                ),
                ReplayInfo(
                    replay_path,
                    1,
                    0,
                    replay_meta.p1.name,
                    replay_meta,
                    tuple(i for i in range(4) if i not in (1, 0)),
                ),
                ReplayInfo(
                    replay_path,
                    2,
                    3,
                    replay_meta.p1.name,
                    replay_meta,
                    tuple(i for i in range(4) if i not in (2, 3)),
                ),
                ReplayInfo(
                    replay_path,
                    3,
                    2,
                    replay_meta.p1.name,
                    replay_meta,
                    tuple(i for i in range(4) if i not in (3, 2)),
                ),
            ]
            
            '''info_list = [ReplayInfo(replay_path, 0, 2, replay_meta.p0.name, replay_meta),
                            ReplayInfo(replay_path, 1, 3, replay_meta.p1.name, replay_meta)]'''

            '''for info in info_list:
                traj = TrajectoryManager([info], 65, 1, True)
                game = traj.load_game(info)

                rewards = reward.compute_rewards(game)

                #with np.printoptions(threshold=np.inf):
                #    print(rewards)

                print(rewards.shape)
                print(np.sum(rewards))
                print(replay_meta)'''
            
            '''info = info_list[0]
            traj = TrajectoryManager([info], 65, 1, True)
            game = traj.load_game(info)
            #game = swap_players(game, info)

            rewards = reward.compute_rewards(game)

            #with np.printoptions(threshold=np.inf):
            #    print(rewards)

            #print(rewards.shape)
            print("port, teammate port", info.main_player_index, info.teammate_index)
            print(np.sum(rewards))
            # get the count of frames where player is dead
            for i in range(4):
                player = getattr(game, f'p{i}')
                print(np.sum(player.is_dead))

            print(replay_meta)'''

            for info in info_list:
                traj = TrajectoryManager([info], 65, 1, True)
                game = traj.load_game(info)
                #game = swap_players(game, info)

                rewards = reward.compute_rewards(game, ledge_grab_penalty=0.00, stalling_penalty=0.02)

                '''with np.printoptions(threshold=np.inf):
                    print(rewards)
                    #print(game.p0.stocks_left)'''

                #print(rewards.shape)
                print("port, teammate port", info.main_player_index, info.teammate_index)
                print(np.sum(rewards))
                # get the count of frames where player is dead
                print(np.sum(game.p0.is_dead))
                print(np.sum(game.p1.is_dead))

                # number of frames where is_teams is true
                #print(np.sum(game.is_teams))

                print(replay_meta)
        

            count += 1
            if count > 2:
                break
    

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None
  app.run(main)

'''

Game(p0=Player(percent=array([ 0,  0,  0, ..., 53, 53, 53], dtype=uint16), facing=array([ True,  True,  True, ..., False, False, False]), x=array([-42.     , -42.     , -42.     , ...,  78.1101 ,  78.17003, 78.22064], dtype=float32), y=array([  5.      ,   5.      ,   5.      , ..., -66.03665 , -65.71422 , -65.441925], dtype=float32), action=array([322, 322, 322, ...,  90, 365, 365], dtype=uint16), invulnerable=array([False, False, False, ..., False,  True, False]), character=array([1, 1, 1, ..., 1, 1, 1], dtype=uint8), jumps_left=array([1, 1, 1, ..., 1, 1, 1], dtype=uint8), shield_strength=array([60., 60., 60., ..., 60., 60., 60.], dtype=float32), on_ground=array([False, False, False, ..., False, False, False]), is_dead=array([False, False, False, ..., False, False, False]), controller=Controller(main_stick=Stick(x=array([0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5], dtype=float32), y=array([0.5, 0.5, 0.5, ..., 0. , 0. , 0. ], dtype=float32)), c_stick=Stick(x=array([0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5], dtype=float32), y=array([0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5], dtype=float32)), shoulder=array([0., 0., 0., ..., 0., 0., 0.], dtype=float32), buttons=Buttons(A=array([False, False, False, ..., False, False, False]), B=array([False, False, False, ..., False,  True,  True]), X=array([False, False, False, ..., False, False, False]), Y=array([False, False, False, ..., False, False, False]), Z=array([False, False, False, ..., False, False, False]), L=array([False, False, False, ..., False, False, False]), R=array([False, False, False, ..., False, False, False]), D_UP=array([False, False, False, ..., False, False, False])))), 
     p1=Player(percent=array([ 0,  0,  0, ..., 38, 38, 38], dtype=uint16), facing=array([ True,  True,  True, ...,  True,  True,  True]), x=array([-42.      , -42.      , -42.      , ...,  66.00788 ,  66.58788 , 67.167885], dtype=float32), y=array([ 26.6    ,  26.6    ,  26.6    , ..., -49.12511, -52.12511, -55.12511], dtype=float32), action=array([322, 322, 322, ...,  66,  66,  66], dtype=uint16), invulnerable=array([False, False, False, ..., False, False, False]), character=array([7, 7, 7, ..., 7, 7, 7], dtype=uint8), jumps_left=array([1, 1, 1, ..., 1, 1, 1], dtype=uint8), shield_strength=array([60., 60., 60., ..., 60., 60., 60.], dtype=float32), on_ground=array([False, False, False, ..., False, False, False]), is_dead=array([False, False, False, ..., False, False, False]), controller=Controller(main_stick=Stick(x=array([0.5   , 0.5   , 0.5   , ..., 0.8625, 0.8625, 0.8625], dtype=float32), y=array([0.5   , 0.5   , 0.5   , ..., 0.1625, 0.1625, 0.1625], dtype=float32)), c_stick=Stick(x=array([0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5], dtype=float32), y=array([0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5], dtype=float32)), shoulder=array([0., 0., 0., ..., 0., 0., 0.], dtype=float32), buttons=Buttons(A=array([False, False, False, ..., False, False, False]), B=array([False, False, False, ..., False, False, False]), X=array([False, False, False, ..., False, False, False]), Y=array([False, False, False, ..., False, False, False]), Z=array([False, False, False, ..., False, False, False]), L=array([False, False, False, ..., False, False, False]), R=array([False, False, False, ..., False, False, False]), D_UP=array([False, False, False, ..., False, False, False])))), 
     p2=Player(percent=array([0, 0, 0, ..., 0, 0, 0], dtype=uint16), facing=array([False, False, False, ..., False, False, False]), x=array([42., 42., 42., ...,  0.,  0.,  0.], dtype=float32), y=array([28., 28., 28., ...,  0.,  0.,  0.], dtype=float32), action=array([322., 322., 322., ...,   0.,   0.,   0.]), invulnerable=array([False, False, False, ...,  True,  True,  True]), character=array([0., 0., 0., ..., 0., 0., 0.]), jumps_left=array([1., 1., 1., ..., 0., 0., 0.]), shield_strength=array([60., 60., 60., ...,  0.,  0.,  0.], dtype=float32), on_ground=array([False, False, False, ..., False, False, False]), is_dead=array([False, False, False, ...,  True,  True,  True]), controller=Controller(main_stick=Stick(x=array([0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5], dtype=float32), y=array([0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5], dtype=float32)), c_stick=Stick(x=array([0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5], dtype=float32), y=array([0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5], dtype=float32)), shoulder=array([0., 0., 0., ..., 0., 0., 0.], dtype=float32), buttons=Buttons(A=array([False, False, False, ..., False, False, False]), B=array([False, False, False, ..., False, False, False]), X=array([False, False, False, ..., False, False, False]), Y=array([False, False, False, ..., False, False, False]), Z=array([False, False, False, ..., False, False, False]), L=array([False, False, False, ..., False, False, False]), R=array([False, False, False, ..., False, False, False]), D_UP=array([False, False, False, ..., False, False, False])))),
     p3=Player(percent=array([  0,   0,   0, ..., 145, 145, 145], dtype=uint16), facing=array([False, False, False, ..., False, False, False]), x=array([ 42.      ,  42.      ,  42.      , ..., 114.858406, 114.858406, 114.858406], dtype=float32), y=array([ 5.      ,  5.      ,  5.      , ..., 71.7327  , 70.979935, 70.979935], dtype=float32), action=array([322, 322, 322, ...,   4,   4,   4], dtype=uint16), invulnerable=array([False, False, False, ..., False, False, False]), character=array([7, 7, 7, ..., 7, 7, 7], dtype=uint8), jumps_left=array([1, 1, 1, ..., 1, 1, 1], dtype=uint8), shield_strength=array([60., 60., 60., ..., 60., 60., 60.], dtype=float32), on_ground=array([False, False, False, ..., False, False, False]), is_dead=array([False, False, False, ..., False, False, False]), controller=Controller(main_stick=Stick(x=array([0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5], dtype=float32), y=array([0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5], dtype=float32)), c_stick=Stick(x=array([0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5], dtype=float32), y=array([0.5, 0.5, 0.5, ..., 0.5, 0.5, 0.5], dtype=float32)), shoulder=array([0., 0., 0., ..., 0., 0., 0.], dtype=float32), buttons=Buttons(A=array([False, False, False, ..., False, False, False]), B=array([False, False, False, ..., False, False, False]), X=array([False, False, False, ..., False, False, False]), Y=array([False, False, False, ..., False, False, False]), Z=array([False, False, False, ..., False, False, False]), L=array([False, False, False, ..., False, False, False]), R=array([False, False, False, ..., False, False, False]), D_UP=array([False, False, False, ..., False, False, False])))), stage=array([6, 6, 6, ..., 6, 6, 6], dtype=uint8))

'''
