"""Run a game between two trained agents, or vs a human player.

To run two agents against each other:

```shell
python scripts/eval_two.py \
  --dolphin.path=/path/to/slippi-dolphin \
  --dolphin.iso=/path/to/SSBM.iso \
  --p1.ai.path=/path/to/agent1 \
  --p2.ai.path=/path/to/agent2
```

To run an agent against a human player in port 1:

```shell
python scripts/eval_two.py \
  --dolphin.path=/path/to/slippi-dolphin \
  --dolphin.iso=/path/to/SSBM.iso \
  --p1.type=human \
  --p2.ai.path=/path/to/agent
```

"""

import logging
import os

from absl import app
from absl import flags
import fancyflags as ff

from slippi_ai import eval_lib, flag_utils, utils
from slippi_ai import dolphin as dolphin_lib
from slippi_ai.rl import run_lib

PORTS = (1, 2, 3, 4)

player_flags = utils.map_nt(lambda x: x, eval_lib.PLAYER_FLAGS)
player_flags['ai']['async_inference'] = ff.Boolean(True)

PLAYERS = {p: ff.DEFINE_dict(f"p{p}", **player_flags) for p in PORTS}

dolphin_config = dolphin_lib.DolphinConfig(
    headless=True,
    infinite_time=False,
    path=os.environ.get('DOLPHIN_PATH'),
    iso=os.environ.get('ISO_PATH'),
    save_replays=True,
    disable_audio=False,
    blocking_input=True,
    #replay_dir="/mnt/c/Users/kyleh/git/slippi-ai/bot-replays",
    #slippi_port=51441,
)
DOLPHIN = ff.DEFINE_dict(
    'dolphin', **flag_utils.get_flags_from_default(dolphin_config))

FLAGS = flags.FLAGS

CHARACTER_WEIGHTINGS = run_lib.CHARACTER_WEIGHTINGS

def main(_):
  eval_lib.disable_gpus()

  players = {
      port: eval_lib.get_player(**player.value)
      for port, player in PLAYERS.items()
  }

  agents: list[eval_lib.Agent] = []

  for port, teammate_port in zip(PORTS, reversed(PORTS)):
    player = players[port]
    if isinstance(player, dolphin_lib.AI):
      '''if port == 1 or port == 4:'''
      #player.character_weight_table = CHARACTER_WEIGHTINGS
      agent = eval_lib.build_agent(
          port=port,
          teammate_port=teammate_port,
          opponent_port=2 if port == 1 or port == 4 else 1,
          console_delay=DOLPHIN.value['online_delay'],
          **PLAYERS[port].value['ai'],
      )
      #player.character_weight_table = CHARACTER_WEIGHTINGS
      agent.start()
      agents.append(agent)

      eval_lib.update_character(player, agent.config)

  dolphin = dolphin_lib.Dolphin(
      players=players,
      desired_teams={1: 0, 2: 1, 3: 1, 4: 0},
      #dolphin_home_path="/tmp/legacydubstest",
      #tmp_home_directory=False,
      **dolphin_lib.DolphinConfig.kwargs_from_flags(DOLPHIN.value),
  )

  for agent in agents:
    agent.set_controller(dolphin.controllers[agent._port])

  step_timer = utils.Profiler()

  import traceback

  # Main loop
  try:
    while True:
      # "step" to the next frame
      gamestate = dolphin.step()

      # if gamestate.frame == -123: # initial frame
      #   controller.release_all()

      with step_timer:
        for agent in agents:
          agent.step(gamestate)

      if gamestate.frame > 0 and gamestate.frame % (5 * 60) == 0:
        logging.info(f'step_time: {step_timer.mean_time():.3f}')
        #for i, player in gamestate.players:
        #  logging.info(f'gamestate: {pformat(player)}')
        #for port, player in gamestate.players.items():
        #  logging.info(f'port {port} x: {player.position.x} y: {player.position.y}')

  except BaseException as e:
    print(f"exception: {repr(e)}\n{traceback.format_exc()}")
  finally:
    for agent in agents:
      agent.stop()
    dolphin.stop()

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None
  app.run(main)
