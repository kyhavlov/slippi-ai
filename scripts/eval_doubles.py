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
import melee
import numpy as np

from slippi_ai import eval_lib, flag_utils, utils
from slippi_ai import dolphin as dolphin_lib
from slippi_ai.controller_lib import send_controller
from slippi_ai.rl import run_lib
from slippi_db.parse_libmelee import get_game

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

USE_GPU = flags.DEFINE_bool(
    'use_gpu',
    os.environ.get('SLIPPI_AI_EVAL_USE_GPU') == '1',
    'Use GPU for model inference. By default local eval preserves historical CPU-only behavior.')

FUSE_AI_INFERENCE = flags.DEFINE_bool(
    'fuse_ai_inference',
    False,
    'Run all local AI ports that share a checkpoint through one batched inference worker.')


def _doubles_player_order(port: int, teammate_port: int) -> tuple[int, ...]:
  order = (port, teammate_port)
  return order + tuple(p for p in PORTS if p not in order)


def _as_bool(value) -> bool:
  return bool(np.asarray(value).reshape(-1)[0])


def _is_game_over(game, gamestate) -> bool:
  left_team_out = _as_bool(game.p0.stocks_left == 0) and _as_bool(game.p1.stocks_left == 0)
  right_team_out = _as_bool(game.p2.stocks_left == 0) and _as_bool(game.p3.stocks_left == 0)
  return left_team_out or right_team_out or gamestate.frame >= 28799


def _fused_agent_key(agent_config: dict) -> tuple[tuple[str, str], ...]:
  ignored = {'async_inference', 'name', 'name_change_mode'}
  return tuple(
      sorted(
          (key, repr(value))
          for key, value in agent_config.items()
          if key not in ignored
      ))


def _group_ai_ports(ai_ports: list[int]) -> list[list[int]]:
  groups_by_key: dict[tuple[tuple[str, str], ...], list[int]] = {}
  for port in ai_ports:
    key = _fused_agent_key(dict(PLAYERS[port].value['ai']))
    groups_by_key.setdefault(key, []).append(port)
  return list(groups_by_key.values())


def _run_fused_loop(players: dict[int, dolphin_lib.Player]):
  ai_ports = [
      port for port in PORTS
      if isinstance(players[port], dolphin_lib.AI)
  ]
  if not ai_ports:
    raise ValueError('No AI ports were provided for fused inference.')

  agent_groups = []
  for group_ports in _group_ai_ports(ai_ports):
    agent_config = dict(PLAYERS[group_ports[0]].value['ai'])
    path = agent_config.pop('path', None)
    tag = agent_config.pop('tag', None)
    agent_config.pop('name', None)
    agent_config.pop('name_change_mode', None)

    if agent_config.pop('async_inference', False):
      logging.info(
          'Ignoring --p*.ai.async_inference=True for fused local eval; '
          'the local eval loop is already the batched inference worker.')

    names = [
        PLAYERS[port].value['ai']['name']
        for port in group_ports
    ]
    state = eval_lib.load_state(path=path, tag=tag)
    agent = eval_lib.build_delayed_agent(
        state=state,
        batch_size=len(group_ports),
        console_delay=DOLPHIN.value['online_delay'],
        name=names,
        async_inference=False,
        **agent_config,
    )
    if agent.batch_steps > agent.delay + 1:
      raise ValueError(
          f'agent.batch_steps={agent.batch_steps} exceeds delay slack '
          f'for policy delay={agent.delay} after console delay.')

    for port in group_ports:
      eval_lib.update_character(players[port], state['config'])

    logging.info(
        'Fused local eval group ports=%s batch_size=%d path=%s tag=%s '
        'jit_compile=%s',
        group_ports, len(group_ports), path, tag, agent_config.get('jit_compile'))
    agent_groups.append(dict(ports=group_ports, agent=agent))

  dolphin = dolphin_lib.Dolphin(
      players=players,
      desired_teams={1: 0, 2: 1, 3: 1, 4: 0},
      **dolphin_lib.DolphinConfig.kwargs_from_flags(DOLPHIN.value),
  )

  teammate_ports = dict(zip(PORTS, reversed(PORTS)))
  player_orders = {
      port: _doubles_player_order(port, teammate_ports[port])
      for port in ai_ports
  }
  dead_frames = {port: 0 for port in ai_ports}
  pressed_start = {port: False for port in ai_ports}
  step_timer = utils.Profiler()

  import traceback

  try:
    for group in agent_groups:
      group['agent'].start()
    while True:
      gamestate = dolphin.step()
      new_game = gamestate.frame == -123

      for group in agent_groups:
        group_ports = group['ports']
        agent = group['agent']
        games = []
        for port in group_ports:
          game = get_game(gamestate, ports=player_orders[port])
          games.append(utils.map_nt(lambda x: np.expand_dims(x, 0), game))

        batched_game = utils.map_nt(
            lambda *xs: np.concatenate(xs, axis=0), *games)
        needs_reset = np.full([len(group_ports)], new_game, dtype=np.bool_)

        with step_timer:
          sample_outputs = agent.step(batched_game, needs_reset)
        decoded = agent.embed_controller.decode(
            sample_outputs.controller_state)

        for index, port in enumerate(group_ports):
          controller = dolphin.controllers[port]
          game = games[index]
          if _is_game_over(game, gamestate):
            controller.release_all()
            continue

          action = utils.map_single_structure(lambda x: x[index], decoded)
          send_controller(controller, action)

          teammate_port = teammate_ports[port]
          if _as_bool(game.p0.is_dead):
            dead_frames[port] += 1
            teammate_has_stock = (
                teammate_port in gamestate.players and
                gamestate.players[teammate_port].stock > 1)
            if (
                not pressed_start[port] and dead_frames[port] >= 120
                and teammate_has_stock):
              logging.info(
                  'p%s is dead, stock stealing from %d', port, teammate_port)
              controller.press_button(melee.Button.BUTTON_START)
              pressed_start[port] = True
            elif pressed_start[port]:
              controller.release_button(melee.Button.BUTTON_START)
              pressed_start[port] = False
          else:
            dead_frames[port] = 0

      if gamestate.frame > 0 and gamestate.frame % (5 * 60) == 0:
        logging.info(f'step_time: {step_timer.mean_time():.3f}')
  except BaseException as e:
    print(f"exception: {repr(e)}\n{traceback.format_exc()}")
  finally:
    for group in agent_groups:
      group['agent'].stop()
    dolphin.stop()


def _run_unfused_loop(players: dict[int, dolphin_lib.Player]):
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


def main(_):
  if not USE_GPU.value:
    eval_lib.disable_gpus()

  players = {
      port: eval_lib.get_player(**player.value)
      for port, player in PLAYERS.items()
  }

  ai_count = sum(isinstance(player, dolphin_lib.AI) for player in players.values())
  if FUSE_AI_INFERENCE.value and ai_count > 1:
    _run_fused_loop(players)
  else:
    _run_unfused_loop(players)

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None
  app.run(main)
