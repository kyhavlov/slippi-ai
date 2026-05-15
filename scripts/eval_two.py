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
import melee
import numpy as np

from absl import app
from absl import flags
import fancyflags as ff

from slippi_ai import eval_lib, flag_utils, utils
from slippi_ai import dolphin as dolphin_lib
from slippi_ai.controller_lib import send_controller
from slippi_db.parse_libmelee import get_game

PORTS = (1, 2)

player_flags = utils.map_nt(lambda x: x, eval_lib.PLAYER_FLAGS)
player_flags['ai']['async_inference'] = ff.Boolean(True)

PLAYERS = {p: ff.DEFINE_dict(f"p{p}", **player_flags) for p in PORTS}
USE_GPU = flags.DEFINE_boolean('use_gpu', False, 'Use GPU for inference.')
FUSE_AI_INFERENCE = flags.DEFINE_bool(
    'fuse_ai_inference',
    False,
    'Run local AI ports that share a checkpoint through one batched inference worker.')

dolphin_config = dolphin_lib.DolphinConfig(
    headless=False,
    infinite_time=False,
    path=os.environ.get('DOLPHIN_PATH'),
    iso=os.environ.get('ISO_PATH'),
    blocking_input=True,
)
DOLPHIN = ff.DEFINE_dict(
    'dolphin', **flag_utils.get_flags_from_default(dolphin_config))

FLAGS = flags.FLAGS

def _as_bool(value) -> bool:
  return bool(np.asarray(value).reshape(-1)[0])


def _is_game_over(game, gamestate) -> bool:
  return (
      _as_bool(game.p0.stocks_left == 0)
      or _as_bool(game.p2.stocks_left == 0)
      or gamestate.frame >= 28799)


def _fused_agent_key(agent_config: dict) -> tuple[tuple[str, str], ...]:
  ignored = {'async_inference', 'name', 'name_change_mode'}
  return tuple(
      sorted(
          (key, repr(value))
          for key, value in agent_config.items()
          if key not in ignored))


def _group_ai_ports(players: dict[int, dolphin_lib.Player]) -> list[list[int]]:
  groups_by_key: dict[tuple[tuple[str, str], ...], list[int]] = {}
  for port in PORTS:
    if not isinstance(players[port], dolphin_lib.AI):
      continue
    key = _fused_agent_key(dict(PLAYERS[port].value['ai']))
    groups_by_key.setdefault(key, []).append(port)
  return list(groups_by_key.values())


def _run_fused_loop(players: dict[int, dolphin_lib.Player]):
  agent_groups = []
  for group_ports in _group_ai_ports(players):
    agent_config = dict(PLAYERS[group_ports[0]].value['ai'])
    path = agent_config.pop('path', None)
    tag = agent_config.pop('tag', None)
    agent_config.pop('name', None)
    agent_config.pop('name_change_mode', None)
    if agent_config.pop('async_inference', False):
      logging.info(
          'Ignoring --p*.ai.async_inference=True for fused local eval; '
          'the local eval loop is already the batched inference worker.')

    names = [PLAYERS[port].value['ai']['name'] for port in group_ports]
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
        'Fused local singles eval ports=%s batch_size=%d path=%s tag=%s',
        group_ports, len(group_ports), path, tag)
    agent_groups.append(dict(ports=group_ports, agent=agent))

  dolphin = dolphin_lib.Dolphin(
      players=players,
      **dolphin_lib.DolphinConfig.kwargs_from_flags(DOLPHIN.value),
  )

  step_timer = utils.Profiler()
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
          opponent_port = 1 if port == 2 else 2
          game = get_game(gamestate, ports=(port, opponent_port))
          games.append(utils.map_nt(lambda x: np.expand_dims(x, 0), game))

        batched_game = utils.map_nt(
            lambda *xs: np.concatenate(xs, axis=0), *games)
        needs_reset = np.full([len(group_ports)], new_game, dtype=np.bool_)

        with step_timer:
          sample_outputs = agent.step(batched_game, needs_reset)
        decoded = agent.embed_controller.decode(sample_outputs.controller_state)

        for index, port in enumerate(group_ports):
          controller = dolphin.controllers[port]
          game = games[index]
          if _is_game_over(game, gamestate):
            controller.release_all()
            continue
          action = utils.map_single_structure(lambda x: x[index], decoded)
          send_controller(controller, action)

      if gamestate.frame > 0 and gamestate.frame % (15 * 60) == 0:
        logging.info(f'step_time: {step_timer.mean_time():.3f}')
  finally:
    for group in agent_groups:
      group['agent'].stop()
    dolphin.stop()


def _run_unfused_loop(players: dict[int, dolphin_lib.Player]):
  agents: list[eval_lib.Agent] = []

  for port, opponent_port in zip(PORTS, reversed(PORTS)):
    player = players[port]
    if isinstance(player, dolphin_lib.AI):
      agent = eval_lib.build_agent(
          port=port,
          opponent_port=opponent_port,
          is_singles=True,
          console_delay=DOLPHIN.value['online_delay'],
          run_on_cpu=not USE_GPU.value,
          **PLAYERS[port].value['ai'],
      )
      agent.start()
      agents.append(agent)

      eval_lib.update_character(player, agent.config)

  dolphin = dolphin_lib.Dolphin(
      players=players,
      **dolphin_lib.DolphinConfig.kwargs_from_flags(DOLPHIN.value),
  )

  for agent in agents:
    agent.set_controller(dolphin.controllers[agent._port])

  step_timer = utils.Profiler()

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

      if gamestate.frame > 0 and gamestate.frame % (15 * 60) == 0:
        logging.info(f'step_time: {step_timer.mean_time():.3f}')
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
