"""Test a trained model."""

import collections
import json
import logging

from absl import app
from absl import flags
import fancyflags as ff

from slippi_ai import eval_lib, types, utils
from slippi_ai import dolphin as dolphin_lib
from slippi_db.parse_libmelee import get_controller

PLAYER = ff.DEFINE_dict('player', **eval_lib.PLAYER_FLAGS)

dolphin_flags = dolphin_lib.DOLPHIN_FLAGS.copy()
dolphin_flags.update(
    online_delay=ff.Integer(15),
    #connect_code=ff.String(None, required=True),
    user_json_path=ff.String(None, required=True),
    blocking_input=ff.Boolean(True),
    disable_audio=ff.Boolean(True),
    teams_connect_code=ff.String(None, required=True),
)
DOLPHIN = ff.DEFINE_dict('dolphin', **dolphin_flags)

CHECK_INPUTS = flags.DEFINE_boolean('check_inputs', False, 'Check inputs.')

RUNTIME = flags.DEFINE_integer('runtime', None, 'Runtime in seconds.')

TEAM = flags.DEFINE_integer('team_id', 0, 'Team ID')

FLAGS = flags.FLAGS

def main(_):
  eval_lib.disable_gpus()

  port = 1

  player = eval_lib.get_player(**PLAYER.value)

  print("team set to ", TEAM.value)

  dolphin = dolphin_lib.Dolphin(
      players={port: player},
      desired_teams={1: TEAM.value},
      **DOLPHIN.value,
  )

  agent: eval_lib.Agent

  # Warm up agent before starting game to prevent initial hiccup.
  if isinstance(player, dolphin_lib.AI):
    agent = eval_lib.build_agent(
        controller=dolphin.controllers[port],
        opponent_port=None,  # will be set later
        teammate_port=None,
        console_delay=DOLPHIN.value['online_delay'],
        run_on_cpu=True,
        **PLAYER.value['ai'],
    )

    eval_lib.update_character(player, agent.config)

  # Start game
  with open(DOLPHIN.value['user_json_path']) as f:
      user_json = json.load(f)

  run_agent(agent, dolphin, user_json['connectCode'])

def run_agent(agent: eval_lib.Agent, 
              dolphin: dolphin_lib.Dolphin, 
              connect_code: str):
  gamestate = dolphin.step()

  def set_player_ports(gamestate):
    actual_port = getattr(gamestate, 'local_player_port', None)
    if actual_port not in gamestate.players:
      matching_ports = [
          port for port, player in gamestate.players.items()
          if player.connectCode == connect_code
      ]
      if len(matching_ports) != 1:
        raise RuntimeError(
            f"Could not uniquely identify bot port for {connect_code}: "
            f"{matching_ports}")
      actual_port = matching_ports[0]
    teammate_port = 1
    for port, player in gamestate.players.items():
      if port == actual_port:
        continue
      if player.team_id == gamestate.players[actual_port].team_id:
        teammate_port = port
        break

    agent.players = (int(actual_port), int(teammate_port))
    agent.players += tuple(p for p in (1, 2, 3, 4) if p not in agent.players)
    agent.teammate_port = teammate_port
    return actual_port

  actual_port = set_player_ports(gamestate)

  # Main loop
  agent.start()

  try:
    num_frames = 0

    while True:
      if gamestate.frame == -123:
        actual_port = set_player_ports(gamestate)
        action_queue = collections.deque(
            [None] * (1 + dolphin.console.online_delay))
        print("starting game with ports: ", agent.players)

      # "step" to the next frame
      prev_frame = gamestate.frame
      gamestate = dolphin.step()
      if CHECK_INPUTS.value and gamestate.frame != -123:
        assert gamestate.frame == prev_frame + 1

      action: types.Controller = agent.step(gamestate).controller_state
      action = utils.map_nt(lambda x: x[0], action)
      action_queue.appendleft(action)

      expected: types.Controller = action_queue.pop()
      if expected is None:
        continue

      if gamestate.frame < 0:
        continue

      if actual_port in gamestate.players:
        observed = agent._agent.embed_controller.from_state(
            get_controller(gamestate.players[actual_port].controller_state))

        # deadzone can change observed stick values
        if observed.buttons != expected.buttons:
          frame = gamestate.frame + 123
          logging.error(f'Wrong controller seen on frame {frame}')

      num_frames += 1

  finally:
    agent.stop()
    dolphin.stop()

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None
  app.run(main)
