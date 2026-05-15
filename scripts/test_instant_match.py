#!/usr/bin/env python3

import argparse
import atexit
import logging
import signal
import time

import melee
import numpy as np

from slippi_ai import dolphin as dolphin_lib
from slippi_ai import eval_lib
from slippi_ai import utils
from slippi_ai.controller_lib import send_controller

PORTS = (1, 2, 3, 4)
TEAMMATE_PORTS = {1: 4, 2: 3, 3: 2, 4: 1}
DESIRED_TEAMS = {1: 0, 2: 1, 3: 1, 4: 0}


def _parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dolphin_path',
      default='/mnt/nvme0/projects/slippi-Ishiiruka/AppDir/usr/bin/dolphin-emu')
  parser.add_argument('--iso_path', default='SSBM.iso')
  parser.add_argument('--model_path', default='models/rl_doubles_v29_5300.pkl')
  parser.add_argument('--matches', type=int, default=3)
  parser.add_argument('--timeout_sec', type=float, default=300.0)
  parser.add_argument('--console_timeout_sec', type=float, default=60.0)
  parser.add_argument('--disable_gpus', action='store_true')
  parser.add_argument('--starting_stocks', type=int, default=0)
  parser.add_argument('--character_pool', default='fox,falco,marth,sheik')
  parser.add_argument('--stage_pool', default='battlefield,final_destination,yoshis_story')
  parser.add_argument('--mode', choices=('suicide', 'model'), default='suicide')
  parser.add_argument('--num_players', type=int, choices=(2, 4), default=2)
  return parser.parse_args()


def _stage_name(stage: melee.Stage) -> str:
  return melee.Stage(stage).name


def _suicide_step(controller: melee.Controller, player: melee.PlayerState, direction: float):
  controller.release_all()
  if player.off_stage:
    controller.tilt_analog(melee.Button.BUTTON_MAIN, direction, 0.0)
    return

  near_edge = abs(player.position.x) > 45
  controller.tilt_analog(melee.Button.BUTTON_MAIN, direction, 0.5)
  if player.on_ground and near_edge:
    controller.press_button(melee.Button.BUTTON_Y)


def main():
  args = _parse_args()
  logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
  if args.disable_gpus:
    eval_lib.disable_gpus()

  active_ports = PORTS[:args.num_players]
  state = None
  inferred_names = ['Player']
  shared_agent = None
  players = {
      port: dolphin_lib.AI(melee.Character.FOX)
      for port in active_ports
  }
  agents = []
  dolphin = None

  def cleanup():
    for agent in agents:
      try:
        agent.stop()
      except Exception:
        pass
    if shared_agent is not None:
      try:
        shared_agent.stop()
      except Exception:
        pass
    if dolphin is not None:
      try:
        dolphin.stop()
      except Exception:
        pass

  def _handle_signal(signum, _frame):
    cleanup()
    raise SystemExit(128 + signum)

  atexit.register(cleanup)
  signal.signal(signal.SIGINT, _handle_signal)
  signal.signal(signal.SIGTERM, _handle_signal)

  try:
    dolphin = dolphin_lib.Dolphin(
        path=args.dolphin_path,
        iso=args.iso_path,
        players=players,
        desired_teams=DESIRED_TEAMS if args.num_players == 4 else {},
        headless=True,
        render=False,
        blocking_input=True,
        console_timeout=args.console_timeout_sec,
        save_replays=False,
        infinite_time=False,
        starting_stocks=args.starting_stocks,
        instant_match=True,
        instant_match_character_pool=args.character_pool.split(','),
        instant_match_stage_pool=args.stage_pool.split(','),
    )

    allowed_characters = {
        name.strip().upper() for name in args.character_pool.split(',') if name.strip()
    }
    allowed_stages = {
        name.strip().upper().replace(' ', '_') for name in args.stage_pool.split(',') if name.strip()
    }

    observed_matches: list[tuple[str, tuple[str, ...]]] = []
    pending_match = False
    previous_frame: int | None = None
    start_time = time.time()
    last_log_time = start_time

    while len(observed_matches) < args.matches:
      if time.time() - start_time > args.timeout_sec:
        raise TimeoutError(
            f'Timed out after {args.timeout_sec:.1f}s with only '
            f'{len(observed_matches)} matches observed.')

      gamestate = dolphin.step()
      if args.mode == 'model':
        if shared_agent is None and gamestate.menu_state == melee.Menu.IN_GAME:
          state = eval_lib.load_state(path=args.model_path)
          inferred_names = eval_lib.get_name_from_rl_state(state) or ['Player']
          shared_agent = eval_lib.build_delayed_agent(
              state=state,
              batch_size=len(active_ports),
              console_delay=0,
              name=[
                  inferred_names[(port - 1) % len(inferred_names)]
                  for port in active_ports
              ],
              async_inference=False,
              run_on_cpu=True,
          )
          shared_agent.start()
        assert shared_agent is not None
        needs_reset = np.full([len(active_ports)], gamestate.frame == -123, dtype=bool)
        games = []
        for port in active_ports:
          if args.num_players == 2:
            players_for_view = (port, TEAMMATE_PORTS.get(port, 0) or (2 if port == 1 else 1))
          else:
            teammate_port = TEAMMATE_PORTS[port]
            players_for_view = (
                port,
                teammate_port,
                *tuple(p for p in active_ports if p not in (port, teammate_port)),
            )
          games.append(eval_lib.get_game(gamestate, ports=players_for_view))
        batched_game = utils.map_nt(lambda *xs: np.stack(xs, axis=0), *games)
        sample_outputs = shared_agent.step(batched_game, needs_reset)
        actions = sample_outputs.controller_state
        for idx, port in enumerate(active_ports):
          action = utils.map_nt(lambda x: x[idx], actions)
          action = shared_agent.embed_controller.decode(action)
          send_controller(dolphin.controllers[port], action)
      elif gamestate.menu_state == melee.Menu.IN_GAME:
        for port in active_ports:
          if port not in gamestate.players:
            dolphin.controllers[port].release_all()
            continue
          direction = 0.0 if port in (1, 3) else 1.0
          _suicide_step(dolphin.controllers[port], gamestate.players[port], direction)

      if (previous_frame is not None and gamestate.menu_state == melee.Menu.IN_GAME
          and gamestate.frame < previous_frame):
        pending_match = True
      previous_frame = gamestate.frame

      if gamestate.menu_state == melee.Menu.IN_GAME and not observed_matches:
        pending_match = True

      if pending_match and gamestate.menu_state == melee.Menu.IN_GAME:
        active_ports_gs = tuple(port for port in active_ports if port in gamestate.players)
        characters = tuple(gamestate.players[port].character for port in active_ports_gs)
        if (gamestate.stage != melee.Stage.NO_STAGE and
            all(character != melee.Character.UNKNOWN_CHARACTER for character in characters)):
          character_names = tuple(character.name for character in characters)
          stage_name = _stage_name(gamestate.stage)
          observed_matches.append((stage_name, character_names))
          pending_match = False
          logging.info(
              'match %d: stage=%s chars=%s',
              len(observed_matches), stage_name, character_names)

      now = time.time()
      if now - last_log_time >= 10.0:
        logging.info('waiting for matches... observed=%d current_frame=%d', len(observed_matches), gamestate.frame)
        last_log_time = now

    for stage_name, characters in observed_matches:
      if stage_name.upper() not in allowed_stages:
        raise AssertionError(f'Observed stage {stage_name} outside pool {sorted(allowed_stages)}')
      for character_name in characters:
        if character_name.upper() not in allowed_characters:
          raise AssertionError(
              f'Observed character {character_name} outside pool {sorted(allowed_characters)}')

    if len(set(observed_matches)) < 2:
      raise AssertionError(f'instant_match did not vary stage/characters: {observed_matches!r}')

    logging.info('observed matches: %s', observed_matches)
  finally:
    cleanup()


if __name__ == '__main__':
  main()
