import argparse
import json
from collections import Counter
from pathlib import Path

import melee
import numpy as np
import peppi_py
import tree

from scripts.analyze_jax_dolphin_replay import _perspective_game
from slippi_ai import data
from slippi_ai import eval_lib
from slippi_ai import embed as tf_embed
from slippi_ai import policies as tf_policies
from slippi_ai import utils
from slippi_ai.controller_lib import neutral_controller
from slippi_db import parse_peppi
from slippi_db import parse_libmelee


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--replay', required=True)
  parser.add_argument('--model-path', default='models/imitation_v19.pkl')
  parser.add_argument('--frames', type=int, default=1024)
  parser.add_argument('--start-frame', type=int, default=123)
  parser.add_argument('--source', choices=('peppi', 'libmelee'), default='peppi')
  parser.add_argument('--sample-temperature', type=float, default=1.0)
  parser.add_argument(
      '--jax-param-dtype', choices=('float32', 'bfloat16'), default='float32')
  args = parser.parse_args()

  state = eval_lib.load_state(path=args.model_path)

  tf_agent = eval_lib.build_delayed_agent(
      state=state,
      batch_size=1,
      console_delay=0,
      platform='tf',
      compile=False,
      async_inference=False,
      name='',
      sample_temperature=args.sample_temperature,
  )
  jax_agent = eval_lib.build_delayed_agent(
      state=state,
      batch_size=1,
      console_delay=0,
      platform='jax',
      compile=False,
      jax_param_dtype=args.jax_param_dtype,
      name='',
      sample_temperature=args.sample_temperature,
  )

  results = []
  if args.source == 'peppi':
    raw = peppi_py.read_slippi(args.replay)
    ports = list(raw.frames.ports)
    if len(ports) != 2:
      raise ValueError(f'expected singles replay with 2 ports, got {len(ports)}')

    players = [parse_peppi._player_from_port(port) for port in ports]
    length = min(len(players[0].x), int(args.frames))
    stage = melee.enums.to_internal_stage(raw.start.stage)
    for self_index, opponent_index in ((0, 1), (1, 0)):
      game = _perspective_game(
          self_player=players[self_index],
          opponent_player=players[opponent_index],
          stage=stage,
          length=length,
      )
      recorded = tf_agent._policy.controller_embedding.from_state(
          players[self_index].controller)
      recorded = utils.map_nt(lambda x: np.asarray(x)[:length], recorded)
      results.append(
          _compare_perspective_unroll(
              tf_agent=tf_agent,
              jax_agent=jax_agent,
              game=game,
              recorded=recorded,
              length=length,
              start_frame=args.start_frame,
          ))
  else:
    games_by_perspective = _libmelee_perspective_games(args.replay, args.frames)
    length = min(len(games_by_perspective[0].stage), len(games_by_perspective[1].stage))
    stage = melee.Stage(int(games_by_perspective[0].stage[0]))
    for game in games_by_perspective:
      game = utils.map_nt(lambda x, length=length: np.asarray(x)[:length], game)
      recorded = tf_agent._policy.controller_embedding.from_state(game.p0.controller)
      results.append(
          _compare_perspective_unroll(
              tf_agent=tf_agent,
              jax_agent=jax_agent,
              game=game,
              recorded=recorded,
              length=length,
              start_frame=args.start_frame,
          ))

  print(json.dumps({
      'replay': args.replay,
      'model_path': args.model_path,
      'frames': length,
      'start_frame': args.start_frame,
      'stage': stage.name,
      'source': args.source,
      'jax_param_dtype': args.jax_param_dtype,
      'results': results,
  }, indent=2, sort_keys=True))


def _libmelee_perspective_games(replay: str, frames: int):
  console = melee.Console(
      is_dolphin=False,
      allow_old_version=True,
      path=str(Path(replay)),
  )
  console.connect()
  games = [[], []]
  ports = None
  try:
    while len(games[0]) < frames:
      gamestate = console.step()
      if gamestate is None:
        break
      if not gamestate.players:
        continue
      if ports is None:
        ports = sorted(gamestate.players)
        if len(ports) != 2:
          raise ValueError(f'expected singles replay with 2 ports, got {ports}')
      if any(port not in gamestate.players for port in ports):
        continue
      games[0].append(parse_libmelee.get_game(gamestate, ports=(ports[0], ports[1])))
      games[1].append(parse_libmelee.get_game(gamestate, ports=(ports[1], ports[0])))
  finally:
    console.stop()

  if not games[0]:
    raise ValueError(f'no libmelee frames parsed from {replay}')
  return [_stack_games(g) for g in games]


def _stack_games(games):
  return utils.map_nt(lambda *xs: np.asarray(xs), *games)


def _compare_perspective_unroll(
    *,
    tf_agent,
    jax_agent,
    game,
    recorded,
    length: int,
    start_frame: int,
):
  action = _shifted_action(
      tf_agent._policy.controller_embedding,
      recorded,
      length,
  )
  state = utils.map_nt(lambda x: np.expand_dims(x, axis=1), game)
  action = utils.map_nt(lambda x: np.expand_dims(x, axis=1), action)
  name = np.full((length, 1), tf_agent.name_code[0], dtype=data.NAME_DTYPE)
  is_resetting = np.zeros((length, 1), dtype=np.bool_)
  is_resetting[0, 0] = True
  reward = np.zeros((length, 1), dtype=np.float32)

  tf_frames = data.Frames(
      state_action=tf_embed.StateAction(
          state=tf_agent._policy.embed_game.from_state(state),
          action=action,
          name=name,
      ),
      is_resetting=is_resetting,
      reward=reward,
  )
  jax_frames = data.Frames(
      state_action=data.StateAction(
          state=jax_agent._policy.network.encode_game(state),
          action=action,
          name=name,
      ),
      is_resetting=is_resetting,
      reward=reward,
  )

  tf_outputs, tf_distances, tf_final_state = _tf_policy_unroll_distances(
      tf_agent._policy, tf_frames, tf_agent._policy.initial_state(1))
  jax_outputs, jax_distances, jax_final_state = _jax_policy_unroll_distances(
      jax_agent._policy, jax_frames, jax_agent._policy.initial_state((1,)))

  window = slice(max(0, start_frame), None)
  per_component_distance_max = Counter()
  per_component_distance_mean = Counter()
  component_counts = Counter()
  _accum_component_stats(
      per_component_distance_max,
      per_component_distance_mean,
      component_counts,
      utils.map_nt(lambda x: np.asarray(x)[window], tf_distances),
      utils.map_nt(lambda x: np.asarray(x)[window], jax_distances),
  )
  distance_max, distance_sum, distance_count = _tree_abs_stats(
      utils.map_nt(lambda x: np.asarray(x)[window], tf_distances),
      utils.map_nt(lambda x: np.asarray(x)[window], jax_distances),
  )
  output_max, output_sum, output_count = _tree_abs_stats(
      np.asarray(tf_outputs)[window],
      np.asarray(jax_outputs)[window],
  )
  hidden_max, hidden_sum, hidden_count = _tree_abs_stats(
      tf_final_state,
      jax_final_state,
  )
  return {
      'distance_max_abs': distance_max,
      'distance_mean_abs': distance_sum / max(distance_count, 1),
      'output_max_abs': output_max,
      'output_mean_abs': output_sum / max(output_count, 1),
      'hidden_max_abs': hidden_max,
      'hidden_mean_abs': hidden_sum / max(hidden_count, 1),
      'per_component_distance_max_abs': dict(per_component_distance_max),
      'per_component_distance_mean_abs': {
          k: per_component_distance_mean[k] / max(component_counts[k], 1)
          for k in per_component_distance_mean
      },
  }


def _shifted_action(controller_embedding, recorded, length: int):
  neutral = controller_embedding.from_state(
      neutral_controller([1]))
  return utils.map_nt(
      lambda n, r: np.concatenate(
          [np.asarray(n), np.asarray(r)[:max(0, length - 1)]], axis=0),
      neutral,
      recorded,
  )


def _tf_policy_unroll_distances(policy, frames, initial_state):
  embedded_inputs = policy.embed_state_action(frames.state_action)
  if policy._opponent_pooling.is_symmetrized():
    all_inputs = embedded_inputs
    swapped_all_inputs = policy._opponent_pooling.swap_opponents(embedded_inputs)
  else:
    all_inputs = policy._opponent_pooling(embedded_inputs)
    swapped_all_inputs = None

  outputs, final_state = policy.network.unroll(
      all_inputs[:-1], frames.is_resetting[:-1], initial_state)
  if swapped_all_inputs is not None:
    swapped_outputs, swapped_final_state = policy.network.unroll(
        swapped_all_inputs[:-1], frames.is_resetting[:-1], initial_state)
    outputs = tf_policies._mean_nest(outputs, swapped_outputs)
    final_state = tf_policies._mean_nest(final_state, swapped_final_state)

  action = frames.state_action.action
  prev_action = utils.map_nt(lambda t: t[:-1], action)
  next_action = utils.map_nt(lambda t: t[1:], action)
  return (
      outputs,
      policy.controller_head.distance(outputs, prev_action, next_action).distance,
      final_state,
  )


def _jax_policy_unroll_distances(policy, frames, initial_state):
  inputs = utils.map_nt(lambda t: t[:-1], frames.state_action)
  outputs, final_state = policy.network.unroll(
      inputs, frames.is_resetting[:-1], initial_state)
  action = frames.state_action.action
  prev_action = utils.map_nt(lambda t: t[:-1], action)
  next_action = utils.map_nt(lambda t: t[1:], action)
  return (
      outputs,
      policy.controller_head.distance(outputs, prev_action, next_action).distance,
      final_state,
  )


def _tree_abs_stats(a, b) -> tuple[float, float, int]:
  max_abs = 0.0
  total = 0.0
  count = 0
  for x, y in zip(tree.flatten(a), tree.flatten(b)):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
      raise ValueError(f'shape mismatch: {x.shape} != {y.shape}')
    if x.size:
      diff = np.abs(x - y)
      max_abs = max(max_abs, float(np.max(diff)))
      total += float(np.sum(diff))
      count += int(diff.size)
  return max_abs, total, count


def _accum_component_stats(max_counter, mean_counter, count_counter, a, b):
  def visit(prefix, x, y):
    if isinstance(x, tuple) and hasattr(x, '_fields'):
      for field in x._fields:
        visit(f'{prefix}.{field}' if prefix else field, getattr(x, field), getattr(y, field))
      return
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size:
      diff = np.abs(x - y)
      max_counter[prefix] = max(max_counter[prefix], float(np.max(diff)))
      mean_counter[prefix] += float(np.sum(diff))
      count_counter[prefix] += int(diff.size)

  visit('', a, b)


if __name__ == '__main__':
  main()
