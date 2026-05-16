import argparse
import json
import math
import time
from collections import defaultdict, deque
from pathlib import Path

import melee
import numpy as np

from slippi_ai import eval_lib
from slippi_ai import sim_env
from slippi_ai import utils


CHARACTER_NAMES = {
    melee.Character.FOX.value: 'fox',
    melee.Character.FALCO.value: 'falco',
}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model-path', default='models/rl_doubles_v27_11000.pkl')
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument('--completed-games', type=int, default=250)
  parser.add_argument('--fixed-steps', type=int, default=0)
  parser.add_argument('--warmup-steps', type=int, default=1)
  parser.add_argument('--length', type=int, default=256)
  parser.add_argument('--max-game-frames', type=int, default=28800)
  parser.add_argument('--sample-temperature', type=float, default=1.0)
  parser.add_argument('--run-on-cpu', action='store_true')
  parser.add_argument('--no-compile', action='store_true')
  parser.add_argument('--batch-steps', type=int, default=0)
  parser.add_argument('--preembed-state', action='store_true')
  parser.add_argument('--fast-path', action='store_true')
  parser.add_argument('--platform', choices=('tf', 'jax'), default='tf')
  parser.add_argument('--print-every', type=int, default=1000)
  args = parser.parse_args()

  model_path = Path(args.model_path)
  if not model_path.exists():
    raise FileNotFoundError(model_path)

  stages = _cycle_stages(args.batch_size)
  env = sim_env.SimBatchedEnvironment(
      num_envs=args.batch_size,
      length=args.length,
      stage=stages,
      character_pool='fox,falco',
      max_frame_id=args.max_game_frames - 123,
  )

  state = eval_lib.load_state(path=str(model_path))
  names = eval_lib.get_name_from_rl_state(state)
  agent = _build_agent(args, state, names)

  game_stats = GameStats(args.batch_size, stages)
  timings = defaultdict(float)
  counters = defaultdict(int)
  total_steps = 0
  measured_steps = 0
  start = time.perf_counter()
  measured_start = None

  try:
    if args.fast_path:
      spacing = _default_controller_spacing(state)
      output = env.current_game_batch(
          needs_reset=np.ones(args.batch_size, dtype=np.bool_))
      agent.step(output.game, output.needs_reset)
    else:
      spacing = None
      warmup_output = env.current_state(needs_reset=np.ones(args.batch_size, dtype=np.bool_))
      agent.step(_pack_games(warmup_output.gamestates, args.batch_size), _pack_reset(warmup_output.needs_reset))
      output = warmup_output

    while _should_continue(args, game_stats.total_games, total_steps):
      measuring = total_steps >= args.warmup_steps
      if measuring and measured_start is None:
        measured_start = time.perf_counter()

      iter_start = time.perf_counter()
      if args.fast_path:
        packed_game = output.game
        packed_reset = output.needs_reset
      else:
        packed_game = _pack_games(output.gamestates, args.batch_size)
        packed_reset = _pack_reset(output.needs_reset)
      elapsed = time.perf_counter() - iter_start
      if measuring:
        timings['pack_s'] += elapsed

      if args.preembed_state:
        embed_start = time.perf_counter()
        packed_game = agent._agent._policy.embed_game.from_state(packed_game)
        elapsed = time.perf_counter() - embed_start
        if measuring:
          timings['embed_s'] += elapsed

      policy_start = time.perf_counter()
      if args.fast_path and hasattr(agent, 'step_controller_state'):
        controller_state = agent.step_controller_state(packed_game, packed_reset)
        sample_outputs = None
      else:
        sample_outputs = agent.step(packed_game, packed_reset)
        controller_state = sample_outputs.controller_state
      elapsed = time.perf_counter() - policy_start
      if measuring:
        timings['policy_s'] += elapsed

      decode_start = time.perf_counter()
      if args.fast_path:
        invalid_actions = _count_invalid_encoded_actions(
            controller_state,
            axis_spacing=spacing[0],
            shoulder_spacing=spacing[1],
        )
      else:
        decoded = agent.embed_controller.decode(sample_outputs.controller_state)
        decoded = utils.map_single_structure(np.asarray, decoded)
        controllers = _split_controllers(decoded, args.batch_size)
        invalid_actions = _count_invalid_actions(controllers)
      elapsed = time.perf_counter() - decode_start
      if measuring:
        counters['invalid_actions'] += invalid_actions
        timings['decode_s'] += elapsed

      env_start = time.perf_counter()
      if args.fast_path:
        needs_reset = env.step_encoded(
            controller_state,
            axis_spacing=spacing[0],
            shoulder_spacing=spacing[1],
        )
        output = env.current_game_batch(needs_reset=needs_reset)
      else:
        output = env.step(controllers)
      elapsed = time.perf_counter() - env_start
      if measuring:
        timings['env_s'] += elapsed

      stat_start = time.perf_counter()
      if args.fast_path:
        game_stats.observe_packed(output.game, output.needs_reset, env.last_step_info)
      else:
        game_stats.observe(output, env.last_step_info)
      elapsed = time.perf_counter() - stat_start
      if measuring:
        timings['stats_s'] += elapsed
        measured_steps += 1

      total_steps += 1
      if args.print_every and total_steps % args.print_every == 0:
        elapsed = time.perf_counter() - start
        print(
            f'steps={total_steps} games={game_stats.total_games} '
            f'stockouts={game_stats.stockout_games} timeouts={game_stats.timeout_games} '
            f'env_steps_per_sec={args.batch_size * total_steps / elapsed:.1f}',
            flush=True,
        )
  finally:
    env.stop()

  elapsed = time.perf_counter() - start
  measured_elapsed = 0.0 if measured_start is None else time.perf_counter() - measured_start
  summary = game_stats.summary()
  summary['benchmark'] = {
      'model_path': str(model_path),
      'batch_size': args.batch_size,
      'length': args.length,
      'target_completed_games': args.completed_games,
      'max_game_frames': args.max_game_frames,
      'total_steps': total_steps,
      'env_steps': total_steps * args.batch_size,
      'elapsed_sec': elapsed,
      'env_steps_per_sec': (total_steps * args.batch_size) / elapsed,
      'ns_per_env_step': elapsed * 1e9 / max(1, total_steps * args.batch_size),
      'fixed_steps': args.fixed_steps,
      'warmup_steps': args.warmup_steps,
      'measured_steps': measured_steps,
      'measured_env_steps': measured_steps * args.batch_size,
      'measured_elapsed_sec': measured_elapsed,
      'measured_env_steps_per_sec': (measured_steps * args.batch_size) / max(measured_elapsed, 1e-9),
      'measured_ns_per_env_step': measured_elapsed * 1e9 / max(1, measured_steps * args.batch_size),
      'run_on_cpu': bool(args.run_on_cpu),
      'compile': not args.no_compile,
      'batch_steps': args.batch_steps,
      'preembed_state': args.preembed_state,
      'fast_path': args.fast_path,
      'platform': args.platform,
  }
  summary['timings_sec'] = dict(timings)
  summary['counters'] = dict(counters)
  print(json.dumps(summary, indent=2, sort_keys=True))


class GameStats:

  def __init__(self, batch_size: int, stages: np.ndarray):
    self.batch_size = int(batch_size)
    self.stages = stages
    self.frames = np.zeros(batch_size, dtype=np.int32)
    self.prev_percent = np.zeros((batch_size, 2), dtype=np.float32)
    self.damage_taken = np.zeros((batch_size, 2), dtype=np.float32)
    self.total_games = 0
    self.stockout_games = 0
    self.timeout_games = 0
    self.draw_games = 0
    self.nan_state_count = 0
    self.stuck_frame_count = 0
    self.last_frame_id = np.full(batch_size, -124, dtype=np.int32)
    self.by_character = {
        'fox': _empty_bucket(),
        'falco': _empty_bucket(),
    }
    self.by_stage = {
        stage.name: {
            'games': 0,
            'stockouts': 0,
            'timeouts': 0,
            'draws': 0,
            'fox_wins': 0,
            'falco_wins': 0,
            'length_sum': 0,
        }
        for stage in sim_env.supported_stages()
    }

  def observe(self, output, step_info: sim_env.SimStepInfo):
    game = output.gamestates[1]
    self._observe_game(game, output.needs_reset, step_info)

  def observe_packed(
      self,
      game: sim_env.Game,
      needs_reset: np.ndarray,
      step_info: sim_env.SimStepInfo,
  ):
    self._observe_game(game, needs_reset, step_info)

  def _observe_game(
      self,
      game: sim_env.Game,
      needs_reset: np.ndarray,
      step_info: sim_env.SimStepInfo,
  ):
    frame_id = step_info.terminal['frame_id']
    self.nan_state_count += int(
        np.isnan(game.p0.x[:self.batch_size]).sum()
        + np.isnan(game.p0.y[:self.batch_size]).sum()
        + np.isnan(game.p2.x[:self.batch_size]).sum()
        + np.isnan(game.p2.y[:self.batch_size]).sum()
    )
    advanced = frame_id > self.last_frame_id
    self.stuck_frame_count += int(np.logical_not(advanced).sum())
    self.last_frame_id = frame_id.copy()

    current_percent = np.stack([
        game.p0.percent[:self.batch_size].astype(np.float32),
        game.p2.percent[:self.batch_size].astype(np.float32),
    ], axis=1)
    self.damage_taken += np.maximum(current_percent - self.prev_percent, 0.0)
    self.prev_percent = current_percent
    self.frames += 1

    done_ids = np.flatnonzero(needs_reset[:self.batch_size])
    for lane in done_ids:
      self._finish_lane(int(lane), game, step_info.terminal[int(lane)])

  def _finish_lane(self, lane: int, game, terminal):
    fox_stocks = int(game.p0.stocks_left[lane])
    falco_stocks = int(game.p2.stocks_left[lane])
    timeout = bool(terminal['max_frame_reached'])
    stockout = bool(terminal['match_ended']) and not timeout
    winner = None
    if stockout:
      if fox_stocks > falco_stocks:
        winner = 'fox'
      elif falco_stocks > fox_stocks:
        winner = 'falco'

    self.total_games += 1
    self.stockout_games += int(stockout)
    self.timeout_games += int(timeout)
    self.draw_games += int(winner is None)

    stage_name = self.stages[lane].name
    stage_bucket = self.by_stage[stage_name]
    stage_bucket['games'] += 1
    stage_bucket['stockouts'] += int(stockout)
    stage_bucket['timeouts'] += int(timeout)
    stage_bucket['draws'] += int(winner is None)
    stage_bucket['length_sum'] += int(self.frames[lane])
    if winner == 'fox':
      stage_bucket['fox_wins'] += 1
    elif winner == 'falco':
      stage_bucket['falco_wins'] += 1

    for player_idx, character in enumerate(('fox', 'falco')):
      bucket = self.by_character[character]
      bucket['games'] += 1
      bucket['wins'] += int(winner == character)
      bucket['losses'] += int(winner is not None and winner != character)
      bucket['timeouts'] += int(timeout)
      bucket['draws'] += int(winner is None)
      bucket['damage_taken_sum'] += float(self.damage_taken[lane, player_idx])
      bucket['final_percent_sum'] += float(self.prev_percent[lane, player_idx])
      bucket['stocks_remaining_sum'] += float(fox_stocks if player_idx == 0 else falco_stocks)

    self.frames[lane] = 0
    self.prev_percent[lane, :] = 0.0
    self.damage_taken[lane, :] = 0.0
    self.last_frame_id[lane] = -124

  def summary(self):
    by_character = {}
    for character, bucket in self.by_character.items():
      games = max(1, bucket['games'])
      resolved = max(1, bucket['wins'] + bucket['losses'])
      by_character[character] = {
          **bucket,
          'winrate_resolved': bucket['wins'] / resolved,
          'avg_damage_taken': bucket['damage_taken_sum'] / games,
          'avg_final_percent': bucket['final_percent_sum'] / games,
          'avg_stocks_remaining': bucket['stocks_remaining_sum'] / games,
      }

    by_stage = {}
    for stage, bucket in self.by_stage.items():
      games = bucket['games']
      by_stage[stage] = {
          **bucket,
          'avg_length_frames': bucket['length_sum'] / max(1, games),
      }

    return {
        'total_games': self.total_games,
        'stockout_games': self.stockout_games,
        'timeout_games': self.timeout_games,
        'draw_games': self.draw_games,
        'nan_state_count': self.nan_state_count,
        'stuck_frame_count': self.stuck_frame_count,
        'by_character': by_character,
        'by_stage': by_stage,
    }


def _empty_bucket():
  return {
      'games': 0,
      'wins': 0,
      'losses': 0,
      'timeouts': 0,
      'draws': 0,
      'damage_taken_sum': 0.0,
      'final_percent_sum': 0.0,
      'stocks_remaining_sum': 0.0,
  }


def _cycle_stages(batch_size: int) -> np.ndarray:
  stages = sim_env.supported_stages()
  return np.asarray([stages[i % len(stages)] for i in range(batch_size)], dtype=object)


class _JaxDelayedAgent:

  def __init__(self, agent, policy, delay: int, batch_size: int):
    self._agent = agent
    self._policy = policy
    self.embed_controller = policy.controller_head.controller_embedding
    self._output_queue = deque()
    dummy = eval_lib.dummy_sample_outputs(self.embed_controller, [batch_size])
    for _ in range(delay):
      self._output_queue.append(dummy)
    self._controller_queue = deque()
    for _ in range(delay):
      self._controller_queue.append(dummy.controller_state)

  def step(self, game, needs_reset):
    self._output_queue.append(self._agent.step(game, needs_reset))
    return self._output_queue.popleft()

  def step_controller_state(self, game, needs_reset):
    self._controller_queue.append(
        self._agent.step_controller_state(game, needs_reset))
    return self._controller_queue.popleft()


def _build_agent(args, state: dict, names):
  name = None if names is None else [
      names[i % len(names)] for i in range(args.batch_size * 2)
  ]
  if args.platform == 'tf':
    return eval_lib.build_delayed_agent(
        state=state,
        batch_size=args.batch_size * 2,
        console_delay=0,
        name=name,
        sample_temperature=args.sample_temperature,
        compile=not args.no_compile,
        batch_steps=args.batch_steps,
        assume_game_is_from_state=args.preembed_state,
        run_on_cpu=args.run_on_cpu,
    )

  if args.run_on_cpu:
    raise ValueError('--run-on-cpu is only supported for --platform=tf')
  if args.preembed_state:
    raise ValueError('--preembed-state is only supported for --platform=tf')
  if args.batch_steps:
    raise ValueError('--batch-steps is not wired for --platform=jax yet')

  from slippi_ai.jax import agents as jax_agents
  from slippi_ai.jax import tf_checkpoint

  policy = tf_checkpoint.load_policy_from_tf_state(state)
  if name is None:
    name_code = 0
  elif isinstance(name, str):
    name_code = eval_lib.get_name_code(state, name)
  else:
    name_code = [eval_lib.get_name_code(state, n) for n in name]
  agent = jax_agents.BasicAgent(
      policy=policy,
      batch_size=args.batch_size * 2,
      name_code=name_code,
      sample_kwargs=dict(temperature=args.sample_temperature),
      compile=not args.no_compile,
      pack_args=True,
  )
  return _JaxDelayedAgent(
      agent=agent,
      policy=policy,
      delay=policy.delay,
      batch_size=args.batch_size * 2,
  )


def _pack_games(gamestates, batch_size: int):
  del batch_size
  return utils.map_nt(
      lambda a, b: np.concatenate([a, b], axis=0),
      gamestates[1],
      gamestates[2],
  )


def _pack_reset(needs_reset: np.ndarray) -> np.ndarray:
  return np.concatenate([needs_reset, needs_reset], axis=0)


def _split_controllers(decoded: sim_env.Controller, batch_size: int):
  return {
      1: utils.map_single_structure(lambda x: x[:batch_size], decoded),
      2: utils.map_single_structure(lambda x: x[batch_size:], decoded),
  }


def _count_invalid_actions(controllers) -> int:
  count = 0
  for controller in controllers.values():
    arrays = [
        controller.main_stick.x,
        controller.main_stick.y,
        controller.c_stick.x,
        controller.c_stick.y,
        controller.shoulder,
    ]
    for arr in arrays:
      count += int(np.logical_not(np.isfinite(arr)).sum())
      count += int((arr < 0.0).sum() + (arr > 1.0).sum())
  return count


def _count_invalid_encoded_actions(
    controller: sim_env.Controller,
    *,
    axis_spacing: int,
    shoulder_spacing: int,
) -> int:
  count = 0
  for arr, limit in (
      (controller.main_stick.x, axis_spacing),
      (controller.main_stick.y, axis_spacing),
      (controller.c_stick.x, axis_spacing),
      (controller.c_stick.y, axis_spacing),
      (controller.shoulder, shoulder_spacing),
  ):
    values = np.asarray(arr)
    count += int((values < 0).sum() + (values > limit).sum())
  return count


def _default_controller_spacing(state: dict) -> tuple[int, int]:
  config = state['config']['embed']['controller']
  if config.get('type', 'default') != 'default':
    raise ValueError('--fast-path currently supports the default controller embedding only')
  default = config.get('default', config)
  return int(default['axis_spacing']), int(default['shoulder_spacing'])


def _should_continue(args, completed_games: int, total_steps: int) -> bool:
  if args.fixed_steps > 0:
    return total_steps < args.warmup_steps + args.fixed_steps
  return completed_games < args.completed_games


if __name__ == '__main__':
  main()
