import argparse
import json
import multiprocessing as mp
import time
import traceback
from collections import Counter, defaultdict, deque
from pathlib import Path

import jax
import melee
import numpy as np
from flax import nnx

from scripts import benchmark_jax_sim_rl
from scripts import benchmark_sim_mp
from slippi_ai import dolphin
from slippi_ai import eval_lib
from slippi_ai import sim_env
from slippi_ai import utils
from slippi_ai.jax import agents as jax_agents
from slippi_ai.jax import tf_checkpoint
from slippi_ai.types import Buttons, Controller


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model-a', required=True)
  parser.add_argument('--model-b', required=True)
  parser.add_argument('--workers', type=int, default=8)
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument('--games', type=int, default=250)
  parser.add_argument('--length', type=int, default=128)
  parser.add_argument('--max-game-frames', type=int, default=28800)
  parser.add_argument('--matchup', choices=sim_env.SUPPORTED_MATCHUPS,
                      default='fox-falco')
  parser.add_argument('--sample-temperature', type=float, default=1.0)
  parser.add_argument('--learner-param-dtype', choices=('float32', 'bfloat16'),
                      default='float32')
  parser.add_argument('--barrier-timeout', type=float, default=900.0)
  parser.add_argument('--print-every', type=int, default=100)
  parser.add_argument('--output-json', default='')
  args = parser.parse_args()

  model_a = eval_lib.load_state(path=args.model_a)
  model_b = eval_lib.load_state(path=args.model_b)
  spacing = benchmark_sim_mp._default_controller_spacing(model_a)
  if spacing != benchmark_sim_mp._default_controller_spacing(model_b):
    raise ValueError('model controller embeddings use different spacing')

  summaries = []
  for a_port in (1, 2):
    summaries.append(_run_round(
        args=args,
        model_a=model_a,
        model_b=model_b,
        controller_spacing=spacing,
        a_port=a_port,
    ))

  result = {
      'model_a': args.model_a,
      'model_b': args.model_b,
      'matchup': args.matchup,
      'rounds': summaries,
      'aggregate': _aggregate_rounds(summaries),
  }
  if args.output_json:
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
  print(json.dumps(result, indent=2, sort_keys=True))


def _run_round(
    *,
    args,
    model_a: dict,
    model_b: dict,
    controller_spacing: tuple[int, int],
    a_port: int,
) -> dict:
  total_batch = args.workers * args.batch_size
  total_packed = total_batch * 2
  ctx = mp.get_context('spawn')

  obs_owner = benchmark_sim_mp.SharedArrayOwner()
  packed = sim_env.make_packed_game_builder(total_batch, array_factory=obs_owner.array)
  action_owner = benchmark_sim_mp.SharedArrayOwner()
  action = benchmark_sim_mp._shared_encoded_controller(total_packed, action_owner.array)

  obs_barrier = ctx.Barrier(args.workers + 1)
  action_barrier = ctx.Barrier(args.workers + 1)
  stop_event = ctx.Event()
  step_counters = ctx.Array('i', args.workers * 4, lock=False)
  result_queue = ctx.Queue()
  processes = []

  try:
    for worker_id in range(args.workers):
      offset = worker_id * args.batch_size
      process = ctx.Process(
          target=_worker_main,
          args=(
              worker_id,
              args.batch_size,
              total_batch,
              offset,
              args.length,
              args.max_game_frames,
              args.matchup,
              obs_owner.specs,
              action_owner.specs,
              controller_spacing,
              obs_barrier,
              action_barrier,
              stop_event,
              step_counters,
              args.barrier_timeout,
              result_queue,
          ),
      )
      process.start()
      processes.append(process)

    agent_a = _build_agent(
        model_a, total_batch, args.sample_temperature,
        args.learner_param_dtype, seed=0)
    agent_b = _build_agent(
        model_b, total_batch, args.sample_temperature,
        args.learner_param_dtype, seed=1)
    action_queue_a = _action_delay_queue(agent_a)
    action_queue_b = _action_delay_queue(agent_b)
    benchmark_sim_mp._barrier_wait(
        obs_barrier, args.barrier_timeout, 'initial observations')

    a_slice, b_slice = _model_slices(a_port, total_batch)
    completed_games = 0
    step = 0
    start = time.perf_counter()
    timings = defaultdict(float)
    counters = defaultdict(int)

    while completed_games < args.games:
      policy_start = time.perf_counter()
      reset_a = np.asarray(packed.needs_reset[a_slice], dtype=np.bool_)
      reset_b = np.asarray(packed.needs_reset[b_slice], dtype=np.bool_)
      _reset_action_delay_queue(action_queue_a, agent_a, reset_a)
      _reset_action_delay_queue(action_queue_b, agent_b, reset_b)
      ctrl_a = agent_a.step_controller_state(
          _slice_tree(packed.game, a_slice),
          reset_a)
      ctrl_b = agent_b.step_controller_state(
          _slice_tree(packed.game, b_slice),
          reset_b)
      ctrl_a = _delay_action(action_queue_a, ctrl_a)
      ctrl_b = _delay_action(action_queue_b, ctrl_b)
      policy_done = time.perf_counter()

      invalid = 0
      invalid += _copy_controller_slice(action, ctrl_a, a_slice, controller_spacing)
      invalid += _copy_controller_slice(action, ctrl_b, b_slice, controller_spacing)
      action_done = time.perf_counter()
      benchmark_sim_mp._barrier_wait(
          action_barrier, args.barrier_timeout, 'action release')
      release_done = time.perf_counter()
      benchmark_sim_mp._barrier_wait(
          obs_barrier, args.barrier_timeout, 'observation wait')
      obs_done = time.perf_counter()

      done, stockout, timeout, max_frame = benchmark_sim_mp._sum_step_counters(
          step_counters, args.workers)
      completed_games += done
      counters['done'] += done
      counters['stockout'] += stockout
      counters['timeout'] += timeout
      counters['max_frame_reached'] += max_frame
      counters['invalid_actions'] += invalid
      timings['policy_s'] += policy_done - policy_start
      timings['action_copy_s'] += action_done - policy_done
      timings['action_release_s'] += release_done - action_done
      timings['obs_wait_s'] += obs_done - release_done
      step += 1

      if args.print_every and step % args.print_every == 0:
        elapsed = time.perf_counter() - start
        print(
            f'round_a_port={a_port} steps={step} games={completed_games} '
            f'env_steps_per_sec={total_batch * step / max(elapsed, 1e-9):.1f}',
            flush=True,
        )

    stop_event.set()
    action_barrier.abort()
    worker_results = [result_queue.get(timeout=30.0) for _ in processes]
    for process in processes:
      process.join(timeout=10.0)
      if process.exitcode != 0:
        raise RuntimeError(f'worker {process.pid} exited with {process.exitcode}')

    elapsed = time.perf_counter() - start
    port_stats = _combine_worker_stats(worker_results)
    model_stats = _map_port_stats_to_models(port_stats, a_port)
    return {
        'a_port': a_port,
        'elapsed_sec': elapsed,
        'steps': step,
        'env_steps_per_sec': total_batch * step / max(elapsed, 1e-9),
        'main_timings_sec': dict(timings),
        'main_counters': dict(counters),
        'port_stats': port_stats,
        'model_stats': model_stats,
        'worker_results': worker_results,
    }
  finally:
    for process in processes:
      if process.is_alive():
        process.terminate()
      process.join(timeout=1.0)
    obs_owner.close()
    action_owner.close()
    obs_owner.unlink()
    action_owner.unlink()


def _worker_main(
    worker_id: int,
    batch_size: int,
    total_batch: int,
    offset: int,
    length: int,
    max_game_frames: int,
    matchup: str,
    obs_specs: list[benchmark_sim_mp.SharedArraySpec],
    action_specs: list[benchmark_sim_mp.SharedArraySpec],
    controller_spacing: tuple[int, int],
    obs_barrier,
    action_barrier,
    stop_event,
    step_counters,
    barrier_timeout: float,
    result_queue,
):
  obs_attacher = benchmark_sim_mp.SharedArrayAttacher(obs_specs)
  action_attacher = benchmark_sim_mp.SharedArrayAttacher(action_specs)
  env = None
  try:
    packed = sim_env.make_packed_game_builder(
        total_batch,
        array_factory=obs_attacher.array,
    )
    action = benchmark_sim_mp._shared_encoded_controller(
        total_batch * 2, action_attacher.array)
    p1_character, p2_character = sim_env.player_pair_for_matchup(matchup)
    env = sim_env.SimBatchedEnvironment(
        num_envs=batch_size,
        players={
            1: dolphin.AI(p1_character),
            2: dolphin.AI(p2_character),
        },
        length=length,
        stage=benchmark_sim_mp._cycle_stages(batch_size, offset),
        character_pairs=sim_env.character_pairs_for_matchup(
            matchup, batch_size, offset),
        max_frame_id=max_game_frames - 123,
    )
    env_slice = slice(offset, offset + batch_size)
    packed.fill_slice(
        env.buffers.gamestate_view[env.cursor],
        np.ones(batch_size, dtype=np.bool_),
        env_slice,
        env._last_controllers,
        controller_slice=slice(None),
    )
    benchmark_sim_mp._barrier_wait(
        obs_barrier, barrier_timeout, f'worker {worker_id} initial observations')

    stats = _empty_port_stats()
    timings = defaultdict(float)
    counters = defaultdict(int)
    while not stop_event.is_set():
      try:
        benchmark_sim_mp._barrier_wait(
            action_barrier, barrier_timeout, f'worker {worker_id} action wait')
      except RuntimeError:
        if stop_event.is_set():
          break
        raise
      if stop_event.is_set():
        break

      step_start = time.perf_counter()
      needs_reset, terminal = benchmark_sim_mp._step_with_global_actions(
          env,
          action,
          total_batch=total_batch,
          offset=offset,
          batch_size=batch_size,
          max_frame_id=max_game_frames - 123,
          controller_spacing=controller_spacing,
      )
      step_done = time.perf_counter()
      _record_done_stats(env, needs_reset, terminal, stats)
      packed.fill_slice(
          env.buffers.gamestate_view[env.cursor],
          needs_reset,
          env_slice,
          env._last_controllers,
          controller_slice=slice(None),
      )
      fill_done = time.perf_counter()
      done_count = int(needs_reset.sum())
      stockout_count = int(terminal['stockout'].sum())
      timeout_count = int(terminal['max_frame_reached'].sum())
      benchmark_sim_mp._write_step_counters(
          step_counters,
          worker_id,
          done_count,
          stockout_count,
          timeout_count,
      )
      benchmark_sim_mp._barrier_wait(
          obs_barrier, barrier_timeout, f'worker {worker_id} observation release')
      obs_done = time.perf_counter()
      timings['step_s'] += step_done - step_start
      timings['fill_s'] += fill_done - step_done
      timings['obs_release_s'] += obs_done - fill_done
      counters['done'] += done_count
      counters['stockout'] += stockout_count
      counters['timeout'] += timeout_count

    result_queue.put({
        'worker_id': worker_id,
        'batch_size': batch_size,
        'timings_sec': dict(timings),
        'counters': dict(counters),
        'stats': _jsonable_port_stats(stats),
    })
  except BaseException:
    result_queue.put({
        'worker_id': worker_id,
        'error': traceback.format_exc(),
    })
    raise
  finally:
    if env is not None:
      env.stop()
    obs_attacher.close()
    action_attacher.close()


def _build_agent(
    state: dict,
    batch_size: int,
    sample_temperature: float,
    param_dtype: str,
    seed: int,
):
  policy = tf_checkpoint.load_policy_from_tf_state(state, param_dtype=param_dtype)
  name_code = benchmark_jax_sim_rl._name_code(state, batch_size)
  return jax_agents.BasicAgent(
      policy,
      batch_size=batch_size,
      name_code=name_code,
      rngs=nnx.Rngs(seed),
      sample_kwargs=dict(temperature=sample_temperature),
      compile=True,
      pack_args=True,
  )


def _action_delay_queue(agent: jax_agents.BasicAgent) -> deque:
  return deque([
      agent._policy.controller_head.dummy_controller([agent._batch_size])
      for _ in range(agent._policy.delay)
  ])


def _delay_action(action_queue: deque, controller_state: Controller) -> Controller:
  if not action_queue:
    return controller_state
  action_queue.append(controller_state)
  return action_queue.popleft()


def _reset_action_delay_queue(
    action_queue: deque,
    agent: jax_agents.BasicAgent,
    reset_mask: np.ndarray,
):
  if not np.any(reset_mask):
    return
  benchmark_jax_sim_rl._reset_delay_queue_lanes(
      action_queue,
      agent._policy.controller_head.dummy_controller([agent._batch_size]),
      reset_mask,
  )


def _model_slices(a_port: int, total_batch: int) -> tuple[slice, slice]:
  port1 = slice(0, total_batch)
  port2 = slice(total_batch, 2 * total_batch)
  return (port1, port2) if a_port == 1 else (port2, port1)


def _slice_tree(tree, item: slice):
  return utils.map_single_structure(lambda x: np.asarray(x)[item], tree)


def _copy_controller_slice(
    dst: Controller,
    src: Controller,
    dst_slice: slice,
    spacing: tuple[int, int],
) -> int:
  invalid = 0
  axis_spacing, shoulder_spacing = spacing
  for dst_arr, src_arr, limit in (
      (dst.main_stick.x, src.main_stick.x, axis_spacing),
      (dst.main_stick.y, src.main_stick.y, axis_spacing),
      (dst.c_stick.x, src.c_stick.x, axis_spacing),
      (dst.c_stick.y, src.c_stick.y, axis_spacing),
      (dst.shoulder, src.shoulder, shoulder_spacing),
  ):
    values = np.asarray(src_arr)
    invalid += int((values < 0).sum() + (values > limit).sum())
    np.copyto(dst_arr[dst_slice], values, casting='unsafe')
  for name in Buttons._fields:
    np.copyto(
        getattr(dst.buttons, name)[dst_slice],
        np.asarray(getattr(src.buttons, name)))
  return invalid


def _empty_port_stats() -> dict:
  return {
      'games': 0,
      'port1_wins': 0,
      'port2_wins': 0,
      'ties': 0,
      'port1_percent_sum': 0.0,
      'port2_percent_sum': 0.0,
      'frame_sum': 0.0,
      'stage_counts': Counter(),
      'character_pair_counts': Counter(),
  }


def _record_done_stats(
    env: sim_env.SimBatchedEnvironment,
    needs_reset: np.ndarray,
    terminal: np.ndarray,
    stats: dict,
):
  done_ids = np.flatnonzero(needs_reset)
  if done_ids.size == 0:
    return
  frame = env.buffers.gamestate_view[env._last_step_info.step_t]
  p1 = _source_slot(frame, 0)
  p2 = _source_slot(frame, 1)
  p1_stocks = p1['stocks'][done_ids]
  p2_stocks = p2['stocks'][done_ids]
  p1_percent = p1['percent'][done_ids]
  p2_percent = p2['percent'][done_ids]
  stats['games'] += int(done_ids.size)
  stats['port1_wins'] += int((p1_stocks > p2_stocks).sum())
  stats['port2_wins'] += int((p2_stocks > p1_stocks).sum())
  stats['ties'] += int((p1_stocks == p2_stocks).sum())
  stats['port1_percent_sum'] += float(np.sum(p1_percent))
  stats['port2_percent_sum'] += float(np.sum(p2_percent))
  stats['frame_sum'] += float(np.sum(terminal['frame_id'][done_ids]))
  stats['stage_counts'].update(map(int, frame['stage_id'][done_ids]))
  stats['character_pair_counts'].update(
      (int(a), int(b))
      for a, b in zip(p1['char_id'][done_ids], p2['char_id'][done_ids]))


def _source_slot(frame: np.ndarray, source_player: int) -> np.ndarray:
  slots = frame['slots']
  for slot_index in range(slots.shape[1]):
    slot = slots[:, slot_index]
    if np.any(slot['present']) and np.all(slot['source_player'] == source_player):
      return slot
  raise RuntimeError(f'missing source player {source_player}')


def _jsonable_port_stats(stats: dict) -> dict:
  games = max(1, int(stats['games']))
  return {
      'games': int(stats['games']),
      'port1_wins': int(stats['port1_wins']),
      'port2_wins': int(stats['port2_wins']),
      'ties': int(stats['ties']),
      'port1_avg_end_percent': stats['port1_percent_sum'] / games,
      'port2_avg_end_percent': stats['port2_percent_sum'] / games,
      'avg_game_length_frames': stats['frame_sum'] / games,
      'stage_counts': {str(k): int(v) for k, v in stats['stage_counts'].items()},
      'character_pair_counts': {
          f'{k[0]}:{k[1]}': int(v)
          for k, v in stats['character_pair_counts'].items()
      },
  }


def _combine_worker_stats(worker_results: list[dict]) -> dict:
  combined = _empty_port_stats()
  for result in worker_results:
    if 'error' in result:
      raise RuntimeError(result['error'])
    stats = result['stats']
    combined['games'] += stats['games']
    combined['port1_wins'] += stats['port1_wins']
    combined['port2_wins'] += stats['port2_wins']
    combined['ties'] += stats['ties']
    combined['port1_percent_sum'] += stats['port1_avg_end_percent'] * max(1, stats['games'])
    combined['port2_percent_sum'] += stats['port2_avg_end_percent'] * max(1, stats['games'])
    combined['frame_sum'] += stats['avg_game_length_frames'] * max(1, stats['games'])
    combined['stage_counts'].update({int(k): v for k, v in stats['stage_counts'].items()})
    for key, value in stats['character_pair_counts'].items():
      a, b = key.split(':')
      combined['character_pair_counts'][(int(a), int(b))] += value
  return _jsonable_port_stats(combined)


def _map_port_stats_to_models(port_stats: dict, a_port: int) -> dict:
  if a_port == 1:
    a_wins = port_stats['port1_wins']
    b_wins = port_stats['port2_wins']
    a_percent = port_stats['port1_avg_end_percent']
    b_percent = port_stats['port2_avg_end_percent']
  else:
    a_wins = port_stats['port2_wins']
    b_wins = port_stats['port1_wins']
    a_percent = port_stats['port2_avg_end_percent']
    b_percent = port_stats['port1_avg_end_percent']
  games = max(1, port_stats['games'])
  return {
      'model_a_wins': a_wins,
      'model_b_wins': b_wins,
      'ties': port_stats['ties'],
      'model_a_winrate': a_wins / games,
      'model_b_winrate': b_wins / games,
      'model_a_avg_end_percent': a_percent,
      'model_b_avg_end_percent': b_percent,
  }


def _aggregate_rounds(rounds: list[dict]) -> dict:
  games = 0
  a_wins = 0
  b_wins = 0
  ties = 0
  timeouts = 0
  for item in rounds:
    stats = item['model_stats']
    games += item['port_stats']['games']
    a_wins += stats['model_a_wins']
    b_wins += stats['model_b_wins']
    ties += stats['ties']
    timeouts += item['main_counters']['timeout']
  denom = max(1, games)
  return {
      'games': games,
      'model_a_wins': a_wins,
      'model_b_wins': b_wins,
      'ties': ties,
      'timeouts': timeouts,
      'model_a_winrate': a_wins / denom,
      'model_b_winrate': b_wins / denom,
  }


if __name__ == '__main__':
  main()
