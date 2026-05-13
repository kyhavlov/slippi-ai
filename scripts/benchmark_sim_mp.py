import argparse
import dataclasses
import json
import multiprocessing as mp
import time
import traceback
from collections import defaultdict
from multiprocessing import shared_memory
from pathlib import Path

import melee
import numpy as np

from slippi_ai import dolphin
from slippi_ai import eval_lib
from slippi_ai import sim_env
from slippi_ai.types import Buttons, Controller, Stick


@dataclasses.dataclass(frozen=True)
class SharedArraySpec:
  name: str
  shape: tuple[int, ...]
  dtype: str


class SharedArrayOwner:

  def __init__(self):
    self.specs: list[SharedArraySpec] = []
    self._blocks: list[shared_memory.SharedMemory] = []

  def array(self, shape: tuple[int, ...], dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    block = shared_memory.SharedMemory(create=True, size=size)
    array = np.ndarray(shape, dtype=dtype, buffer=block.buf)
    array.fill(0)
    self.specs.append(SharedArraySpec(block.name, tuple(shape), dtype.str))
    self._blocks.append(block)
    return array

  def close(self):
    for block in self._blocks:
      block.close()

  def unlink(self):
    for block in self._blocks:
      try:
        block.unlink()
      except FileNotFoundError:
        pass


class SharedArrayAttacher:

  def __init__(self, specs: list[SharedArraySpec]):
    self._specs = specs
    self._index = 0
    self._blocks: list[shared_memory.SharedMemory] = []

  def array(self, shape: tuple[int, ...], dtype) -> np.ndarray:
    if self._index >= len(self._specs):
      raise IndexError('shared array spec exhausted')
    spec = self._specs[self._index]
    self._index += 1
    dtype = np.dtype(dtype)
    if tuple(shape) != spec.shape or dtype.str != spec.dtype:
      raise ValueError(
          f'shared array mismatch: expected {spec.shape}/{spec.dtype}, '
          f'got {tuple(shape)}/{dtype.str}')
    block = shared_memory.SharedMemory(name=spec.name)
    self._blocks.append(block)
    return np.ndarray(spec.shape, dtype=dtype, buffer=block.buf)

  def close(self):
    for block in self._blocks:
      block.close()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model-path', default='models/rl_doubles_v27_11000.pkl')
  parser.add_argument('--workers', type=int, default=2)
  parser.add_argument('--batch-size', type=int, default=1024,
                      help='Env batch per worker.')
  parser.add_argument('--fixed-steps', type=int, default=0)
  parser.add_argument('--completed-games', type=int, default=250)
  parser.add_argument('--warmup-steps', type=int, default=100)
  parser.add_argument('--length', type=int, default=256)
  parser.add_argument('--max-game-frames', type=int, default=28800)
  parser.add_argument('--sample-temperature', type=float, default=1.0)
  parser.add_argument('--print-every', type=int, default=0)
  parser.add_argument('--barrier-timeout', type=float, default=60.0)
  args = parser.parse_args()

  model_path = Path(args.model_path)
  if not model_path.exists():
    raise FileNotFoundError(model_path)
  if args.workers <= 0 or args.batch_size <= 0:
    raise ValueError('--workers and --batch-size must be positive')
  if args.fixed_steps <= 0 and args.completed_games <= 0:
    raise ValueError('set --fixed-steps or --completed-games')

  total_batch = args.workers * args.batch_size
  total_packed = total_batch * 2
  ctx = mp.get_context('spawn')

  obs_owner = SharedArrayOwner()
  packed = sim_env.make_packed_game_builder(total_batch, array_factory=obs_owner.array)
  action_owner = SharedArrayOwner()
  action = _shared_encoded_controller(total_packed, action_owner.array)
  state = eval_lib.load_state(path=str(model_path))
  spacing = _default_controller_spacing(state)

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
              args.fixed_steps,
              args.warmup_steps,
              obs_owner.specs,
              action_owner.specs,
              spacing,
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

    names = eval_lib.get_name_from_rl_state(state)
    agent = _build_agent(
        state=state,
        names=names,
        batch_size=total_packed,
        sample_temperature=args.sample_temperature,
    )
    _barrier_wait(obs_barrier, args.barrier_timeout, 'initial observations')
    timings = defaultdict(float)
    counters = defaultdict(int)
    measured_steps = 0
    completed_games = 0
    measured_start = None
    total_start = time.perf_counter()
    step = 0

    while True:
      measuring = step >= args.warmup_steps
      if measuring and measured_start is None:
        measured_start = time.perf_counter()

      policy_start = time.perf_counter()
      controller_state = agent.step_controller_state(packed.game, packed.needs_reset)
      policy_done = time.perf_counter()
      invalid = _copy_controller(action, controller_state, spacing)
      action_done = time.perf_counter()
      _barrier_wait(action_barrier, args.barrier_timeout, 'action release')
      released_actions = time.perf_counter()
      _barrier_wait(obs_barrier, args.barrier_timeout, 'observation wait')
      obs_done = time.perf_counter()
      done, stockout, timeout, max_frame = _sum_step_counters(
          step_counters, args.workers)

      if measuring:
        timings['policy_submit_s'] += policy_done - policy_start
        timings['action_copy_s'] += action_done - policy_done
        timings['action_release_s'] += released_actions - action_done
        timings['obs_wait_s'] += obs_done - released_actions
        counters['invalid_actions'] += invalid
        counters['done'] += done
        counters['stockout'] += stockout
        counters['timeout'] += timeout
        counters['max_frame_reached'] += max_frame
        completed_games += done
        measured_steps += 1

      if args.print_every and (step + 1) % args.print_every == 0:
        elapsed = time.perf_counter() - total_start
        print(
            f'steps={step + 1} games={completed_games} '
            f'env_steps_per_sec={total_batch * (step + 1) / elapsed:.1f}',
            flush=True,
        )

      step += 1
      if _should_stop(args, measured_steps, completed_games):
        stop_event.set()
        action_barrier.abort()
        break

    total_elapsed = time.perf_counter() - total_start
    measured_elapsed = 0.0 if measured_start is None else time.perf_counter() - measured_start
    worker_results = [result_queue.get(timeout=30.0) for _ in processes]
    for process in processes:
      process.join(timeout=10.0)
      if process.exitcode != 0:
        raise RuntimeError(f'worker {process.pid} exited with {process.exitcode}')

    measured_env_steps = measured_steps * total_batch
    summary = {
        'benchmark': {
            'workers': args.workers,
            'batch_size_per_worker': args.batch_size,
            'total_batch_size': total_batch,
            'fixed_steps': args.fixed_steps,
            'target_completed_games': args.completed_games,
            'warmup_steps': args.warmup_steps,
            'length': args.length,
            'completed_games': completed_games,
            'total_elapsed_sec': total_elapsed,
            'measured_elapsed_sec': measured_elapsed,
            'measured_steps': measured_steps,
            'measured_env_steps': measured_env_steps,
            'measured_env_steps_per_sec': measured_env_steps / max(measured_elapsed, 1e-9),
            'measured_ns_per_env_step': measured_elapsed * 1e9 / max(1, measured_env_steps),
        },
        'main_timings_sec': dict(timings),
        'main_counters': dict(counters),
        'workers': worker_results,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
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
    fixed_steps: int,
    warmup_steps: int,
    obs_specs: list[SharedArraySpec],
    action_specs: list[SharedArraySpec],
    controller_spacing: tuple[int, int],
    obs_barrier,
    action_barrier,
    stop_event,
    step_counters,
    barrier_timeout: float,
    result_queue,
):
  obs_attacher = SharedArrayAttacher(obs_specs)
  action_attacher = SharedArrayAttacher(action_specs)
  env = None
  try:
    packed = sim_env.make_packed_game_builder(
        total_batch,
        array_factory=obs_attacher.array,
    )
    action = _shared_encoded_controller(total_batch * 2, action_attacher.array)
    env = sim_env.SimBatchedEnvironment(
        num_envs=batch_size,
        players={
            1: dolphin.AI(melee.Character.FOX),
            2: dolphin.AI(melee.Character.FALCO),
        },
        length=length,
        stage=_cycle_stages(batch_size, offset),
        character_pairs=sim_env.balanced_fox_falco_pairs(batch_size, offset),
        max_frame_id=max_game_frames - 123,
    )
    env_slice = slice(offset, offset + batch_size)
    initial_reset = np.ones(batch_size, dtype=np.bool_)
    packed.fill_slice(
        env.buffers.gamestate_view[env.cursor],
        initial_reset,
        env_slice,
    )
    _barrier_wait(obs_barrier, barrier_timeout, f'worker {worker_id} initial observations')

    timings = defaultdict(float)
    counters = defaultdict(int)
    measured_steps = 0
    last_frame_id = np.full(batch_size, -124, dtype=np.int32)
    step = 0
    while not stop_event.is_set():
      try:
        _barrier_wait(action_barrier, barrier_timeout, f'worker {worker_id} action wait')
      except RuntimeError:
        if stop_event.is_set():
          break
        raise
      if stop_event.is_set():
        break
      measuring = step >= warmup_steps

      step_start = time.perf_counter()
      needs_reset, terminal = _step_with_global_actions(
          env,
          action,
          total_batch=total_batch,
          offset=offset,
          batch_size=batch_size,
          max_frame_id=max_game_frames - 123,
          controller_spacing=controller_spacing,
      )
      step_done = time.perf_counter()
      packed.fill_slice(
          env.buffers.gamestate_view[env.cursor],
          needs_reset,
          env_slice,
      )
      fill_done = time.perf_counter()
      done_count = int(needs_reset.sum())
      stockout_count = int(terminal['stockout'].sum())
      timeout_count = int(terminal['max_frame_reached'].sum())
      _write_step_counters(
          step_counters,
          worker_id,
          done_count,
          stockout_count,
          timeout_count,
      )
      _barrier_wait(obs_barrier, barrier_timeout, f'worker {worker_id} observation release')
      obs_done = time.perf_counter()

      if measuring:
        frame_id = terminal['frame_id']
        advanced = np.logical_or(frame_id > last_frame_id, needs_reset)
        timings['step_s'] += step_done - step_start
        timings['fill_s'] += fill_done - step_done
        timings['obs_release_s'] += obs_done - fill_done
        counters['done'] += done_count
        counters['stockout'] += stockout_count
        counters['timeout'] += timeout_count
        counters['stuck_frame'] += int(np.logical_not(advanced).sum())
        counters['nan_state'] += _count_nan_state(env.buffers.gamestate_view[env.cursor])
        measured_steps += 1
        last_frame_id[:] = frame_id
        last_frame_id[needs_reset] = -124
      step += 1

    result_queue.put({
        'worker_id': worker_id,
        'batch_size': batch_size,
        'measured_steps': measured_steps,
        'timings_sec': dict(timings),
        'counters': dict(counters),
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


def _step_with_global_actions(
    env: sim_env.SimBatchedEnvironment,
    action_controller: Controller,
    *,
    total_batch: int,
    offset: int,
    batch_size: int,
    max_frame_id: int,
    controller_spacing: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
  env._ensure_cursor_room()
  if np.any(env._pending_reset):
    env.reset(np.flatnonzero(env._pending_reset))
    env._pending_reset[:] = False

  action = env.buffers.controller_action_view[env.cursor]
  first = slice(offset, offset + batch_size)
  second = slice(total_batch + offset, total_batch + offset + batch_size)
  axis_spacing, shoulder_spacing = controller_spacing
  sim_env._write_encoded_controller_action(
      action,
      action_controller,
      player_index=0,
      source_slice=first,
      axis_spacing=axis_spacing,
      shoulder_spacing=shoulder_spacing,
  )
  sim_env._write_encoded_controller_action(
      action,
      action_controller,
      player_index=1,
      source_slice=second,
      axis_spacing=axis_spacing,
      shoulder_spacing=shoulder_spacing,
  )

  step_t = env.cursor
  env._env.step(max_frame_id=max_frame_id)
  needs_reset = env.buffers.done[step_t].astype(np.bool_, copy=True)
  env._pending_reset[:] = needs_reset
  terminal = sim_env.terminal_view(env.buffers)[step_t].copy()
  env._last_step_info = sim_env.SimStepInfo(terminal=terminal, step_t=step_t)
  return needs_reset, terminal


def _count_nan_state(frame: np.ndarray) -> int:
  slots = frame['slots']
  total = 0
  for i in range(slots.shape[1]):
    slot = slots[:, i]
    if not np.any(slot['present']):
      continue
    total += int(np.isnan(slot['pos_x']).sum())
    total += int(np.isnan(slot['pos_y']).sum())
  return total


def _write_step_counters(
    counters,
    worker_id: int,
    done: int,
    stockout: int,
    timeout: int,
):
  base = int(worker_id) * 4
  counters[base] = int(done)
  counters[base + 1] = int(stockout)
  counters[base + 2] = int(timeout)
  counters[base + 3] = int(timeout)


def _sum_step_counters(counters, workers: int) -> tuple[int, int, int, int]:
  done = 0
  stockout = 0
  timeout = 0
  max_frame = 0
  for worker_id in range(workers):
    base = worker_id * 4
    done += int(counters[base])
    stockout += int(counters[base + 1])
    timeout += int(counters[base + 2])
    max_frame += int(counters[base + 3])
  return done, stockout, timeout, max_frame


def _should_stop(args, measured_steps: int, completed_games: int) -> bool:
  if measured_steps <= 0:
    return False
  if args.fixed_steps > 0:
    return measured_steps >= args.fixed_steps
  return completed_games >= args.completed_games


def _barrier_wait(barrier, timeout: float, label: str):
  try:
    return barrier.wait(timeout=timeout)
  except Exception as exc:
    raise RuntimeError(f'timed out or broke barrier during {label}') from exc


def _shared_encoded_controller(total_packed: int, array_factory) -> Controller:
  shape = (int(total_packed),)
  return Controller(
      main_stick=Stick(
          x=array_factory(shape, np.uint8),
          y=array_factory(shape, np.uint8),
      ),
      c_stick=Stick(
          x=array_factory(shape, np.uint8),
          y=array_factory(shape, np.uint8),
      ),
      shoulder=array_factory(shape, np.uint8),
      buttons=Buttons(**{
          name: array_factory(shape, np.bool_)
          for name in Buttons._fields
      }),
  )


def _copy_controller(dst: Controller, src: Controller, spacing: tuple[int, int]) -> int:
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
    np.copyto(dst_arr, values, casting='unsafe')
  for name in Buttons._fields:
    np.copyto(getattr(dst.buttons, name), np.asarray(getattr(src.buttons, name)))
  return invalid


def _build_agent(
    *,
    state: dict,
    names,
    batch_size: int,
    sample_temperature: float,
):
  from scripts import benchmark_sim_env

  args = argparse.Namespace(
      batch_size=batch_size // 2,
      sample_temperature=sample_temperature,
      platform='jax',
      run_on_cpu=False,
      preembed_state=False,
      batch_steps=0,
      no_compile=False,
  )
  return benchmark_sim_env._build_agent(args, state, names)


def _default_controller_spacing(state: dict) -> tuple[int, int]:
  config = state['config']['embed']['controller']
  if config.get('type', 'default') != 'default':
    raise ValueError('benchmark only supports the default controller embedding')
  default = config.get('default', config)
  return int(default['axis_spacing']), int(default['shoulder_spacing'])


def _cycle_stages(batch_size: int, offset: int) -> np.ndarray:
  stages = sim_env.supported_stages()
  return np.asarray(
      [stages[(offset + i) % len(stages)] for i in range(batch_size)],
      dtype=object,
  )


if __name__ == '__main__':
  main()
