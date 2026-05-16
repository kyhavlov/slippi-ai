"""Multiprocessing support for CPU sim shards.

The main JAX process owns policy inference and allocates shared-memory
observation/action arrays. Each worker process attaches to those arrays, owns a
single `SimBatchedEnvironment` shard, and synchronizes with the main process
through barriers: wait for actions, step the shard, write the next observations,
then release the main process to run another policy batch.

This file intentionally stays below the RL layer. It knows how to allocate and
attach shared arrays, decode shared policy actions into worker-local env steps,
cycle supported stages across lanes, and report timing/counter data. It does not
own learner logic, reward construction, or checkpoint handling; `jax_rollout.py`
builds those pieces around these workers.
"""

import dataclasses
import time
import traceback
from collections import defaultdict
from multiprocessing import shared_memory

import melee
import numpy as np

from slippi_ai import dolphin
from slippi_ai import sim_env
from slippi_ai.types import Buttons, Controller, Stick


@dataclasses.dataclass(frozen=True)
class SharedArraySpec:
  name: str
  shape: tuple[int, ...]
  dtype: str


class SharedArrayOwner:
  """Owns shared NumPy buffers that worker processes attach to in order."""

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
  """Reconstructs the owner's shared arrays in the same allocation order."""

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


def worker_main(
    worker_id: int,
    batch_size: int,
    total_batch: int,
    offset: int,
    length: int,
    max_game_frames: int,
    warmup_steps: int,
    character_pool: str,
    obs_specs: list[SharedArraySpec],
    terminal_obs_specs: list[SharedArraySpec] | None,
    action_specs: list[SharedArraySpec],
    controller_spacing: tuple[int, int],
    obs_barrier,
    action_barrier,
    stop_event,
    step_counters,
    step_timings,
    barrier_timeout: float,
    result_queue,
    active_worker_count=None,
    measure_worker_steps=None,
):
  """Run one CPU sim shard against shared observation/action buffers."""
  obs_attacher = SharedArrayAttacher(obs_specs)
  terminal_obs_attacher = (
      SharedArrayAttacher(terminal_obs_specs)
      if terminal_obs_specs is not None else None)
  action_attacher = SharedArrayAttacher(action_specs)
  env = None
  try:
    game_batch = sim_env.make_game_batch_buffers(
        total_batch,
        array_factory=obs_attacher.array,
    )
    terminal_game_batch = (
        sim_env.make_game_batch_buffers(
            total_batch,
            array_factory=terminal_obs_attacher.array,
        )
        if terminal_obs_attacher is not None else None)
    action = shared_action_buffer(total_batch * 2, action_attacher.array)
    p1_character, p2_character = sim_env.character_assignments_for_pool(
        character_pool, 1, offset)[0]
    env = sim_env.SimBatchedEnvironment(
        num_envs=batch_size,
        players={
            1: dolphin.AI(p1_character),
            2: dolphin.AI(p2_character),
        },
        length=length,
        stage=cycle_stages(batch_size, offset),
        character_pool=character_pool,
        max_frame_id=max_game_frames - 123,
    )
    env_slice = slice(offset, offset + batch_size)
    initial_reset = np.ones(batch_size, dtype=np.bool_)
    game_batch.fill_slice(
        env.buffers.gamestate_view[env.cursor],
        initial_reset,
        env_slice,
        env._last_controllers,
        controller_slice=slice(None),
    )
    if terminal_game_batch is not None:
      # Terminal rewards need the post-step game state for the transition that
      # ended. This separate Game batch snapshots that state before any reset.
      terminal_game_batch.fill_slice(
          env.buffers.gamestate_view[env.cursor],
          initial_reset,
          env_slice,
          env._last_controllers,
          controller_slice=slice(None),
      )
    barrier_wait(obs_barrier, barrier_timeout, f'worker {worker_id} initial observations')

    timings = defaultdict(float)
    counters = defaultdict(int)
    measured_steps = 0
    last_frame_id = np.full(batch_size, -124, dtype=np.int32)
    step = 0
    while not stop_event.is_set():
      try:
        barrier_wait(action_barrier, barrier_timeout, f'worker {worker_id} action wait')
      except RuntimeError:
        if stop_event.is_set():
          break
        raise
      if stop_event.is_set():
        break
      active = worker_is_active(active_worker_count, worker_id)

      if not active:
        # Startup staggering activates workers one at a time while still
        # releasing the barriers expected by the main rollout loop.
        write_step_counters(
            step_counters,
            worker_id,
            0,
            0,
            0,
            0,
        )
        write_step_timings(step_timings, worker_id, 0.0, 0.0, 0.0)
        barrier_wait(
            obs_barrier,
            barrier_timeout,
            f'worker {worker_id} inactive observation release')
        step += 1
        continue

      measuring = (
          step >= warmup_steps
          and worker_measurement_enabled(measure_worker_steps))

      step_start = time.perf_counter()
      needs_reset, terminal = step_worker_with_shared_actions(
          env,
          action,
          total_batch=total_batch,
          offset=offset,
          batch_size=batch_size,
          max_frame_id=max_game_frames - 123,
          controller_spacing=controller_spacing,
          terminal_game_batch=terminal_game_batch,
          terminal_env_slice=env_slice,
      )
      step_done = time.perf_counter()
      game_batch.fill_slice(
          env.buffers.gamestate_view[env.cursor],
          needs_reset,
          env_slice,
          env._last_controllers,
          controller_slice=slice(None),
      )
      fill_done = time.perf_counter()
      step_s = step_done - step_start
      fill_s = fill_done - step_done
      done_count = int(needs_reset.sum())
      stockout_count = int(terminal['stockout'].sum())
      timeout_count = int(terminal['max_frame_reached'].sum())
      write_step_counters(
          step_counters,
          worker_id,
          done_count,
          stockout_count,
          timeout_count,
          timeout_count,
      )
      write_step_timings(
          step_timings,
          worker_id,
          step_s,
          fill_s,
          0.0,
      )
      barrier_wait(obs_barrier, barrier_timeout, f'worker {worker_id} observation release')
      obs_done = time.perf_counter()
      obs_release_s = obs_done - fill_done

      if measuring:
        frame_id = terminal['frame_id']
        advanced = np.logical_or(frame_id > last_frame_id, needs_reset)
        timings['step_s'] += step_s
        timings['fill_s'] += fill_s
        timings['obs_release_s'] += obs_release_s
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
    if terminal_obs_attacher is not None:
      terminal_obs_attacher.close()
    action_attacher.close()


def worker_is_active(active_worker_count, worker_id: int) -> bool:
  if active_worker_count is None:
    return True
  return int(worker_id) < int(active_worker_count.value)


def worker_measurement_enabled(measure_worker_steps) -> bool:
  if measure_worker_steps is None:
    return True
  return bool(measure_worker_steps.value)


def step_worker_with_shared_actions(
    env: sim_env.SimBatchedEnvironment,
    action_controller: Controller,
    *,
    total_batch: int,
    offset: int,
    batch_size: int,
    max_frame_id: int,
    controller_spacing: tuple[int, int],
    terminal_game_batch = None,
    terminal_env_slice: slice | None = None,
    post_step_frame_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Consume shared [all p1 actions, all p2 actions], step, and refill output."""
  env._ensure_cursor_room()

  action = env.buffers.controller_action_view[env.cursor]
  first = slice(offset, offset + batch_size)
  second = slice(total_batch + offset, total_batch + offset + batch_size)
  axis_spacing, shoulder_spacing = controller_spacing
  sim_env.write_encoded_controller_action(
      action,
      action_controller,
      player_index=0,
      source_slice=first,
      axis_spacing=axis_spacing,
      shoulder_spacing=shoulder_spacing,
  )
  sim_env.write_encoded_controller_action(
      action,
      action_controller,
      player_index=1,
      source_slice=second,
      axis_spacing=axis_spacing,
      shoulder_spacing=shoulder_spacing,
  )
  sim_env.copy_encoded_controller(
      env._last_controllers[1],
      action_controller,
      source_slice=first,
      axis_spacing=axis_spacing,
      shoulder_spacing=shoulder_spacing,
  )
  sim_env.copy_encoded_controller(
      env._last_controllers[2],
      action_controller,
      source_slice=second,
      axis_spacing=axis_spacing,
      shoulder_spacing=shoulder_spacing,
  )

  step_t = env.cursor
  env._env.step(max_frame_id=max_frame_id)
  needs_reset = env.buffers.done[step_t].astype(np.bool_, copy=True)
  terminal = sim_env.terminal_view(env.buffers)[step_t].copy()
  env._last_step_info = sim_env.SimStepInfo(terminal=terminal, step_t=step_t)
  post_step_frame = env.buffers.gamestate_view[env.cursor]
  # `needs_reset` refers to the transition at step_t. The current cursor already
  # points at the post-step frame, which is the state reward/eval code must use
  # for games that ended on this transition.
  if post_step_frame_out is not None:
    np.copyto(post_step_frame_out, post_step_frame)
  if terminal_game_batch is not None:
    if terminal_env_slice is None:
      raise ValueError('terminal_env_slice must be set with terminal_game_batch')
    terminal_game_batch.fill_slice(
        post_step_frame,
        needs_reset,
        terminal_env_slice,
        env._last_controllers,
        controller_slice=slice(None),
    )
  env._reset_finished_lanes_for_next_observation(needs_reset)
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


def write_step_counters(
    counters,
    worker_id: int,
    done: int,
    stockout: int,
    timeout: int,
    max_frame: int,
):
  base = int(worker_id) * 4
  counters[base] = int(done)
  counters[base + 1] = int(stockout)
  counters[base + 2] = int(timeout)
  counters[base + 3] = int(max_frame)


def write_step_timings(
    timings,
    worker_id: int,
    step_s: float,
    fill_s: float,
    obs_release_s: float,
):
  base = int(worker_id) * 3
  timings[base] = float(step_s)
  timings[base + 1] = float(fill_s)
  timings[base + 2] = float(obs_release_s)


def sum_step_counters(counters, workers: int) -> tuple[int, int, int, int]:
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


def sum_step_timings(timings, workers: int) -> tuple[float, float, float]:
  step_s = 0.0
  fill_s = 0.0
  obs_release_s = 0.0
  for worker_id in range(workers):
    base = worker_id * 3
    step_s += float(timings[base])
    fill_s += float(timings[base + 1])
    obs_release_s += float(timings[base + 2])
  return step_s, fill_s, obs_release_s


def barrier_wait(barrier, timeout: float, label: str):
  try:
    return barrier.wait(timeout=timeout)
  except Exception as exc:
    raise RuntimeError(f'timed out or broke barrier during {label}') from exc


def shared_action_buffer(total_players: int, array_factory) -> Controller:
  """Allocate the uint8 controller-bucket buffer written by policy inference."""
  shape = (int(total_players),)
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


def copy_action_to_shared_buffer(
    dst: Controller,
    src: Controller,
    spacing: tuple[int, int],
) -> int:
  """Copy sampled controller buckets into shared memory and count invalid bins."""
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


def default_controller_spacing(state: dict) -> tuple[int, int]:
  config = state['config']['embed']['controller']
  if config.get('type', 'default') != 'default':
    raise ValueError('sim env only supports the default controller embedding')
  default = config.get('default', config)
  return int(default['axis_spacing']), int(default['shoulder_spacing'])


def cycle_stages(batch_size: int, offset: int) -> np.ndarray:
  stages = sim_env.supported_stages()
  return np.asarray(
      [stages[(offset + i) % len(stages)] for i in range(batch_size)],
      dtype=object,
  )
