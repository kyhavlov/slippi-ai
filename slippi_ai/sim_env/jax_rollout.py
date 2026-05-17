"""JAX rollout assembly on top of multiprocessing sim workers.

`JaxSimRolloutWorker` is the object that lets the normal JAX RL training loop
use melee-sim-light instead of Dolphin. It starts the worker processes from
`multiprocess_env.py`, keeps the policy delay queues, runs batched JAX inference
from shared observations, writes delayed controller actions back to shared
memory, and converts each rollout window into the same `Trajectory` shape the
learner already consumes.

The implementation is deliberately split from both `env.py` and the learner:
`env.py` handles one process worth of native sim state, `multiprocess_env.py`
handles shared-memory worker coordination, and this module owns rollout-level
concerns such as startup staggering, terminal reward correction, async policy
stepping, and the old-policy sample data needed by PPO.
"""

import concurrent.futures
import dataclasses
import multiprocessing as mp
import time
from collections import defaultdict, deque

import jax
import jax.numpy as jnp
import numpy as np

from slippi_ai import reward as reward_lib
from slippi_ai import sim_env
from slippi_ai import utils
from slippi_ai.evaluators import Trajectory
from slippi_ai.jax import agents as jax_agents
from slippi_ai.sim_env import multiprocess_env
from slippi_ai.sim_env import rewards as sim_rewards


class JaxSimRolloutWorker:
  """RolloutWorker-compatible wrapper around multiprocessing melee_sim shards.

  The learner sees the same Trajectory shape as Dolphin rollouts. Internally,
  worker processes own CPU sim batches while the main process owns JAX policy
  inference and shared-memory action/observation buffers.
  """

  ports = (1, 2)

  def __init__(
      self,
      *,
      actor: jax_agents.BasicAgent,
      total_batch: int,
      worker_batch_size: int,
      length: int,
      max_game_frames: int,
      character_pool: str,
      controller_spacing: tuple[int, int],
      name_code: np.ndarray,
      reward_config: reward_lib.RewardConfig,
      actor_step_chunk_size: int,
      async_rollout_inference: bool,
      initial_stagger_total_steps: int,
      barrier_timeout: float = 900.0,
      print_every: int = 0,
  ):
    if total_batch <= 0 or worker_batch_size <= 0:
      raise ValueError('total_batch and worker_batch_size must be positive')
    if total_batch % worker_batch_size:
      raise ValueError(
          f'total_batch={total_batch} must be divisible by worker_batch_size='
          f'{worker_batch_size}')
    self.actor = actor
    self.total_batch = int(total_batch)
    self.worker_batch_size = int(worker_batch_size)
    self.workers = self.total_batch // self.worker_batch_size
    self.length = int(length)
    self.max_game_frames = int(max_game_frames)
    self.character_pool = character_pool
    self.controller_spacing = controller_spacing
    self.name_code = np.asarray(name_code, dtype=np.int32)
    self.reward_config = reward_config
    self.actor_step_chunk_size = int(actor_step_chunk_size)
    self.async_rollout_inference = bool(async_rollout_inference)
    self.initial_stagger_total_steps = int(initial_stagger_total_steps)
    self.stagger_steps_per_worker = stagger_steps_per_worker(
        self.workers,
        self.initial_stagger_total_steps,
    )
    self.barrier_timeout = float(barrier_timeout)
    self.print_every = int(print_every)

    self._ctx = mp.get_context('spawn')
    self._processes = []
    self._started = False

  def start(self):
    if self._started:
      return
    # Observation and action buffers live in shared memory. Workers write Game
    # leaves in-place; the main process reads the same arrays for JAX inference.
    self._obs_owner = multiprocess_env.SharedArrayOwner()
    self.game_batch = sim_env.make_game_batch_buffers(
        self.total_batch,
        array_factory=self._obs_owner.array,
    )
    self._terminal_obs_owner = multiprocess_env.SharedArrayOwner()
    self.terminal_game_batch = sim_env.make_game_batch_buffers(
        self.total_batch,
        array_factory=self._terminal_obs_owner.array,
    )
    self._action_owner = multiprocess_env.SharedArrayOwner()
    self.action = multiprocess_env.shared_action_buffer(
        self.total_batch * 2,
        self._action_owner.array,
    )
    self._obs_barrier = self._ctx.Barrier(self.workers + 1)
    self._action_barrier = self._ctx.Barrier(self.workers + 1)
    self._stop_event = self._ctx.Event()
    self._step_counters = self._ctx.Array('i', self.workers * 4, lock=False)
    self._step_timings = self._ctx.Array('d', self.workers * 3, lock=False)
    self._active_worker_count = self._ctx.Value(
        'i',
        1 if self.stagger_steps_per_worker > 0 else self.workers,
        lock=False,
    )
    self._measure_worker_steps = self._ctx.Value(
        'b',
        self.stagger_steps_per_worker == 0,
        lock=False,
    )
    self._result_queue = self._ctx.Queue()
    self._processes = []

    for worker_id in range(self.workers):
      offset = worker_id * self.worker_batch_size
      process = self._ctx.Process(
          target=multiprocess_env.worker_main,
          args=(
              worker_id,
              self.worker_batch_size,
              self.total_batch,
              offset,
              self.length,
              self.max_game_frames,
              0,
              self.character_pool,
              self._obs_owner.specs,
              self._terminal_obs_owner.specs,
              self._action_owner.specs,
              self.controller_spacing,
              self._obs_barrier,
              self._action_barrier,
              self._stop_event,
              self._step_counters,
              self._step_timings,
              self.barrier_timeout,
              self._result_queue,
              self._active_worker_count,
              self._measure_worker_steps,
          ),
      )
      process.start()
      self._processes.append(process)

    dummy_outputs = self.actor._policy.controller_head.dummy_sample_outputs(
        [self.total_batch * 2])
    self._dummy_outputs = dummy_outputs
    # Model delay is represented by two queues: controller states for the env and
    # full sampled outputs for PPO old-policy data. They reset differently.
    self._env_action_queue = deque(
        [to_numpy_tree(dummy_outputs.controller_state)
         for _ in range(self.actor._policy.delay)])
    self._learner_action_queue = deque(
        [to_numpy_tree(dummy_outputs)
         for _ in range(self.actor._policy.delay + 1)])

    multiprocess_env.barrier_wait(
        self._obs_barrier,
        self.barrier_timeout,
        'initial observations',
    )
    self.initial_stagger = run_initial_stagger_warmup(
        actor=self.actor,
        game_batch=self.game_batch,
        action=self.action,
        env_action_queue=self._env_action_queue,
        learner_action_queue=self._learner_action_queue,
        dummy_outputs=self._dummy_outputs,
        active_worker_count=self._active_worker_count,
        measure_worker_steps=self._measure_worker_steps,
        action_barrier=self._action_barrier,
        obs_barrier=self._obs_barrier,
        step_counters=self._step_counters,
        step_timings=self._step_timings,
        workers=self.workers,
        stagger_steps=self.stagger_steps_per_worker,
        total_batch=self.total_batch,
        controller_spacing=self.controller_spacing,
        barrier_timeout=self.barrier_timeout,
        print_every=self.print_every,
    )
    self._started = True

  def reset_env(self):
    self.stop()
    self.start()

  def update_variables(self, updates):
    del updates

  def rollout(self, num_steps: int) -> tuple[Trajectory, dict]:
    trajectory, stats = collect_trajectory(
        actor=self.actor,
        game_batch=self.game_batch,
        terminal_game_batch=self.terminal_game_batch,
        action=self.action,
        env_action_queue=self._env_action_queue,
        learner_action_queue=self._learner_action_queue,
        dummy_outputs=self._dummy_outputs,
        action_barrier=self._action_barrier,
        obs_barrier=self._obs_barrier,
        step_counters=self._step_counters,
        step_timings=self._step_timings,
        workers=self.workers,
        total_batch=self.total_batch,
        rollout_length=num_steps,
        actor_step_chunk_size=self.actor_step_chunk_size,
        async_rollout_inference=self.async_rollout_inference,
        controller_spacing=self.controller_spacing,
        name_code=self.name_code,
        reward_config=self.reward_config,
        barrier_timeout=self.barrier_timeout,
    )
    return trajectory, {
        'timing': {
            'sim': timing_summary(stats['timings_sec']),
        },
        'counters': stats['counters'],
    }

  def stop(self):
    if not self._started:
      return
    self._stop_event.set()
    try:
      self._action_barrier.abort()
    except Exception:
      pass
    for process in self._processes:
      if process.is_alive():
        process.terminate()
      process.join(timeout=1.0)
    self._obs_owner.close()
    self._terminal_obs_owner.close()
    self._action_owner.close()
    self._obs_owner.unlink()
    self._terminal_obs_owner.unlink()
    self._action_owner.unlink()
    self._processes = []
    self._started = False


def initial_stagger_total_steps(workers: int, stagger_steps: int) -> int:
  if stagger_steps <= 0:
    return 0
  return int(workers) * int(stagger_steps)


def stagger_steps_per_worker(workers: int, total_steps: int) -> int:
  if total_steps <= 0:
    return 0
  workers = int(workers)
  return max(1, (int(total_steps) + workers - 1) // workers)


def active_workers_for_stagger_step(
    step: int,
    workers: int,
    stagger_steps: int,
) -> int:
  if stagger_steps <= 0:
    return int(workers)
  return min(int(workers), 1 + int(step) // int(stagger_steps))


def run_initial_stagger_warmup(
    *,
    actor: jax_agents.BasicAgent,
    game_batch,
    action,
    env_action_queue: deque,
    learner_action_queue: deque,
    dummy_outputs,
    active_worker_count,
    measure_worker_steps,
    action_barrier,
    obs_barrier,
    step_counters,
    step_timings,
    workers: int,
    stagger_steps: int,
    total_batch: int,
    controller_spacing: tuple[int, int],
    barrier_timeout: float,
    print_every: int,
) -> dict:
  # Activating workers gradually avoids every game starting from the exact same
  # opening frame distribution. The inactive workers still synchronize on the
  # barriers, so the main rollout loop does not need a separate startup path.
  total_steps = initial_stagger_total_steps(workers, stagger_steps)
  if total_steps == 0:
    active_worker_count.value = workers
    measure_worker_steps.value = True
    return dict(steps=0, elapsed_sec=0.0, counters={}, timings_sec={})

  measure_worker_steps.value = False
  timings = defaultdict(float)
  counters = defaultdict(int)
  start = time.perf_counter()
  report_every = max(1, int(print_every), int(stagger_steps))
  active_env_steps = 0
  for step in range(total_steps):
    active = active_workers_for_stagger_step(step, workers, stagger_steps)
    active_worker_count.value = active
    active_env_steps += active * (total_batch // workers)

    reset_start = time.perf_counter()
    reset_mask = np.asarray(game_batch.needs_reset, dtype=np.bool_)
    if np.any(reset_mask):
      reset_delay_queues(
          env_action_queue=env_action_queue,
          learner_action_queue=learner_action_queue,
          dummy_outputs=dummy_outputs,
          reset_mask=reset_mask,
      )
    reset_done = time.perf_counter()

    policy_start = time.perf_counter()
    sample_outputs = actor.step_device(game_batch.game, game_batch.needs_reset)
    policy_done = time.perf_counter()
    env_action_queue.append(to_numpy_tree(sample_outputs.controller_state))
    delayed_controller = env_action_queue.popleft()
    learner_action_queue.append(sample_outputs)
    learner_action_queue.popleft()

    invalid = multiprocess_env.copy_action_to_shared_buffer(
        action, delayed_controller, controller_spacing)
    action_done = time.perf_counter()
    multiprocess_env.barrier_wait(
        action_barrier, barrier_timeout, 'stagger action release')
    release_done = time.perf_counter()
    multiprocess_env.barrier_wait(
        obs_barrier, barrier_timeout, 'stagger observation wait')
    obs_done = time.perf_counter()

    done, stockout, timeout, max_frame = multiprocess_env.sum_step_counters(
        step_counters, workers)
    worker_step_s, worker_fill_s, _ = (
        multiprocess_env.sum_step_timings(step_timings, workers))
    timings['reset_queue_s'] += reset_done - reset_start
    timings['policy_sample_s'] += policy_done - policy_start
    timings['action_copy_s'] += action_done - policy_done
    timings['action_release_s'] += release_done - action_done
    timings['obs_wait_s'] += obs_done - release_done
    timings['env_step_s'] += worker_step_s
    timings['env_fill_s'] += worker_fill_s
    counters['invalid_actions'] += invalid
    counters['done'] += done
    counters['stockout'] += stockout
    counters['timeout'] += timeout
    counters['max_frame_reached'] += max_frame

    if print_every and (step + 1) % report_every == 0:
      elapsed = time.perf_counter() - start
      print(
          f'initial_stagger={step + 1}/{total_steps} '
          f'active_workers={active}/{workers} '
          f'env_steps_per_sec={active_env_steps / max(elapsed, 1e-9):.1f}',
          flush=True,
      )

  active_worker_count.value = workers
  measure_worker_steps.value = True
  return dict(
      steps=total_steps,
      active_env_steps=active_env_steps,
      elapsed_sec=time.perf_counter() - start,
      counters=dict(counters),
      timings_sec=dict(timings),
      timing_summary=timing_summary(timings),
  )


def timing_summary(timings: dict) -> dict:
  return {
      'total_s': float(timings.get('update_total_s', 0.0)),
      'rollout_s': float(timings.get('trajectory_collect_total_s', 0.0)),
      'learner_s': float(timings.get('learner_ppo_s', 0.0)),
      'agent_step_s': float(timings.get('policy_sample_s', 0.0)),
      'policy_wait_s': float(timings.get('policy_wait_s', 0.0)),
      'policy_dependency_wait_s': float(
          timings.get('policy_dependency_wait_s', 0.0)),
      'policy_blocking_s': float(
          timings.get('policy_sample_s', 0.0)
          + timings.get('policy_wait_s', 0.0)
          + timings.get('policy_dependency_wait_s', 0.0)),
      'env_step_s': float(timings.get('env_step_s', 0.0)),
      'env_fill_s': float(timings.get('env_fill_s', 0.0)),
      'obs_wait_s': float(timings.get('obs_wait_s', 0.0)),
      'action_copy_s': float(timings.get('action_copy_s', 0.0)),
      'action_release_s': float(timings.get('action_release_s', 0.0)),
      'state_copy_s': float(timings.get('state_copy_s', 0.0)),
      'game_tree_copy_s': float(timings.get('game_tree_copy_s', 0.0)),
      'reset_mask_copy_s': float(timings.get('reset_mask_copy_s', 0.0)),
      'trajectory_batch_states_s': float(
          timings.get('trajectory_batch_states_s', 0.0)),
      'trajectory_encode_states_s': float(
          timings.get('trajectory_encode_states_s', 0.0)),
      'trajectory_batch_actions_s': float(
          timings.get('trajectory_batch_actions_s', 0.0)),
      'trajectory_stack_resets_s': float(
          timings.get('trajectory_stack_resets_s', 0.0)),
      'trajectory_name_broadcast_s': float(
          timings.get('trajectory_name_broadcast_s', 0.0)),
      'reward_compute_s': float(timings.get('reward_compute_s', 0.0)),
  }


def collect_trajectory(
    *,
    actor: jax_agents.BasicAgent,
    game_batch,
    terminal_game_batch,
    action,
    env_action_queue: deque,
    learner_action_queue: deque,
    dummy_outputs,
    action_barrier,
    obs_barrier,
    step_counters,
    step_timings,
    workers: int,
    total_batch: int,
    rollout_length: int,
    actor_step_chunk_size: int,
    async_rollout_inference: bool,
    controller_spacing: tuple[int, int],
    name_code: np.ndarray,
    reward_config: reward_lib.RewardConfig,
    barrier_timeout: float,
) -> tuple[Trajectory, dict]:
  """Collect one rollout while preserving policy delay and terminal rewards."""
  states = []
  terminal_reward_overrides = []
  actions = []
  resets = []
  timings = defaultdict(float)
  counters = defaultdict(int)
  initial_state = actor.hidden_state()
  actor_step_chunk_size = max(1, int(actor_step_chunk_size))
  policy_executor = (
      concurrent.futures.ThreadPoolExecutor(max_workers=1)
      if async_rollout_inference and actor_step_chunk_size > 1 else None)
  pending_policy_future = None

  try:
    for chunk_start in range(0, rollout_length, actor_step_chunk_size):
      chunk_len = min(actor_step_chunk_size, rollout_length - chunk_start)
      chunk_inputs = []
      chunk_reset_masks = []
      env_queue_start = list(env_action_queue) if policy_executor is None else None

      for _ in range(chunk_len):
        state_start = time.perf_counter()
        # Snapshot only the leaf arrays. The shared Game buffers are immediately
        # reused by workers for the next observation.
        actor_state = to_numpy_tree(game_batch.game)
        game_copy_done = time.perf_counter()
        reset_mask = np.asarray(game_batch.needs_reset, dtype=np.bool_).copy()
        reset_copy_done = time.perf_counter()
        states.append(actor_state)
        resets.append(reset_mask)
        chunk_inputs.append((actor_state, reset_mask))
        chunk_reset_masks.append(reset_mask)

        if np.any(reset_mask):
          reset_delay_queues(
              env_action_queue=env_action_queue,
              learner_action_queue=learner_action_queue,
              dummy_outputs=dummy_outputs,
              reset_mask=reset_mask,
          )

        delayed_entry = env_action_queue.popleft()
        action_entry = learner_action_queue.popleft()
        wait_start = time.perf_counter()
        delayed_controller = resolve_env_action_entry(
            delayed_entry,
            dummy_outputs=dummy_outputs,
        )
        learner_action = _resolve_learner_action_entry(action_entry)
        wait_done = time.perf_counter()
        actions.append(learner_action)

        action_start = time.perf_counter()
        invalid = multiprocess_env.copy_action_to_shared_buffer(
            action, delayed_controller, controller_spacing)
        action_done = time.perf_counter()
        multiprocess_env.barrier_wait(
            action_barrier, barrier_timeout, 'action release')
        release_done = time.perf_counter()
        multiprocess_env.barrier_wait(
            obs_barrier, barrier_timeout, 'observation wait')
        obs_done = time.perf_counter()

        transition_state_start = time.perf_counter()
        next_reset_mask = np.asarray(game_batch.needs_reset, dtype=np.bool_).copy()
        terminal_reset_done = time.perf_counter()
        if np.any(next_reset_mask):
          # The worker resets lanes before the next observation is exposed.
          # Reward for the transition that just ended must use the pre-reset
          # post-step terminal frame instead.
          terminal_override_start = time.perf_counter()
          terminal_reward_overrides.append(sim_rewards.TerminalRewardOverride(
              transition_index=len(states) - 1,
              reset_mask=next_reset_mask,
              terminal_game=sim_rewards.masked_numpy_tree(
                  terminal_game_batch.game,
                  next_reset_mask,
              ),
          ))
          terminal_override_done = time.perf_counter()
          timings['terminal_override_copy_s'] += (
              terminal_override_done - terminal_override_start)

        done, stockout, timeout, max_frame = multiprocess_env.sum_step_counters(
            step_counters, workers)
        worker_step_s, worker_fill_s, _ = (
            multiprocess_env.sum_step_timings(step_timings, workers))

        timings['state_copy_s'] += reset_copy_done - state_start
        timings['game_tree_copy_s'] += game_copy_done - state_start
        timings['reset_mask_copy_s'] += reset_copy_done - game_copy_done
        timings['policy_wait_s'] += wait_done - wait_start
        timings['action_copy_s'] += action_done - action_start
        timings['action_release_s'] += release_done - action_done
        timings['obs_wait_s'] += obs_done - release_done
        timings['terminal_reward_state_s'] += (
            terminal_reset_done - transition_state_start)
        timings['terminal_reset_mask_copy_s'] += (
            terminal_reset_done - transition_state_start)
        timings['env_step_s'] += worker_step_s
        timings['env_fill_s'] += worker_fill_s
        counters['invalid_actions'] += invalid
        counters['done'] += done
        counters['stockout'] += stockout
        counters['timeout'] += timeout
        counters['max_frame_reached'] += max_frame

      if policy_executor is None:
        policy_start = time.perf_counter()
        # Batch several sequential actor steps into one compiled call. This
        # reduces Python/JAX launch overhead but still replays delay-queue reset
        # effects so the visible action stream matches single-step rollout.
        if chunk_len == 1:
          sample_outputs_list = [
              actor.step_device(chunk_inputs[0][0], chunk_inputs[0][1])]
        else:
          sample_outputs_list = actor.multi_step_device(chunk_inputs)
        policy_done = time.perf_counter()
        timings['policy_sample_s'] += policy_done - policy_start
        counters['policy_sample_calls'] += 1

        replace_env_action_queue_after_chunk(
            env_action_queue=env_action_queue,
            queue_start=env_queue_start,
            sample_outputs_list=sample_outputs_list,
            reset_masks=chunk_reset_masks,
            dummy_outputs=dummy_outputs,
        )
        for sample_outputs in sample_outputs_list:
          learner_action_queue.append(sample_outputs)
        continue

      dependency_wait_start = time.perf_counter()
      if pending_policy_future is not None:
        pending_policy_future.result()
      dependency_wait_done = time.perf_counter()
      timings['policy_dependency_wait_s'] += (
          dependency_wait_done - dependency_wait_start)

      # The actor carries recurrent state and previous-controller state, so
      # policy chunks must be launched in order. The overlap comes from stepping
      # the env with already-delayed actions while the current chunk is sampled.
      policy_start = time.perf_counter()
      pending_policy_future = policy_executor.submit(
          actor.multi_step_device,
          chunk_inputs,
      )
      policy_done = time.perf_counter()
      timings['policy_sample_s'] += policy_done - policy_start
      counters['policy_sample_calls'] += 1
      counters['async_policy_sample_calls'] += 1

      for index in range(chunk_len):
        env_action_queue.append(PendingEnvAction(pending_policy_future, index))
        learner_action_queue.append(
            PendingLearnerAction(pending_policy_future, index))
  finally:
    if pending_policy_future is not None:
      pending_policy_future.result()
    if policy_executor is not None:
      policy_executor.shutdown(wait=True)

  final_state_start = time.perf_counter()
  states.append(to_numpy_tree(game_batch.game))
  final_game_copy_done = time.perf_counter()
  resets.append(np.asarray(game_batch.needs_reset, dtype=np.bool_).copy())
  final_reset_copy_done = time.perf_counter()
  actions.append(_resolve_learner_action_entry(learner_action_queue[0]))
  final_action_done = time.perf_counter()
  _resolve_delay_queues(
      env_action_queue=env_action_queue,
      learner_action_queue=learner_action_queue,
      dummy_outputs=dummy_outputs,
  )
  delay_queue_done = time.perf_counter()
  delayed_actions = list(learner_action_queue)[1:]
  delayed_actions_done = time.perf_counter()
  trajectory, build_timings = _build_trajectory(
      actor=actor,
      states=states,
      reward_config=reward_config,
      terminal_reward_overrides=terminal_reward_overrides,
      actions=actions,
      resets=resets,
      initial_state=initial_state,
      delayed_actions=delayed_actions,
      name_code=name_code,
      rollout_length=rollout_length,
      total_batch=total_batch,
  )
  final_state_done = time.perf_counter()
  timings['state_copy_s'] += final_reset_copy_done - final_state_start
  timings['game_tree_copy_s'] += final_game_copy_done - final_state_start
  timings['reset_mask_copy_s'] += final_reset_copy_done - final_game_copy_done
  timings['final_action_resolve_s'] += final_action_done - final_reset_copy_done
  timings['delay_queue_resolve_s'] += delay_queue_done - final_action_done
  timings['delayed_actions_list_s'] += delayed_actions_done - delay_queue_done
  for key, value in build_timings.items():
    timings[key] += value
  timings['terminal_reward_state_s'] += build_timings['reward_compute_s']
  timings['trajectory_build_s'] += (
      final_state_done - final_state_start - build_timings['reward_compute_s'])
  return trajectory, {
      'timings_sec': dict(timings),
      'counters': dict(counters),
  }


def _build_trajectory(
    *,
    actor: jax_agents.BasicAgent,
    states: list,
    reward_config: reward_lib.RewardConfig,
    terminal_reward_overrides: list[sim_rewards.TerminalRewardOverride],
    actions: list,
    resets: list[np.ndarray],
    initial_state,
    delayed_actions: list,
    name_code: np.ndarray,
    rollout_length: int,
    total_batch: int,
) -> tuple[Trajectory, dict[str, float]]:
  # Encode observations and compute rewards after rollout collection, rather
  # than once per frame, to keep the CPU stepping loop focused on env progress.
  timings = {}
  batch_states_start = time.perf_counter()
  time_major_states = utils.batch_nest_nt(states)
  batch_states_done = time.perf_counter()
  encode_start = time.perf_counter()
  encoded_states = actor._policy.network.encode_game(time_major_states)
  encode_done = time.perf_counter()
  reward_start = time.perf_counter()
  rewards = sim_rewards.compute_transition_rewards(
      time_major_states,
      terminal_reward_overrides=terminal_reward_overrides,
      reward_config=reward_config,
  )
  reward_done = time.perf_counter()
  name_start = time.perf_counter()
  name = np.broadcast_to(
      np.asarray(name_code, dtype=np.int32),
      [rollout_length + 1, total_batch * 2],
  ).copy()
  name_done = time.perf_counter()
  actions_start = time.perf_counter()
  batched_actions = _batch_nest_jax(actions)
  actions_done = time.perf_counter()
  resets_start = time.perf_counter()
  is_resetting = np.stack(resets, axis=0)
  resets_done = time.perf_counter()
  timings['trajectory_batch_states_s'] = batch_states_done - batch_states_start
  timings['trajectory_encode_states_s'] = encode_done - encode_start
  timings['reward_compute_s'] = reward_done - reward_start
  timings['trajectory_name_broadcast_s'] = name_done - name_start
  timings['trajectory_batch_actions_s'] = actions_done - actions_start
  timings['trajectory_stack_resets_s'] = resets_done - resets_start
  return Trajectory(
      states=encoded_states,
      name=name,
      actions=batched_actions,
      rewards=rewards,
      is_resetting=is_resetting,
      initial_state=initial_state,
      delayed_actions=delayed_actions,
  ), timings

def to_numpy_tree(value):
  return utils.map_single_structure(lambda x: np.asarray(x).copy(), value)


def reset_delay_queue_lanes(queue: deque, default, reset_mask: np.ndarray):
  if not queue:
    return
  for index, value in enumerate(queue):
    if isinstance(value, PendingEnvAction):
      value.add_reset(reset_mask)
    else:
      queue[index] = _reset_tree_lanes(value, default, reset_mask)


def reset_delay_queues(
    *,
    env_action_queue: deque,
    learner_action_queue: deque,
    dummy_outputs,
    reset_mask: np.ndarray,
) -> None:
  # Fresh games should not receive delayed controller inputs from the game that
  # just ended. The learner queue is different: it contains historical actor
  # outputs for frames already collected into the trajectory, so rewriting it
  # corrupts PPO old-policy logits/actions around terminal boundaries.
  reset_delay_queue_lanes(
      env_action_queue,
      dummy_outputs.controller_state,
      reset_mask,
  )
  _ = learner_action_queue


@dataclasses.dataclass
class PendingEnvAction:
  future: concurrent.futures.Future
  index: int
  reset_mask: np.ndarray | None = None

  def add_reset(self, reset_mask: np.ndarray) -> None:
    reset_mask = np.asarray(reset_mask, dtype=np.bool_).copy()
    if self.reset_mask is None:
      self.reset_mask = reset_mask
    else:
      self.reset_mask = np.logical_or(self.reset_mask, reset_mask)


@dataclasses.dataclass(frozen=True)
class PendingLearnerAction:
  future: concurrent.futures.Future
  index: int


def resolve_env_action_entry(entry, *, dummy_outputs):
  if not isinstance(entry, PendingEnvAction):
    return entry
  sample_outputs = entry.future.result()[entry.index]
  controller = to_numpy_tree(sample_outputs.controller_state)
  if entry.reset_mask is not None and np.any(entry.reset_mask):
    controller = _reset_tree_lanes(
        controller,
        dummy_outputs.controller_state,
        entry.reset_mask,
    )
  return controller


def _resolve_learner_action_entry(entry):
  if not isinstance(entry, PendingLearnerAction):
    return entry
  return entry.future.result()[entry.index]


def _resolve_delay_queues(
    *,
    env_action_queue: deque,
    learner_action_queue: deque,
    dummy_outputs,
) -> None:
  for index, entry in enumerate(env_action_queue):
    env_action_queue[index] = resolve_env_action_entry(
        entry,
        dummy_outputs=dummy_outputs,
    )
  for index, entry in enumerate(learner_action_queue):
    learner_action_queue[index] = _resolve_learner_action_entry(entry)


def replace_env_action_queue_after_chunk(
    *,
    env_action_queue: deque,
    queue_start: list,
    sample_outputs_list: list,
    reset_masks: list[np.ndarray],
    dummy_outputs,
) -> None:
  """Replay single-frame env-delay queue updates after chunked sampling.

  During a chunk, env steps consume only controllers that were already delayed at
  the chunk start. New samples from the chunk cannot be consumed until at least
  `policy.delay` frames later, so policy inference can run after the env steps.
  Resets are the subtle case: the single-frame path clears queued env actions at
  the reset frame, including earlier samples from the same chunk. Replaying the
  queue updates after sampling preserves that final queue state.
  """
  env_action_queue.clear()
  env_action_queue.extend(queue_start)
  for sample_outputs, reset_mask in zip(sample_outputs_list, reset_masks):
    if np.any(reset_mask):
      reset_delay_queue_lanes(
          env_action_queue,
          dummy_outputs.controller_state,
          reset_mask,
      )
    env_action_queue.append(to_numpy_tree(sample_outputs.controller_state))
    env_action_queue.popleft()


def _reset_tree_lanes(value, default, reset_mask: np.ndarray):
  reset_mask = np.asarray(reset_mask, dtype=np.bool_)

  def reset_leaf(leaf, default_leaf):
    if isinstance(leaf, np.ndarray):
      reset = reset_mask
      while reset.ndim < leaf.ndim:
        reset = reset[..., None]
      return np.where(reset, np.asarray(default_leaf), leaf)
    leaf_array = jnp.asarray(leaf)
    reset = jnp.asarray(reset_mask)
    while reset.ndim < leaf_array.ndim:
      reset = reset[..., None]
    return jnp.where(reset, jnp.asarray(default_leaf), leaf_array)

  return utils.map_nt(reset_leaf, value, default)


def _batch_nest_jax(nests):
  return utils.map_nt(lambda *xs: jnp.stack(xs), *nests)


def block_until_ready(value):
  for leaf in jax.tree.leaves(value):
    if hasattr(leaf, 'block_until_ready'):
      leaf.block_until_ready()
    elif isinstance(leaf, np.ndarray):
      # NumPy arrays may wrap pending jax.copy_to_host_async results.
      np.asarray(leaf)
