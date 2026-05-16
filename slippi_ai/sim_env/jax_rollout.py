import concurrent.futures
import dataclasses
import time
from collections import defaultdict, deque

import jax
import jax.numpy as jnp
import numpy as np

from slippi_ai import reward as reward_lib
from slippi_ai import utils
from slippi_ai.evaluators import Trajectory
from slippi_ai.jax import agents as jax_agents
from slippi_ai.sim_env import multiprocess_env


def initial_stagger_total_steps(workers: int, stagger_steps: int) -> int:
  if stagger_steps <= 0:
    return 0
  return int(workers) * int(stagger_steps)


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
    packed,
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
    reset_mask = np.asarray(packed.needs_reset, dtype=np.bool_)
    if np.any(reset_mask):
      reset_delay_queues(
          env_action_queue=env_action_queue,
          learner_action_queue=learner_action_queue,
          dummy_outputs=dummy_outputs,
          reset_mask=reset_mask,
      )
    reset_done = time.perf_counter()

    policy_start = time.perf_counter()
    sample_outputs = actor.step_device(packed.game, packed.needs_reset)
    policy_done = time.perf_counter()
    env_action_queue.append(to_numpy_tree(sample_outputs.controller_state))
    delayed_controller = env_action_queue.popleft()
    learner_action_queue.append(sample_outputs)
    learner_action_queue.popleft()

    invalid = multiprocess_env.copy_controller(
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
  }


def collect_trajectory(
    *,
    actor: jax_agents.BasicAgent,
    packed,
    terminal_packed,
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
        actor_state = to_numpy_tree(packed.game)
        reset_mask = np.asarray(packed.needs_reset, dtype=np.bool_).copy()
        states.append(actor_state)
        resets.append(reset_mask)
        chunk_inputs.append((actor_state, reset_mask))
        chunk_reset_masks.append(reset_mask)
        state_done = time.perf_counter()

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
        invalid = multiprocess_env.copy_controller(
            action, delayed_controller, controller_spacing)
        action_done = time.perf_counter()
        multiprocess_env.barrier_wait(
            action_barrier, barrier_timeout, 'action release')
        release_done = time.perf_counter()
        multiprocess_env.barrier_wait(
            obs_barrier, barrier_timeout, 'observation wait')
        obs_done = time.perf_counter()

        transition_state_start = time.perf_counter()
        next_reset_mask = np.asarray(packed.needs_reset, dtype=np.bool_).copy()
        if np.any(next_reset_mask):
          terminal_reward_overrides.append(TerminalRewardOverride(
              transition_index=len(states) - 1,
              reset_mask=next_reset_mask,
              terminal_game=masked_numpy_tree(
                  terminal_packed.game,
                  next_reset_mask,
              ),
          ))
        transition_state_done = time.perf_counter()

        done, stockout, timeout, max_frame = multiprocess_env.sum_step_counters(
            step_counters, workers)
        worker_step_s, worker_fill_s, _ = (
            multiprocess_env.sum_step_timings(step_timings, workers))

        timings['state_copy_s'] += state_done - state_start
        timings['policy_wait_s'] += wait_done - wait_start
        timings['action_copy_s'] += action_done - action_start
        timings['action_release_s'] += release_done - action_done
        timings['obs_wait_s'] += obs_done - release_done
        timings['terminal_reward_state_s'] += (
            transition_state_done - transition_state_start)
        timings['env_step_s'] += worker_step_s
        timings['env_fill_s'] += worker_fill_s
        counters['invalid_actions'] += invalid
        counters['done'] += done
        counters['stockout'] += stockout
        counters['timeout'] += timeout
        counters['max_frame_reached'] += max_frame

      if policy_executor is None:
        policy_start = time.perf_counter()
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
  states.append(to_numpy_tree(packed.game))
  resets.append(np.asarray(packed.needs_reset, dtype=np.bool_).copy())
  actions.append(_resolve_learner_action_entry(learner_action_queue[0]))
  _resolve_delay_queues(
      env_action_queue=env_action_queue,
      learner_action_queue=learner_action_queue,
      dummy_outputs=dummy_outputs,
  )
  trajectory, reward_compute_s = _build_trajectory(
      actor=actor,
      states=states,
      reward_config=reward_config,
      terminal_reward_overrides=terminal_reward_overrides,
      actions=actions,
      resets=resets,
      initial_state=initial_state,
      delayed_actions=list(learner_action_queue)[1:],
      name_code=name_code,
      rollout_length=rollout_length,
      total_batch=total_batch,
  )
  final_state_done = time.perf_counter()
  timings['terminal_reward_state_s'] += reward_compute_s
  timings['trajectory_build_s'] += (
      final_state_done - final_state_start - reward_compute_s)
  return trajectory, {
      'timings_sec': dict(timings),
      'counters': dict(counters),
  }


def _build_trajectory(
    *,
    actor: jax_agents.BasicAgent,
    states: list,
    reward_config: reward_lib.RewardConfig,
    terminal_reward_overrides: list['TerminalRewardOverride'],
    actions: list,
    resets: list[np.ndarray],
    initial_state,
    delayed_actions: list,
    name_code: np.ndarray,
    rollout_length: int,
    total_batch: int,
) -> tuple[Trajectory, float]:
  time_major_states = utils.batch_nest_nt(states)
  encoded_states = actor._policy.network.encode_game(time_major_states)
  reward_start = time.perf_counter()
  rewards = batched_transition_rewards(
      time_major_states,
      terminal_reward_overrides=terminal_reward_overrides,
      reward_config=reward_config,
  )
  reward_done = time.perf_counter()
  return Trajectory(
      states=encoded_states,
      name=np.broadcast_to(
          np.asarray(name_code, dtype=np.int32),
          [rollout_length + 1, total_batch * 2],
      ).copy(),
      actions=_batch_nest_jax(actions),
      rewards=rewards,
      is_resetting=np.stack(resets, axis=0),
      initial_state=initial_state,
      delayed_actions=delayed_actions,
  ), reward_done - reward_start


def transition_reward(state, next_state, reward_config: reward_lib.RewardConfig):
  transition = utils.batch_nest_nt([state, to_numpy_tree(next_state)])
  return reward_lib.compute_rewards(
      transition,
      **dataclasses.asdict(reward_config))[0]


@dataclasses.dataclass(frozen=True)
class TerminalRewardOverride:
  transition_index: int
  reset_mask: np.ndarray
  terminal_game: object


def masked_numpy_tree(value, mask: np.ndarray):
  mask = np.asarray(mask, dtype=np.bool_)
  return utils.map_single_structure(lambda x: np.asarray(x)[mask].copy(), value)


def batched_transition_rewards(
    time_major_states,
    *,
    terminal_reward_overrides: list[TerminalRewardOverride],
    reward_config: reward_lib.RewardConfig,
) -> np.ndarray:
  """Compute all per-transition rewards in one vectorized reward pass.

  Terminal frames are only valid as the next frame for the transition that just
  ended. The following transition must still start from the post-reset state, so
  this builds a pair-shaped game: [current_or_seed, corrected_next] x T x B.
  """

  if terminal_reward_overrides:
    terminal_games = [override.terminal_game
                      for override in terminal_reward_overrides]

    def pair_leaf(leaf, *terminal_leaves):
      next_leaf = np.array(leaf[1:], copy=True)
      for override, terminal_leaf in zip(
          terminal_reward_overrides,
          terminal_leaves,
      ):
        next_leaf[override.transition_index, override.reset_mask] = terminal_leaf
      return np.stack([leaf[:-1], next_leaf], axis=0)

    transition_pairs = utils.map_nt(pair_leaf, time_major_states, *terminal_games)
  else:
    transition_pairs = utils.map_single_structure(
        lambda leaf: np.stack([leaf[:-1], leaf[1:]], axis=0),
        time_major_states,
    )

  return reward_lib.compute_rewards(
      transition_pairs,
      **dataclasses.asdict(reward_config))[0]


def terminal_corrected_game(*, reset_game, terminal_game, needs_reset):
  needs_reset = np.asarray(needs_reset, dtype=np.bool_)

  def select(reset_leaf, terminal_leaf):
    reset = needs_reset
    reset_leaf = np.asarray(reset_leaf)
    while reset.ndim < reset_leaf.ndim:
      reset = reset[..., None]
    return np.where(reset, np.asarray(terminal_leaf), reset_leaf)

  return utils.map_nt(select, reset_game, terminal_game)


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
