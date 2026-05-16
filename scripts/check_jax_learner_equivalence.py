import argparse
import json
import multiprocessing as mp
import traceback
from collections import deque
from pathlib import Path

import jax
import numpy as np

from scripts import benchmark_jax_sim_rl
from scripts import benchmark_sim_mp
from slippi_ai import data
from slippi_ai import eval_lib
from slippi_ai import sim_env
from slippi_ai import utils


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model-path', default='models/rl_doubles_v27_11000.pkl')
  parser.add_argument('--batch-size', type=int, default=8)
  parser.add_argument('--rollout-length', type=int, default=32)
  parser.add_argument('--actor-step-chunk-size', type=int, default=1)
  parser.add_argument('--ppo-batches', type=int, default=1)
  parser.add_argument('--matchup', choices=sim_env.SUPPORTED_MATCHUPS,
                      default='fox-falco')
  parser.add_argument('--minibatch-size', type=int, default=4)
  parser.add_argument('--minibatch-scan-size', type=int, default=4)
  parser.add_argument(
      '--compare-minibatch-paths',
      action=argparse.BooleanOptionalAction,
      default=True,
      help='Compare fused minibatch PPO against the non-fused minibatch path.')
  parser.add_argument(
      '--check-controller-math',
      action='store_true',
      help='Compare fast controller reducers against reference embedding math.')
  parser.add_argument(
      '--learner-param-dtype',
      choices=('float32', 'bfloat16'),
      default='float32')
  parser.add_argument('--length', type=int, default=64)
  parser.add_argument('--barrier-timeout', type=float, default=120.0)
  parser.add_argument('--atol', type=float, default=1e-4)
  parser.add_argument('--rtol', type=float, default=1e-4)
  args = parser.parse_args()

  model_path = Path(args.model_path)
  if not model_path.exists():
    raise FileNotFoundError(model_path)
  if args.rollout_length <= 21:
    raise ValueError('--rollout-length must exceed the policy delay')
  if args.batch_size <= 0 or args.ppo_batches <= 0 or args.minibatch_size < 0:
    raise ValueError('--batch-size must be positive and --minibatch-size nonnegative')
  total_packed = args.batch_size * 2
  if args.minibatch_size > 0 and total_packed % args.minibatch_size:
    raise ValueError('--minibatch-size must divide batch_size * 2')

  state = eval_lib.load_state(path=str(model_path))
  trajectories = _collect_trajectories(
      state=state,
      batch_size=args.batch_size,
      rollout_length=args.rollout_length,
      actor_step_chunk_size=args.actor_step_chunk_size,
      ppo_batches=args.ppo_batches,
      matchup=args.matchup,
      length=args.length,
      barrier_timeout=args.barrier_timeout,
  )

  fallback_scan_size = (
      args.ppo_batches * total_packed // args.minibatch_size + 1
      if args.minibatch_size > 0 else 1)
  full_minibatch_size = args.minibatch_size if args.compare_minibatch_paths else 0
  full_scan_size = fallback_scan_size if args.compare_minibatch_paths else 1
  full, _, _ = benchmark_jax_sim_rl._build_learner_and_actor(
      state=state,
      batch_size=total_packed,
      ppo_batches=args.ppo_batches,
      ppo_epochs=1,
      learner_minibatch_size=full_minibatch_size,
      learner_minibatch_scan_size=full_scan_size,
      offload_minibatch_outputs=False,
      learning_rate=None,
      sample_temperature=1.0,
      learner_param_dtype=args.learner_param_dtype,
  )
  mini, _, _ = benchmark_jax_sim_rl._build_learner_and_actor(
      state=state,
      batch_size=total_packed,
      ppo_batches=args.ppo_batches,
      ppo_epochs=1,
      learner_minibatch_size=args.minibatch_size,
      learner_minibatch_scan_size=args.minibatch_scan_size,
      offload_minibatch_outputs=False,
      learning_rate=None,
      sample_temperature=1.0,
      learner_param_dtype=args.learner_param_dtype,
  )

  initial_full = full.initial_state(total_packed)
  initial_mini = mini.initial_state(total_packed)
  step = int(state.get('step', 0))

  controller_math_diff = None
  if args.check_controller_math:
    controller_math_diff = _check_controller_math(
        mini, trajectories[0], initial_mini)

  full_state, full_metrics = full.ppo(trajectories, initial_full, step=step)
  mini_state, mini_metrics = mini.ppo(trajectories, initial_mini, step=step)
  _block_until_ready((full_state, mini_state, full_metrics, mini_metrics))

  state_diff = _max_tree_diff(full.get_state(), mini.get_state())
  hidden_diff = _max_tree_diff(full_state, mini_state)
  metrics_diff = _compare_metric_summaries(full_metrics, mini_metrics)
  max_diff = max(
      state_diff['max_abs_diff'],
      hidden_diff['max_abs_diff'],
      metrics_diff['max_abs_diff'],
  )
  if controller_math_diff is not None:
    max_diff = max(max_diff, controller_math_diff['max_abs_diff'])
  passed = max_diff <= args.atol
  result = {
      'passed': passed,
      'max_abs_diff': max_diff,
      'state_diff': state_diff,
      'hidden_diff': hidden_diff,
      'metrics_diff': metrics_diff,
      'controller_math_diff': controller_math_diff,
      'atol': args.atol,
      'rtol': args.rtol,
      'batch_size': args.batch_size,
      'total_player_batch_size': total_packed,
      'rollout_length': args.rollout_length,
      'actor_step_chunk_size': args.actor_step_chunk_size,
      'ppo_batches': args.ppo_batches,
      'matchup': args.matchup,
      'minibatch_size': args.minibatch_size,
      'minibatch_scan_size': args.minibatch_scan_size,
      'compare_minibatch_paths': args.compare_minibatch_paths,
      'check_controller_math': args.check_controller_math,
      'learner_param_dtype': args.learner_param_dtype,
  }
  print(json.dumps(result, indent=2, sort_keys=True))
  if not passed:
    raise SystemExit(1)


def _collect_trajectories(
    *,
    state: dict,
    batch_size: int,
    rollout_length: int,
    actor_step_chunk_size: int,
    ppo_batches: int,
    matchup: str,
    length: int,
    barrier_timeout: float,
):
  ctx = mp.get_context('spawn')
  total_packed = batch_size * 2
  obs_owner = benchmark_sim_mp.SharedArrayOwner()
  packed = sim_env.make_packed_game_builder(batch_size, array_factory=obs_owner.array)
  terminal_obs_owner = benchmark_sim_mp.SharedArrayOwner()
  terminal_packed = sim_env.make_packed_game_builder(
      batch_size, array_factory=terminal_obs_owner.array)
  action_owner = benchmark_sim_mp.SharedArrayOwner()
  action = benchmark_sim_mp._shared_encoded_controller(
      total_packed, action_owner.array)
  spacing = benchmark_sim_mp._default_controller_spacing(state)
  obs_barrier = ctx.Barrier(2)
  action_barrier = ctx.Barrier(2)
  stop_event = ctx.Event()
  step_counters = ctx.Array('i', 4, lock=False)
  step_timings = ctx.Array('d', 3, lock=False)
  result_queue = ctx.Queue()
  process = None

  try:
    process = ctx.Process(
        target=benchmark_sim_mp._worker_main,
        args=(
            0,
            batch_size,
            batch_size,
            0,
            length,
            28800,
            0,
            0,
            matchup,
            obs_owner.specs,
            terminal_obs_owner.specs,
            action_owner.specs,
            spacing,
            obs_barrier,
            action_barrier,
            stop_event,
            step_counters,
            step_timings,
            barrier_timeout,
            result_queue,
        ),
    )
    process.start()

    collect_learner, actor, name_code = benchmark_jax_sim_rl._build_learner_and_actor(
        state=state,
        batch_size=total_packed,
        ppo_batches=1,
        ppo_epochs=1,
        learner_minibatch_size=0,
        learner_minibatch_scan_size=1,
        offload_minibatch_outputs=False,
        learning_rate=None,
        sample_temperature=1.0,
        learner_param_dtype='float32',
    )
    dummy_outputs = actor._policy.controller_head.dummy_sample_outputs([total_packed])
    env_action_queue = deque(
        [benchmark_jax_sim_rl._to_numpy_tree(dummy_outputs.controller_state)
         for _ in range(actor._policy.delay)])
    learner_action_queue = deque(
        [benchmark_jax_sim_rl._to_numpy_tree(dummy_outputs)
         for _ in range(actor._policy.delay + 1)])

    benchmark_sim_mp._barrier_wait(
        obs_barrier, barrier_timeout, 'initial observations')
    trajectories = []
    for _ in range(ppo_batches):
      trajectory, _ = benchmark_jax_sim_rl._collect_trajectory(
          actor=actor,
          packed=packed,
          terminal_packed=terminal_packed,
          action=action,
          env_action_queue=env_action_queue,
          learner_action_queue=learner_action_queue,
          dummy_outputs=dummy_outputs,
          action_barrier=action_barrier,
          obs_barrier=obs_barrier,
          step_counters=step_counters,
          step_timings=step_timings,
          workers=1,
          total_batch=batch_size,
          rollout_length=rollout_length,
          actor_step_chunk_size=actor_step_chunk_size,
          async_rollout_inference=False,
          controller_spacing=spacing,
          name_code=name_code,
          reward_config=collect_learner._config.reward,
          barrier_timeout=barrier_timeout,
      )
      trajectories.append(trajectory)
    stop_event.set()
    action_barrier.abort()
    result_queue.get(timeout=30.0)
    process.join(timeout=10.0)
    if process.exitcode != 0:
      raise RuntimeError(f'worker exited with {process.exitcode}')
    return trajectories
  except BaseException:
    stop_event.set()
    try:
      action_barrier.abort()
    except Exception:
      pass
    traceback.print_exc()
    raise
  finally:
    if process is not None and process.is_alive():
      process.terminate()
    if process is not None:
      process.join(timeout=1.0)
    obs_owner.close()
    terminal_obs_owner.close()
    action_owner.close()
    obs_owner.unlink()
    terminal_obs_owner.unlink()
    action_owner.unlink()


def _block_until_ready(value):
  for leaf in jax.tree.leaves(value):
    if hasattr(leaf, 'block_until_ready'):
      leaf.block_until_ready()
    elif isinstance(leaf, np.ndarray):
      np.asarray(leaf)


def _check_controller_math(learner, trajectory, initial_state):
  delay = learner.policy.delay
  remove_first = lambda t: t[delay:] if delay > 0 else t
  remove_last = lambda t: t[:t.shape[0] - delay] if delay > 0 else t

  outputs, _ = learner._unroll_teacher_and_vf(
      trajectory, initial_state, train_value_function=False)
  actor_outputs = utils.map_single_structure(
      lambda t: t[1 + delay:], trajectory.actions)
  policy_frames = data.Frames(
      state_action=data.StateAction(
          state=jax.tree.map(remove_last, trajectory.states),
          action=jax.tree.map(remove_first, trajectory.actions.controller_state),
          name=remove_last(trajectory.name),
      ),
      is_resetting=remove_last(trajectory.is_resetting),
      reward=remove_first(trajectory.rewards),
  )
  policy_outputs = learner.policy.unroll(policy_frames, trajectory.initial_state)
  logit_outputs = learner.policy.unroll_logits(
      policy_frames, trajectory.initial_state)

  new_logits = logit_outputs.logits
  actor_logits = actor_outputs.logits
  teacher_logits = jax.tree.map(remove_last, outputs.teacher.logits)
  target_action = actor_outputs.controller_state

  comparisons = {
      'policy_logits': _max_tree_diff(policy_outputs.distances.logits, new_logits),
      'new_log_prob': _max_tree_diff(
          learner._get_log_prob_reference(new_logits, target_action),
          learner._get_log_prob(new_logits, target_action),
      ),
      'new_log_prob_vs_unroll': _max_tree_diff(
          policy_outputs.log_probs,
          learner._get_log_prob(new_logits, target_action),
      ),
      'actor_log_prob': _max_tree_diff(
          learner._get_log_prob_reference(actor_logits, target_action),
          learner._get_log_prob(actor_logits, target_action),
      ),
      'entropy': _max_tree_diff(
          learner._compute_entropy_reference(new_logits),
          learner._compute_entropy(new_logits),
      ),
      'actor_kl': _max_tree_diff(
          learner._compute_kl_reference(actor_logits, new_logits),
          learner._compute_kl(actor_logits, new_logits),
      ),
      'teacher_kl': _max_tree_diff(
          learner._compute_kl_reference(new_logits, teacher_logits),
          learner._compute_kl(new_logits, teacher_logits),
      ),
      'reverse_teacher_kl': _max_tree_diff(
          learner._compute_kl_reference(teacher_logits, new_logits),
          learner._compute_kl(teacher_logits, new_logits),
      ),
  }
  _block_until_ready(comparisons)
  max_abs = max(v['max_abs_diff'] for v in comparisons.values())
  max_rel = max(v['max_rel_diff'] for v in comparisons.values())
  return {
      'max_abs_diff': max_abs,
      'max_rel_diff': max_rel,
      'comparisons': comparisons,
  }


def _tree_leaves(tree):
  return jax.tree_util.tree_leaves(tree)


def _max_tree_diff(a, b):
  max_abs = 0.0
  max_rel = 0.0
  compared = 0
  for left, right in zip(_tree_leaves(a), _tree_leaves(b), strict=True):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
      raise ValueError(f'shape mismatch: {left.shape} != {right.shape}')
    if not np.issubdtype(left.dtype, np.number):
      continue
    if not np.issubdtype(right.dtype, np.number):
      continue
    diff = np.abs(left - right)
    denom = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-12)
    if diff.size:
      max_abs = max(max_abs, float(np.max(diff)))
      max_rel = max(max_rel, float(np.max(diff / denom)))
      compared += 1
  return {
      'max_abs_diff': max_abs,
      'max_rel_diff': max_rel,
      'leaves_compared': compared,
  }


def _compare_metric_summaries(full_metrics: dict, mini_metrics: dict):
  full_summary = _metric_summary(full_metrics)
  mini_summary = _metric_summary(mini_metrics)
  return _max_tree_diff(full_summary, mini_summary)


def _metric_summary(metrics: dict):
  post = metrics['post_update']
  return {
      key: _leaf_summary(post[key])
      for key in (
          'total_loss',
          'ppo_objective',
          'teacher_kl',
          'entropy',
          'actor_kl',
          'reverse_teacher_kl',
      )
      if key in post
  }


def _leaf_summary(value):
  if isinstance(value, dict) and {'mean', 'max'}.issubset(value):
    return {
        'mean': np.asarray(value['mean']),
        'max': np.asarray(value['max']),
        'min': np.asarray(value.get('min', value['max'])),
    }
  value = np.asarray(value)
  return {
      'mean': np.asarray(np.mean(value)),
      'max': np.asarray(np.max(value)),
      'min': np.asarray(np.min(value)),
  }


if __name__ == '__main__':
  main()
