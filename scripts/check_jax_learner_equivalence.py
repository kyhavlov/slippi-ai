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
from slippi_ai import eval_lib
from slippi_ai import sim_env


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model-path', default='models/rl_doubles_v27_11000.pkl')
  parser.add_argument('--batch-size', type=int, default=8)
  parser.add_argument('--rollout-length', type=int, default=32)
  parser.add_argument('--minibatch-size', type=int, default=4)
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
  if args.batch_size <= 0 or args.minibatch_size < 0:
    raise ValueError('--batch-size must be positive and --minibatch-size nonnegative')
  total_packed = args.batch_size * 2
  if args.minibatch_size > 0 and total_packed % args.minibatch_size:
    raise ValueError('--minibatch-size must divide batch_size * 2')

  state = eval_lib.load_state(path=str(model_path))
  trajectory = _collect_one_trajectory(
      state=state,
      batch_size=args.batch_size,
      rollout_length=args.rollout_length,
      length=args.length,
      barrier_timeout=args.barrier_timeout,
  )

  full, _, _ = benchmark_jax_sim_rl._build_learner_and_actor(
      state=state,
      batch_size=total_packed,
      ppo_batches=1,
      ppo_epochs=1,
      learner_minibatch_size=0,
      learner_minibatch_scan_size=1,
      offload_minibatch_outputs=False,
      learning_rate=None,
      sample_temperature=1.0,
      learner_param_dtype=args.learner_param_dtype,
  )
  mini, _, _ = benchmark_jax_sim_rl._build_learner_and_actor(
      state=state,
      batch_size=total_packed,
      ppo_batches=1,
      ppo_epochs=1,
      learner_minibatch_size=args.minibatch_size,
      learner_minibatch_scan_size=4,
      offload_minibatch_outputs=False,
      learning_rate=None,
      sample_temperature=1.0,
      learner_param_dtype=args.learner_param_dtype,
  )

  initial_full = full.initial_state(total_packed)
  initial_mini = mini.initial_state(total_packed)
  step = int(state.get('step', 0))

  full_state, full_metrics = full.ppo([trajectory], initial_full, step=step)
  mini_state, mini_metrics = mini.ppo([trajectory], initial_mini, step=step)
  _block_until_ready((full_state, mini_state, full_metrics, mini_metrics))

  state_diff = _max_tree_diff(full.get_state(), mini.get_state())
  hidden_diff = _max_tree_diff(full_state, mini_state)
  metrics_diff = _compare_metric_summaries(full_metrics, mini_metrics)
  max_diff = max(
      state_diff['max_abs_diff'],
      hidden_diff['max_abs_diff'],
      metrics_diff['max_abs_diff'],
  )
  passed = max_diff <= args.atol
  result = {
      'passed': passed,
      'max_abs_diff': max_diff,
      'state_diff': state_diff,
      'hidden_diff': hidden_diff,
      'metrics_diff': metrics_diff,
      'atol': args.atol,
      'rtol': args.rtol,
      'batch_size': args.batch_size,
      'total_player_batch_size': total_packed,
      'rollout_length': args.rollout_length,
      'minibatch_size': args.minibatch_size,
      'learner_param_dtype': args.learner_param_dtype,
  }
  print(json.dumps(result, indent=2, sort_keys=True))
  if not passed:
    raise SystemExit(1)


def _collect_one_trajectory(
    *,
    state: dict,
    batch_size: int,
    rollout_length: int,
    length: int,
    barrier_timeout: float,
):
  ctx = mp.get_context('spawn')
  total_packed = batch_size * 2
  obs_owner = benchmark_sim_mp.SharedArrayOwner()
  packed = sim_env.make_packed_game_builder(batch_size, array_factory=obs_owner.array)
  action_owner = benchmark_sim_mp.SharedArrayOwner()
  action = benchmark_sim_mp._shared_encoded_controller(
      total_packed, action_owner.array)
  spacing = benchmark_sim_mp._default_controller_spacing(state)
  obs_barrier = ctx.Barrier(2)
  action_barrier = ctx.Barrier(2)
  stop_event = ctx.Event()
  step_counters = ctx.Array('i', 4, lock=False)
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
            obs_owner.specs,
            action_owner.specs,
            spacing,
            obs_barrier,
            action_barrier,
            stop_event,
            step_counters,
            barrier_timeout,
            result_queue,
        ),
    )
    process.start()

    learner, actor, name_code = benchmark_jax_sim_rl._build_learner_and_actor(
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
    del learner
    action_queue = deque(
        [benchmark_jax_sim_rl._to_numpy_tree(
            actor._policy.controller_head.dummy_sample_outputs([total_packed]))
         for _ in range(actor._policy.delay + 1)])

    benchmark_sim_mp._barrier_wait(
        obs_barrier, barrier_timeout, 'initial observations')
    trajectory, _ = benchmark_jax_sim_rl._collect_trajectory(
        actor=actor,
        packed=packed,
        action=action,
        action_queue=action_queue,
        action_barrier=action_barrier,
        obs_barrier=obs_barrier,
        step_counters=step_counters,
        workers=1,
        total_batch=batch_size,
        rollout_length=rollout_length,
        controller_spacing=spacing,
        name_code=name_code,
        barrier_timeout=barrier_timeout,
    )
    stop_event.set()
    action_barrier.abort()
    result_queue.get(timeout=30.0)
    process.join(timeout=10.0)
    if process.exitcode != 0:
      raise RuntimeError(f'worker exited with {process.exitcode}')
    return trajectory
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
    action_owner.close()
    obs_owner.unlink()
    action_owner.unlink()


def _block_until_ready(value):
  for leaf in jax.tree.leaves(value):
    if hasattr(leaf, 'block_until_ready'):
      leaf.block_until_ready()
    elif isinstance(leaf, np.ndarray):
      np.asarray(leaf)


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
