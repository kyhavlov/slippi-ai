import argparse
import dataclasses
import json
import multiprocessing as mp
import pickle
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path

import jax
import numpy as np

from slippi_ai import eval_lib
from slippi_ai import sim_env
from slippi_ai import utils
from slippi_ai.jax.rl import build as rl_build
from slippi_ai.jax.rl import learner as learner_lib
from slippi_ai.sim_env import multiprocess_env
from slippi_ai.sim_env import jax_rollout


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model-path', default='models/rl_doubles_v27_11000.pkl')
  parser.add_argument('--workers', type=int, default=2)
  parser.add_argument('--batch-size', type=int, default=64,
                      help='Env batch per sim worker.')
  parser.add_argument('--rollout-length', type=int, default=64)
  parser.add_argument(
      '--actor-step-chunk-size',
      type=int,
      default=1,
      help=(
          'Number of sequential rollout observations to sample in one JAX '
          'actor call. Must be no larger than the policy delay.'))
  parser.add_argument(
      '--async-rollout-inference',
      action='store_true',
      help='Overlap rollout env stepping with delayed actor inference.')
  parser.add_argument('--updates', type=int, default=2)
  parser.add_argument('--warmup-updates', type=int, default=1)
  parser.add_argument('--ppo-batches', type=int, default=0,
                      help='Override checkpoint PPO batches. 0 uses config.')
  parser.add_argument('--ppo-epochs', type=int, default=0,
                      help='Override checkpoint PPO epochs. 0 uses config.')
  parser.add_argument('--learner-minibatch-size', type=int, default=512)
  parser.add_argument('--learner-minibatch-scan-size', type=int, default=4)
  parser.add_argument('--offload-minibatch-outputs', action='store_true')
  parser.add_argument('--profile-learner', action='store_true')
  parser.add_argument('--learning-rate', type=float, default=None,
                      help='Override checkpoint learner learning rate.')
  parser.add_argument('--policy-gradient-weight', type=float, default=None)
  parser.add_argument('--kl-teacher-weight', type=float, default=None)
  parser.add_argument('--value-cost', type=float, default=None)
  parser.add_argument('--reward-halflife', type=float, default=None)
  parser.add_argument('--reward-damage-ratio', type=float, default=None)
  parser.add_argument('--reward-stalling-penalty', type=float, default=None)
  parser.add_argument('--reward-stalling-threshold', type=float, default=None)
  parser.add_argument('--reward-approaching-factor', type=float, default=None)
  parser.add_argument('--reward-ledge-grab-penalty', type=float, default=None)
  parser.add_argument('--reward-zelda-penalty', type=float, default=None)
  parser.add_argument('--ppo-beta', type=float, default=None)
  parser.add_argument('--ppo-epsilon', type=float, default=None)
  parser.add_argument('--ppo-max-mean-actor-kl', type=float, default=None)
  parser.add_argument('--post-update-eval-interval', type=int, default=None)
  parser.add_argument('--revert-on-post-update-actor-kl', action='store_true')
  parser.add_argument('--optimizer-burnin-epochs', type=int, default=None)
  parser.add_argument('--value-burnin-epochs', type=int, default=None)
  parser.add_argument(
      '--learner-param-dtype',
      choices=('float32', 'bfloat16'),
      default='float32')
  parser.add_argument('--length', type=int, default=256,
                      help='Sim EnvBatch ring buffer length.')
  parser.add_argument('--max-game-frames', type=int, default=28800)
  parser.add_argument(
      '--initial-stagger-steps',
      type=int,
      default=0,
      help=(
          'Before training, activate one sim worker at a time and run this '
          'many policy-driven sim steps between activations.'))
  parser.add_argument('--matchup', choices=sim_env.SUPPORTED_MATCHUPS,
                      default='fox-falco')
  parser.add_argument('--sample-temperature', type=float, default=1.0)
  parser.add_argument('--barrier-timeout', type=float, default=900.0)
  parser.add_argument('--print-every', type=int, default=1)
  parser.add_argument('--save-path', default='')
  parser.add_argument('--save-every', type=int, default=0)
  parser.add_argument('--log-jsonl', default='')
  parser.add_argument(
      '--jax-trace-dir',
      default='',
      help='If set, capture a JAX profiler trace around one measured learner update.')
  parser.add_argument(
      '--jax-trace-measured-index',
      type=int,
      default=0,
      help='Zero-based measured update index to trace when --jax-trace-dir is set.')
  args = parser.parse_args()

  model_path = Path(args.model_path)
  if not model_path.exists():
    raise FileNotFoundError(model_path)
  if args.workers <= 0 or args.batch_size <= 0:
    raise ValueError('--workers and --batch-size must be positive')
  if args.rollout_length <= 0 or args.ppo_batches < 0 or args.updates <= 0:
    raise ValueError('--rollout-length and --updates must be positive')
  if args.initial_stagger_steps < 0:
    raise ValueError('--initial-stagger-steps must be non-negative')
  save_path = Path(args.save_path) if args.save_path else None
  log_path = Path(args.log_jsonl) if args.log_jsonl else None
  trace_path = Path(args.jax_trace_dir) if args.jax_trace_dir else None
  if save_path is not None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
  if log_path is not None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
  if trace_path is not None:
    trace_path.mkdir(parents=True, exist_ok=True)

  state = eval_lib.load_state(path=str(model_path))
  total_batch = args.workers * args.batch_size
  total_packed = total_batch * 2
  ctx = mp.get_context('spawn')

  obs_owner = multiprocess_env.SharedArrayOwner()
  packed = sim_env.make_packed_game_builder(total_batch, array_factory=obs_owner.array)
  terminal_obs_owner = multiprocess_env.SharedArrayOwner()
  terminal_packed = sim_env.make_packed_game_builder(
      total_batch, array_factory=terminal_obs_owner.array)
  action_owner = multiprocess_env.SharedArrayOwner()
  action = multiprocess_env.shared_encoded_controller(total_packed, action_owner.array)
  spacing = multiprocess_env.default_controller_spacing(state)

  obs_barrier = ctx.Barrier(args.workers + 1)
  action_barrier = ctx.Barrier(args.workers + 1)
  stop_event = ctx.Event()
  step_counters = ctx.Array('i', args.workers * 4, lock=False)
  step_timings = ctx.Array('d', args.workers * 3, lock=False)
  active_worker_count = ctx.Value(
      'i',
      1 if args.initial_stagger_steps > 0 else args.workers,
      lock=False,
  )
  measure_worker_steps = ctx.Value(
      'b',
      args.initial_stagger_steps == 0,
      lock=False,
  )
  result_queue = ctx.Queue()
  processes = []

  try:
    for worker_id in range(args.workers):
      offset = worker_id * args.batch_size
      process = ctx.Process(
          target=multiprocess_env.worker_main,
          args=(
              worker_id,
              args.batch_size,
              total_batch,
              offset,
              args.length,
              args.max_game_frames,
              0,
              0,
              args.matchup,
              obs_owner.specs,
              terminal_obs_owner.specs,
              action_owner.specs,
              spacing,
              obs_barrier,
              action_barrier,
              stop_event,
              step_counters,
              step_timings,
              args.barrier_timeout,
              result_queue,
              active_worker_count,
              measure_worker_steps,
          ),
      )
      process.start()
      processes.append(process)

    learner, actor, name_code = rl_build.build_learner_and_actor(
        state=state,
        batch_size=total_packed,
        ppo_batches=args.ppo_batches,
        ppo_epochs=args.ppo_epochs,
        learner_minibatch_size=args.learner_minibatch_size,
        learner_minibatch_scan_size=args.learner_minibatch_scan_size,
        offload_minibatch_outputs=args.offload_minibatch_outputs,
        learning_rate=args.learning_rate,
        policy_gradient_weight=args.policy_gradient_weight,
        kl_teacher_weight=args.kl_teacher_weight,
        value_cost=args.value_cost,
        reward_halflife=args.reward_halflife,
        reward_damage_ratio=args.reward_damage_ratio,
        reward_stalling_penalty=args.reward_stalling_penalty,
        reward_stalling_threshold=args.reward_stalling_threshold,
        reward_approaching_factor=args.reward_approaching_factor,
        reward_ledge_grab_penalty=args.reward_ledge_grab_penalty,
        reward_zelda_penalty=args.reward_zelda_penalty,
        ppo_beta=args.ppo_beta,
        ppo_epsilon=args.ppo_epsilon,
        ppo_max_mean_actor_kl=args.ppo_max_mean_actor_kl,
        post_update_eval_interval=args.post_update_eval_interval,
        revert_on_post_update_actor_kl=args.revert_on_post_update_actor_kl,
        optimizer_burnin_epochs=args.optimizer_burnin_epochs,
        value_burnin_epochs=args.value_burnin_epochs,
        sample_temperature=args.sample_temperature,
        learner_param_dtype=args.learner_param_dtype,
    )
    ppo_batches = learner._config.ppo.num_batches
    if args.rollout_length <= actor._policy.delay:
      raise ValueError(
          f'--rollout-length must be greater than policy delay '
          f'{actor._policy.delay}, got {args.rollout_length}')
    if args.actor_step_chunk_size <= 0:
      raise ValueError('--actor-step-chunk-size must be positive')
    if args.actor_step_chunk_size > actor._policy.delay:
      raise ValueError(
          f'--actor-step-chunk-size must be <= policy delay '
          f'{actor._policy.delay}, got {args.actor_step_chunk_size}')
    learner_state = learner.initial_state(total_packed)
    dummy_outputs = actor._policy.controller_head.dummy_sample_outputs([total_packed])
    env_action_queue = deque(
        [jax_rollout.to_numpy_tree(dummy_outputs.controller_state)
         for _ in range(actor._policy.delay)])
    learner_action_queue = deque(
        [jax_rollout.to_numpy_tree(dummy_outputs) for _ in range(actor._policy.delay + 1)])

    multiprocess_env.barrier_wait(
        obs_barrier, args.barrier_timeout, 'initial observations')
    initial_stagger = jax_rollout.run_initial_stagger_warmup(
        actor=actor,
        packed=packed,
        action=action,
        env_action_queue=env_action_queue,
        learner_action_queue=learner_action_queue,
        dummy_outputs=dummy_outputs,
        active_worker_count=active_worker_count,
        measure_worker_steps=measure_worker_steps,
        action_barrier=action_barrier,
        obs_barrier=obs_barrier,
        step_counters=step_counters,
        step_timings=step_timings,
        workers=args.workers,
        stagger_steps=args.initial_stagger_steps,
        total_batch=total_batch,
        controller_spacing=spacing,
        barrier_timeout=args.barrier_timeout,
        print_every=args.print_every,
    )

    timings = defaultdict(float)
    counters = defaultdict(int)
    measured_updates = 0
    measured_rollout_steps = 0
    total_updates = args.warmup_updates + args.updates
    measured_start = None
    last_metrics = None

    for update_index in range(total_updates):
      measuring = update_index >= args.warmup_updates
      if measuring and measured_start is None:
        measured_start = time.perf_counter()

      trajectories = []
      phase_times = None
      if log_path is not None:
        phase_times = {
            'update_start_perf': time.perf_counter(),
            'update_start_unix_ns': time.time_ns(),
            'rollout_batches': [],
        }
      update_timings = defaultdict(float)
      update_counters = defaultdict(int)
      update_start = time.perf_counter()
      for batch_index in range(ppo_batches):
        if phase_times is not None:
          rollout_phase = {
              'batch_index': batch_index,
              'start_perf': time.perf_counter(),
              'start_unix_ns': time.time_ns(),
          }
        rollout_start = time.perf_counter()
        trajectory, rollout_stats = jax_rollout.collect_trajectory(
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
            workers=args.workers,
            total_batch=total_batch,
            rollout_length=args.rollout_length,
            actor_step_chunk_size=args.actor_step_chunk_size,
            async_rollout_inference=args.async_rollout_inference,
            controller_spacing=spacing,
            name_code=name_code,
            reward_config=learner._config.reward,
            barrier_timeout=args.barrier_timeout,
        )
        rollout_done = time.perf_counter()
        if phase_times is not None:
          rollout_phase['end_perf'] = time.perf_counter()
          rollout_phase['end_unix_ns'] = time.time_ns()
          phase_times['rollout_batches'].append(rollout_phase)
        trajectories.append(trajectory)
        for key, value in rollout_stats['timings_sec'].items():
          update_timings[key] += value
        for key, value in rollout_stats['counters'].items():
          update_counters[key] += value
        update_timings['trajectory_collect_total_s'] += rollout_done - rollout_start
        if measuring:
          for key, value in rollout_stats['timings_sec'].items():
            timings[key] += value
          for key, value in rollout_stats['counters'].items():
            counters[key] += value
          timings['trajectory_collect_total_s'] += rollout_done - rollout_start
          measured_rollout_steps += args.rollout_length

      step_number = int(state.get('step', 0)) + update_index + 1
      trace_this_update = (
          trace_path is not None
          and measuring
          and measured_updates == args.jax_trace_measured_index)
      learner_start = time.perf_counter()
      if phase_times is not None:
        phase_times['learner_start_perf'] = learner_start
        phase_times['learner_start_unix_ns'] = time.time_ns()
      if trace_this_update:
        jax.profiler.start_trace(str(trace_path))
        try:
          with jax.profiler.StepTraceAnnotation(
              'learner_ppo', step_num=step_number):
            learner_state, last_metrics = learner.ppo(
                trajectories,
                learner_state,
                step=int(state.get('step', 0)) + update_index,
                recompute_rewards=False,
                profile=args.profile_learner and measuring)
            jax_rollout.block_until_ready((learner_state, last_metrics))
        finally:
          jax.profiler.stop_trace()
      else:
        learner_state, last_metrics = learner.ppo(
            trajectories,
            learner_state,
            step=int(state.get('step', 0)) + update_index,
            recompute_rewards=False,
            profile=args.profile_learner and measuring)
        jax_rollout.block_until_ready((learner_state, last_metrics))
      learner_done = time.perf_counter()
      if phase_times is not None:
        phase_times['learner_end_perf'] = learner_done
        phase_times['learner_end_unix_ns'] = time.time_ns()
      update_timings['learner_ppo_s'] += learner_done - learner_start
      update_timings['update_total_s'] += learner_done - update_start
      if phase_times is not None:
        phase_times['update_end_perf'] = time.perf_counter()
        phase_times['update_end_unix_ns'] = time.time_ns()

      if measuring:
        timings['learner_ppo_s'] += learner_done - learner_start
        measured_updates += 1

      update_env_steps = total_batch * args.rollout_length
      update_player_frames = total_packed * args.rollout_length
      update_total_s = max(update_timings['update_total_s'], 1e-9)
      timing_summary = jax_rollout.timing_summary(update_timings)
      if log_path is not None:
        _append_jsonl(log_path, {
            'update': step_number,
            'phase': 'measure' if measuring else 'warmup',
            'env_steps': update_env_steps,
            'player_frames': update_player_frames,
            'env_steps_per_sec': update_env_steps / update_total_s,
            'player_frames_per_sec': update_player_frames / update_total_s,
            'timing_summary': timing_summary,
            'timings_sec': dict(update_timings),
            'phase_times': phase_times,
            'counters': dict(update_counters),
            'metrics': _jsonable_metrics(last_metrics),
        })
      if save_path is not None and args.save_every > 0 and step_number % args.save_every == 0:
        _save_checkpoint(save_path.with_name(
            f'{save_path.stem}_{step_number}{save_path.suffix}'),
            learner=learner,
            source_state=state,
            step=step_number)

      if args.print_every and (update_index + 1) % args.print_every == 0:
        phase = 'measure' if measuring else 'warmup'
        print(
            f'update={update_index + 1}/{total_updates} phase={phase} '
            f'env_steps_per_sec={update_env_steps / update_total_s:.1f} '
            f'rollout_s={timing_summary["rollout_s"]:.3f} '
            f'learner_s={timing_summary["learner_s"]:.3f}',
            flush=True,
        )

    measured_end = time.perf_counter()
    stop_event.set()
    action_barrier.abort()
    worker_results = [result_queue.get(timeout=30.0) for _ in processes]
    for process in processes:
      process.join(timeout=10.0)
      if process.exitcode != 0:
        raise RuntimeError(f'worker {process.pid} exited with {process.exitcode}')

    measured_elapsed = 0.0 if measured_start is None else measured_end - measured_start
    measured_env_steps = measured_rollout_steps * total_batch
    measured_player_frames = measured_rollout_steps * total_packed
    summary = {
        'benchmark': {
            'model_path': str(model_path),
            'workers': args.workers,
            'batch_size_per_worker': args.batch_size,
            'total_env_batch_size': total_batch,
            'total_player_batch_size': total_packed,
            'rollout_length': args.rollout_length,
            'actor_step_chunk_size': args.actor_step_chunk_size,
            'async_rollout_inference': args.async_rollout_inference,
            'ppo_batches': ppo_batches,
            'ppo_epochs': learner._config.ppo.num_epochs,
            'learner_minibatch_size': args.learner_minibatch_size,
            'learner_minibatch_scan_size': args.learner_minibatch_scan_size,
            'offload_minibatch_outputs': args.offload_minibatch_outputs,
            'profile_learner': args.profile_learner,
            'updates': args.updates,
            'warmup_updates': args.warmup_updates,
            'policy_delay': actor._policy.delay,
            'initial_stagger_steps': args.initial_stagger_steps,
            'initial_stagger': initial_stagger,
            'matchup': args.matchup,
            'learning_rate': learner._config.learning_rate,
            'policy_gradient_weight': learner._config.policy_gradient_weight,
            'kl_teacher_weight': learner._config.kl_teacher_weight,
            'value_cost': learner._config.value_cost,
            'reward_halflife': learner._config.reward_halflife,
            'reward': dataclasses.asdict(learner._config.reward),
            'ppo_beta': learner._config.ppo.beta,
            'ppo_epsilon': learner._config.ppo.epsilon,
            'ppo_max_mean_actor_kl': learner._config.ppo.max_mean_actor_kl,
            'post_update_eval_interval': (
                learner._config.ppo.post_update_eval_interval),
            'revert_on_post_update_actor_kl': (
                learner._config.ppo.revert_on_post_update_actor_kl),
            'optimizer_burnin_epochs': learner._config.optimizer_burnin_epochs,
            'value_burnin_epochs': learner._config.value_burnin_epochs,
            'learner_param_dtype': args.learner_param_dtype,
            'measured_elapsed_sec': measured_elapsed,
            'measured_updates': measured_updates,
            'measured_rollout_steps': measured_rollout_steps,
            'measured_env_steps': measured_env_steps,
            'measured_player_frames': measured_player_frames,
            'env_steps_per_sec': measured_env_steps / max(measured_elapsed, 1e-9),
            'player_frames_per_sec': measured_player_frames / max(measured_elapsed, 1e-9),
            'ns_per_env_step': measured_elapsed * 1e9 / max(1, measured_env_steps),
            'updates_per_sec': measured_updates / max(measured_elapsed, 1e-9),
        },
        'main_timings_sec': dict(timings),
        'main_counters': dict(counters),
        'worker_results': worker_results,
        'last_metrics': _jsonable_metrics(last_metrics),
    }
    if save_path is not None:
      _save_checkpoint(save_path, learner=learner, source_state=state,
                       step=int(state.get('step', 0)) + total_updates)
    print(json.dumps(summary, indent=2, sort_keys=True))
  except BaseException:
    stop_event.set()
    try:
      action_barrier.abort()
    except Exception:
      pass
    traceback.print_exc()
    raise
  finally:
    for process in processes:
      if process.is_alive():
        process.terminate()
      process.join(timeout=1.0)
    obs_owner.close()
    terminal_obs_owner.close()
    action_owner.close()
    obs_owner.unlink()
    terminal_obs_owner.unlink()
    action_owner.unlink()


def _save_checkpoint(
    path: Path,
    *,
    learner: learner_lib.Learner,
    source_state: dict,
    step: int,
):
  path.parent.mkdir(parents=True, exist_ok=True)
  combined_state = dict(
      state=learner.get_state(),
      config=source_state['config'],
      name_map=source_state['name_map'],
      step=int(step),
      rl_config=dict(learner=dataclasses.asdict(learner._config)),
  )
  with open(path, 'wb') as f:
    pickle.dump(combined_state, f)


def _append_jsonl(path: Path, row: dict):
  with open(path, 'a') as f:
    f.write(json.dumps(row, sort_keys=True))
    f.write('\n')


def _jsonable_metrics(metrics):
  if metrics is None:
    return None
  post = metrics.get('post_update', {})
  return {
      'post_update': {
          'actor_kl': _jsonable_leaf(post.get('actor_kl')),
          'total_loss': _jsonable_leaf(post.get('total_loss')),
          'teacher_kl': _jsonable_leaf(post.get('teacher_kl')),
          'entropy': _jsonable_leaf(post.get('entropy')),
          'ppo_objective': _jsonable_leaf(post.get('ppo_objective')),
      },
      'value': _jsonable_leaf(metrics.get('value')),
      'profile_sec': _jsonable_leaf(metrics.get('profile_sec')),
      'reverted': bool(metrics.get('reverted', False)),
      'post_update_evaluated': bool(
          metrics.get('post_update_evaluated', False)),
  }


def _jsonable_leaf(value):
  def convert(x):
    if isinstance(x, jax.Array):
      x = np.asarray(x)
    if isinstance(x, np.ndarray):
      if x.shape == ():
        return x.item()
      return {
          'shape': list(x.shape),
          'mean': float(np.mean(x)),
          'min': float(np.min(x)),
          'max': float(np.max(x)),
      }
    if isinstance(x, np.generic):
      return x.item()
    return x

  return utils.map_single_structure(convert, value)


if __name__ == '__main__':
  main()
