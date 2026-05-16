import argparse
import concurrent.futures
import copy
import dataclasses
import json
import multiprocessing as mp
import pickle
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from scripts import benchmark_sim_mp
from slippi_ai import eval_lib
from slippi_ai import reward as reward_lib
from slippi_ai import sim_env
from slippi_ai import utils
from slippi_ai.evaluators import Trajectory
from slippi_ai.flag_utils import dataclass_from_dict
from slippi_ai.jax import agents as jax_agents
from slippi_ai.jax import jax_utils
from slippi_ai.jax import networks
from slippi_ai.jax import tf_checkpoint
from slippi_ai.jax import train_lib
from slippi_ai.jax.rl import learner as learner_lib


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

  obs_owner = benchmark_sim_mp.SharedArrayOwner()
  packed = sim_env.make_packed_game_builder(total_batch, array_factory=obs_owner.array)
  terminal_obs_owner = benchmark_sim_mp.SharedArrayOwner()
  terminal_packed = sim_env.make_packed_game_builder(
      total_batch, array_factory=terminal_obs_owner.array)
  action_owner = benchmark_sim_mp.SharedArrayOwner()
  action = benchmark_sim_mp._shared_encoded_controller(total_packed, action_owner.array)
  spacing = benchmark_sim_mp._default_controller_spacing(state)

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
          target=benchmark_sim_mp._worker_main,
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

    learner, actor, name_code = _build_learner_and_actor(
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
        [_to_numpy_tree(dummy_outputs.controller_state)
         for _ in range(actor._policy.delay)])
    learner_action_queue = deque(
        [_to_numpy_tree(dummy_outputs) for _ in range(actor._policy.delay + 1)])

    benchmark_sim_mp._barrier_wait(
        obs_barrier, args.barrier_timeout, 'initial observations')
    initial_stagger = _run_initial_stagger_warmup(
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
        trajectory, rollout_stats = _collect_trajectory(
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
            _block_until_ready((learner_state, last_metrics))
        finally:
          jax.profiler.stop_trace()
      else:
        learner_state, last_metrics = learner.ppo(
            trajectories,
            learner_state,
            step=int(state.get('step', 0)) + update_index,
            recompute_rewards=False,
            profile=args.profile_learner and measuring)
        _block_until_ready((learner_state, last_metrics))
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
      timing_summary = _timing_summary(update_timings)
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


def _build_learner_and_actor(
    *,
    state: dict,
    batch_size: int,
    ppo_batches: int,
    ppo_epochs: int,
    learner_minibatch_size: int,
    learner_minibatch_scan_size: int,
    offload_minibatch_outputs: bool,
    learning_rate: float | None,
    policy_gradient_weight: float | None = None,
    kl_teacher_weight: float | None = None,
    value_cost: float | None = None,
    reward_halflife: float | None = None,
    reward_damage_ratio: float | None = None,
    reward_stalling_penalty: float | None = None,
    reward_stalling_threshold: float | None = None,
    reward_approaching_factor: float | None = None,
    reward_ledge_grab_penalty: float | None = None,
    reward_zelda_penalty: float | None = None,
    ppo_beta: float | None = None,
    ppo_epsilon: float | None = None,
    ppo_max_mean_actor_kl: float | None = None,
    post_update_eval_interval: int | None = None,
    revert_on_post_update_actor_kl: bool = False,
    optimizer_burnin_epochs: int | None = None,
    value_burnin_epochs: int | None = None,
    sample_temperature: float,
    learner_param_dtype: str = 'float32',
):
  policy = tf_checkpoint.load_policy_from_tf_state(
      state, param_dtype=learner_param_dtype)
  teacher = tf_checkpoint.load_policy_from_tf_state(
      state, param_dtype=learner_param_dtype)
  config_dict = tf_checkpoint.jax_config_from_tf_config(copy.deepcopy(state['config']))
  _add_missing_network_embed(config_dict['value_function']['network'])
  train_config = dataclass_from_dict(
      train_lib.Config,
      config_dict,
  )
  value_function = train_lib.value_function_from_config(
      train_config, rngs=nnx.Rngs(1))
  value_state = jax_utils.get_module_state(value_function, to_numpy=False)
  value_state = tf_checkpoint.cast_floating_state(
      value_state, learner_param_dtype)
  jax_utils.set_module_state(value_function, value_state)
  tf_checkpoint.set_compute_dtype(value_function, learner_param_dtype)

  learner_config = dataclass_from_dict(
      learner_lib.LearnerConfig,
      copy.deepcopy(state.get('rl_config', {}).get('learner', {})),
  )
  if ppo_batches > 0:
    learner_config.ppo.num_batches = ppo_batches
  if ppo_epochs > 0:
    learner_config.ppo.num_epochs = ppo_epochs
  if learning_rate is not None:
    learner_config.learning_rate = learning_rate
  if policy_gradient_weight is not None:
    learner_config.policy_gradient_weight = policy_gradient_weight
  if kl_teacher_weight is not None:
    learner_config.kl_teacher_weight = kl_teacher_weight
  if value_cost is not None:
    learner_config.value_cost = value_cost
  if reward_halflife is not None:
    learner_config.reward_halflife = reward_halflife
  if reward_damage_ratio is not None:
    learner_config.reward.damage_ratio = reward_damage_ratio
  if reward_stalling_penalty is not None:
    learner_config.reward.stalling_penalty = reward_stalling_penalty
  if reward_stalling_threshold is not None:
    learner_config.reward.stalling_threshold = reward_stalling_threshold
  if reward_approaching_factor is not None:
    learner_config.reward.approaching_factor = reward_approaching_factor
  if reward_ledge_grab_penalty is not None:
    learner_config.reward.ledge_grab_penalty = reward_ledge_grab_penalty
  if reward_zelda_penalty is not None:
    learner_config.reward.zelda_penalty = reward_zelda_penalty
  if ppo_beta is not None:
    learner_config.ppo.beta = ppo_beta
  if ppo_epsilon is not None:
    learner_config.ppo.epsilon = ppo_epsilon
  if ppo_max_mean_actor_kl is not None:
    learner_config.ppo.max_mean_actor_kl = ppo_max_mean_actor_kl
  if post_update_eval_interval is not None:
    learner_config.ppo.post_update_eval_interval = post_update_eval_interval
  if revert_on_post_update_actor_kl:
    learner_config.ppo.revert_on_post_update_actor_kl = True
  if optimizer_burnin_epochs is not None:
    learner_config.optimizer_burnin_epochs = optimizer_burnin_epochs
  if value_burnin_epochs is not None:
    learner_config.value_burnin_epochs = value_burnin_epochs
  learner_config.ppo.minibatch_size = learner_minibatch_size
  learner_config.ppo.minibatch_scan_size = learner_minibatch_scan_size
  learner_config.ppo.offload_minibatch_outputs = offload_minibatch_outputs

  learner = learner_lib.Learner(
      config=learner_config,
      policy=policy,
      teacher=teacher,
      value_function=value_function,
  )
  if 'state' in state:
    learner.restore_from_imitation(
        state['state'], param_dtype=learner_param_dtype)

  name_code = _name_code(state, batch_size)
  actor = jax_agents.BasicAgent(
      policy=learner.policy,
      batch_size=batch_size,
      name_code=name_code,
      sample_kwargs=dict(temperature=sample_temperature),
      compile=True,
      pack_args=True,
  )
  return learner, actor, np.asarray(name_code, dtype=np.int32)


def _add_missing_network_embed(network_config: dict):
  network_config.setdefault('embed', {
      'name': 'simple',
      'simple': {},
      'enhanced': networks.EnhancedEmbedModule.default_config(),
  })


def _name_code(state: dict, batch_size: int):
  names = eval_lib.get_name_from_rl_state(state)
  if names is None:
    return np.zeros(batch_size, dtype=np.int32)
  return np.asarray(
      [eval_lib.get_name_code(state, names[i % len(names)])
       for i in range(batch_size)],
      dtype=np.int32,
  )


def _initial_stagger_total_steps(workers: int, stagger_steps: int) -> int:
  if stagger_steps <= 0:
    return 0
  return int(workers) * int(stagger_steps)


def _active_workers_for_stagger_step(
    step: int,
    workers: int,
    stagger_steps: int,
) -> int:
  if stagger_steps <= 0:
    return int(workers)
  return min(int(workers), 1 + int(step) // int(stagger_steps))


def _run_initial_stagger_warmup(
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
  total_steps = _initial_stagger_total_steps(workers, stagger_steps)
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
    active = _active_workers_for_stagger_step(step, workers, stagger_steps)
    active_worker_count.value = active
    active_env_steps += active * (total_batch // workers)

    reset_start = time.perf_counter()
    reset_mask = np.asarray(packed.needs_reset, dtype=np.bool_)
    if np.any(reset_mask):
      _handle_reset_delay_queues(
          env_action_queue=env_action_queue,
          learner_action_queue=learner_action_queue,
          dummy_outputs=dummy_outputs,
          reset_mask=reset_mask,
      )
    reset_done = time.perf_counter()

    policy_start = time.perf_counter()
    sample_outputs = actor.step_device(packed.game, packed.needs_reset)
    policy_done = time.perf_counter()
    env_action_queue.append(_to_numpy_tree(sample_outputs.controller_state))
    delayed_controller = env_action_queue.popleft()
    learner_action_queue.append(sample_outputs)
    learner_action_queue.popleft()

    invalid = benchmark_sim_mp._copy_controller(
        action, delayed_controller, controller_spacing)
    action_done = time.perf_counter()
    benchmark_sim_mp._barrier_wait(
        action_barrier, barrier_timeout, 'stagger action release')
    release_done = time.perf_counter()
    benchmark_sim_mp._barrier_wait(
        obs_barrier, barrier_timeout, 'stagger observation wait')
    obs_done = time.perf_counter()

    done, stockout, timeout, max_frame = benchmark_sim_mp._sum_step_counters(
        step_counters, workers)
    worker_step_s, worker_fill_s, _ = (
        benchmark_sim_mp._sum_step_timings(step_timings, workers))
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
      timing_summary=_timing_summary(timings),
  )


def _timing_summary(timings: dict) -> dict:
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


def _collect_trajectory(
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
        actor_state = _to_numpy_tree(packed.game)
        reset_mask = np.asarray(packed.needs_reset, dtype=np.bool_).copy()
        states.append(actor_state)
        resets.append(reset_mask)
        chunk_inputs.append((actor_state, reset_mask))
        chunk_reset_masks.append(reset_mask)
        state_done = time.perf_counter()

        if np.any(reset_mask):
          _handle_reset_delay_queues(
              env_action_queue=env_action_queue,
              learner_action_queue=learner_action_queue,
              dummy_outputs=dummy_outputs,
              reset_mask=reset_mask,
          )

        delayed_entry = env_action_queue.popleft()
        action_entry = learner_action_queue.popleft()
        wait_start = time.perf_counter()
        delayed_controller = _resolve_env_action_entry(
            delayed_entry,
            dummy_outputs=dummy_outputs,
        )
        learner_action = _resolve_learner_action_entry(action_entry)
        wait_done = time.perf_counter()
        actions.append(learner_action)

        action_start = time.perf_counter()
        invalid = benchmark_sim_mp._copy_controller(
            action, delayed_controller, controller_spacing)
        action_done = time.perf_counter()
        benchmark_sim_mp._barrier_wait(
            action_barrier, barrier_timeout, 'action release')
        release_done = time.perf_counter()
        benchmark_sim_mp._barrier_wait(
            obs_barrier, barrier_timeout, 'observation wait')
        obs_done = time.perf_counter()

        transition_state_start = time.perf_counter()
        next_reset_mask = np.asarray(packed.needs_reset, dtype=np.bool_).copy()
        if np.any(next_reset_mask):
          terminal_reward_overrides.append(_TerminalRewardOverride(
              transition_index=len(states) - 1,
              reset_mask=next_reset_mask,
              terminal_game=_masked_numpy_tree(
                  terminal_packed.game,
                  next_reset_mask,
              ),
          ))
        transition_state_done = time.perf_counter()

        done, stockout, timeout, max_frame = benchmark_sim_mp._sum_step_counters(
            step_counters, workers)
        worker_step_s, worker_fill_s, _ = (
            benchmark_sim_mp._sum_step_timings(step_timings, workers))

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

        _replace_env_action_queue_after_chunk(
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
        env_action_queue.append(_PendingEnvAction(pending_policy_future, index))
        learner_action_queue.append(
            _PendingLearnerAction(pending_policy_future, index))
  finally:
    if pending_policy_future is not None:
      pending_policy_future.result()
    if policy_executor is not None:
      policy_executor.shutdown(wait=True)

  final_state_start = time.perf_counter()
  states.append(_to_numpy_tree(packed.game))
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
    terminal_reward_overrides: list['_TerminalRewardOverride'],
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
  rewards = _batched_transition_rewards(
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


def _transition_reward(state, next_state, reward_config: reward_lib.RewardConfig):
  transition = utils.batch_nest_nt([state, _to_numpy_tree(next_state)])
  return reward_lib.compute_rewards(
      transition,
      **dataclasses.asdict(reward_config))[0]


@dataclasses.dataclass(frozen=True)
class _TerminalRewardOverride:
  transition_index: int
  reset_mask: np.ndarray
  terminal_game: object


def _masked_numpy_tree(value, mask: np.ndarray):
  mask = np.asarray(mask, dtype=np.bool_)
  return utils.map_single_structure(lambda x: np.asarray(x)[mask].copy(), value)


def _batched_transition_rewards(
    time_major_states,
    *,
    terminal_reward_overrides: list[_TerminalRewardOverride],
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


def _terminal_corrected_game(*, reset_game, terminal_game, needs_reset):
  needs_reset = np.asarray(needs_reset, dtype=np.bool_)

  def select(reset_leaf, terminal_leaf):
    reset = needs_reset
    reset_leaf = np.asarray(reset_leaf)
    while reset.ndim < reset_leaf.ndim:
      reset = reset[..., None]
    return np.where(reset, np.asarray(terminal_leaf), reset_leaf)

  return utils.map_nt(select, reset_game, terminal_game)


def _to_numpy_tree(value):
  return utils.map_single_structure(lambda x: np.asarray(x).copy(), value)


def _reset_delay_queue_lanes(queue: deque, default, reset_mask: np.ndarray):
  if not queue:
    return
  for index, value in enumerate(queue):
    if isinstance(value, _PendingEnvAction):
      value.add_reset(reset_mask)
    else:
      queue[index] = _reset_tree_lanes(value, default, reset_mask)


def _handle_reset_delay_queues(
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
  _reset_delay_queue_lanes(
      env_action_queue,
      dummy_outputs.controller_state,
      reset_mask,
  )
  _ = learner_action_queue


@dataclasses.dataclass
class _PendingEnvAction:
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
class _PendingLearnerAction:
  future: concurrent.futures.Future
  index: int


def _resolve_env_action_entry(entry, *, dummy_outputs):
  if not isinstance(entry, _PendingEnvAction):
    return entry
  sample_outputs = entry.future.result()[entry.index]
  controller = _to_numpy_tree(sample_outputs.controller_state)
  if entry.reset_mask is not None and np.any(entry.reset_mask):
    controller = _reset_tree_lanes(
        controller,
        dummy_outputs.controller_state,
        entry.reset_mask,
    )
  return controller


def _resolve_learner_action_entry(entry):
  if not isinstance(entry, _PendingLearnerAction):
    return entry
  return entry.future.result()[entry.index]


def _resolve_delay_queues(
    *,
    env_action_queue: deque,
    learner_action_queue: deque,
    dummy_outputs,
) -> None:
  for index, entry in enumerate(env_action_queue):
    env_action_queue[index] = _resolve_env_action_entry(
        entry,
        dummy_outputs=dummy_outputs,
    )
  for index, entry in enumerate(learner_action_queue):
    learner_action_queue[index] = _resolve_learner_action_entry(entry)


def _replace_env_action_queue_after_chunk(
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
      _reset_delay_queue_lanes(
          env_action_queue,
          dummy_outputs.controller_state,
          reset_mask,
      )
    env_action_queue.append(_to_numpy_tree(sample_outputs.controller_state))
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


def _block_until_ready(value):
  for leaf in jax.tree.leaves(value):
    if hasattr(leaf, 'block_until_ready'):
      leaf.block_until_ready()
    elif isinstance(leaf, np.ndarray):
      # NumPy arrays may wrap pending jax.copy_to_host_async results.
      np.asarray(leaf)


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
