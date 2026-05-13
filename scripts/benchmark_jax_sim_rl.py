import argparse
import copy
import json
import multiprocessing as mp
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path

import jax
import numpy as np
from flax import nnx

from scripts import benchmark_sim_mp
from slippi_ai import eval_lib
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
  parser.add_argument(
      '--learner-param-dtype',
      choices=('float32', 'bfloat16'),
      default='float32')
  parser.add_argument('--length', type=int, default=256,
                      help='Sim EnvBatch ring buffer length.')
  parser.add_argument('--max-game-frames', type=int, default=28800)
  parser.add_argument('--sample-temperature', type=float, default=1.0)
  parser.add_argument('--barrier-timeout', type=float, default=900.0)
  parser.add_argument('--print-every', type=int, default=1)
  args = parser.parse_args()

  model_path = Path(args.model_path)
  if not model_path.exists():
    raise FileNotFoundError(model_path)
  if args.workers <= 0 or args.batch_size <= 0:
    raise ValueError('--workers and --batch-size must be positive')
  if args.rollout_length <= 0 or args.ppo_batches < 0 or args.updates <= 0:
    raise ValueError('--rollout-length and --updates must be positive')

  state = eval_lib.load_state(path=str(model_path))
  total_batch = args.workers * args.batch_size
  total_packed = total_batch * 2
  ctx = mp.get_context('spawn')

  obs_owner = benchmark_sim_mp.SharedArrayOwner()
  packed = sim_env.make_packed_game_builder(total_batch, array_factory=obs_owner.array)
  action_owner = benchmark_sim_mp.SharedArrayOwner()
  action = benchmark_sim_mp._shared_encoded_controller(total_packed, action_owner.array)
  spacing = benchmark_sim_mp._default_controller_spacing(state)

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

    learner, actor, name_code = _build_learner_and_actor(
        state=state,
        batch_size=total_packed,
        ppo_batches=args.ppo_batches,
        ppo_epochs=args.ppo_epochs,
        learner_minibatch_size=args.learner_minibatch_size,
        learner_minibatch_scan_size=args.learner_minibatch_scan_size,
        offload_minibatch_outputs=args.offload_minibatch_outputs,
        learning_rate=args.learning_rate,
        sample_temperature=args.sample_temperature,
        learner_param_dtype=args.learner_param_dtype,
    )
    ppo_batches = learner._config.ppo.num_batches
    if args.rollout_length <= actor._policy.delay:
      raise ValueError(
          f'--rollout-length must be greater than policy delay '
          f'{actor._policy.delay}, got {args.rollout_length}')
    learner_state = learner.initial_state(total_packed)
    action_queue = deque(
        [_to_numpy_tree(actor._policy.controller_head.dummy_sample_outputs([total_packed]))
         for _ in range(actor._policy.delay + 1)])

    benchmark_sim_mp._barrier_wait(
        obs_barrier, args.barrier_timeout, 'initial observations')

    timings = defaultdict(float)
    counters = defaultdict(int)
    measured_updates = 0
    measured_rollout_steps = 0
    total_updates = args.warmup_updates + args.updates
    total_start = time.perf_counter()
    measured_start = None
    last_metrics = None

    for update_index in range(total_updates):
      measuring = update_index >= args.warmup_updates
      if measuring and measured_start is None:
        measured_start = time.perf_counter()

      trajectories = []
      for batch_index in range(ppo_batches):
        rollout_start = time.perf_counter()
        trajectory, rollout_stats = _collect_trajectory(
            actor=actor,
            packed=packed,
            action=action,
            action_queue=action_queue,
            action_barrier=action_barrier,
            obs_barrier=obs_barrier,
            step_counters=step_counters,
            workers=args.workers,
            total_batch=total_batch,
            rollout_length=args.rollout_length,
            controller_spacing=spacing,
            name_code=name_code,
            barrier_timeout=args.barrier_timeout,
        )
        rollout_done = time.perf_counter()
        trajectories.append(trajectory)
        if measuring:
          for key, value in rollout_stats['timings_sec'].items():
            timings[key] += value
          for key, value in rollout_stats['counters'].items():
            counters[key] += value
          timings['trajectory_collect_total_s'] += rollout_done - rollout_start
          measured_rollout_steps += args.rollout_length

      learner_start = time.perf_counter()
      learner_state, last_metrics = learner.ppo(
          trajectories,
          learner_state,
          step=int(state.get('step', 0)) + update_index,
          profile=args.profile_learner and measuring)
      _block_until_ready((learner_state, last_metrics))
      learner_done = time.perf_counter()

      if measuring:
        timings['learner_ppo_s'] += learner_done - learner_start
        measured_updates += 1

      if args.print_every and (update_index + 1) % args.print_every == 0:
        phase = 'measure' if measuring else 'warmup'
        elapsed = time.perf_counter() - total_start
        print(
            f'update={update_index + 1}/{total_updates} phase={phase} '
            f'env_steps_per_sec={total_batch * args.rollout_length * ppo_batches * (update_index + 1) / max(elapsed, 1e-9):.1f}',
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
            'ppo_batches': ppo_batches,
            'ppo_epochs': learner._config.ppo.num_epochs,
            'learner_minibatch_size': args.learner_minibatch_size,
            'learner_minibatch_scan_size': args.learner_minibatch_scan_size,
            'offload_minibatch_outputs': args.offload_minibatch_outputs,
            'profile_learner': args.profile_learner,
            'updates': args.updates,
            'warmup_updates': args.warmup_updates,
            'policy_delay': actor._policy.delay,
            'learning_rate': learner._config.learning_rate,
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
    action_owner.close()
    obs_owner.unlink()
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


def _collect_trajectory(
    *,
    actor: jax_agents.BasicAgent,
    packed,
    action,
    action_queue: deque,
    action_barrier,
    obs_barrier,
    step_counters,
    workers: int,
    total_batch: int,
    rollout_length: int,
    controller_spacing: tuple[int, int],
    name_code: np.ndarray,
    barrier_timeout: float,
) -> tuple[Trajectory, dict]:
  states = []
  actions = []
  resets = []
  timings = defaultdict(float)
  counters = defaultdict(int)
  initial_state = actor.hidden_state()

  for _ in range(rollout_length):
    state_start = time.perf_counter()
    states.append(_to_numpy_tree(packed.game))
    resets.append(np.asarray(packed.needs_reset, dtype=np.bool_).copy())
    state_done = time.perf_counter()

    policy_start = time.perf_counter()
    sample_outputs = _to_numpy_tree(actor.step(packed.game, packed.needs_reset))
    policy_done = time.perf_counter()
    action_queue.append(sample_outputs)
    delayed_output = action_queue.popleft()
    actions.append(delayed_output)

    invalid = benchmark_sim_mp._copy_controller(
        action, delayed_output.controller_state, controller_spacing)
    action_done = time.perf_counter()
    benchmark_sim_mp._barrier_wait(
        action_barrier, barrier_timeout, 'action release')
    release_done = time.perf_counter()
    benchmark_sim_mp._barrier_wait(
        obs_barrier, barrier_timeout, 'observation wait')
    obs_done = time.perf_counter()

    done, stockout, timeout, max_frame = benchmark_sim_mp._sum_step_counters(
        step_counters, workers)

    timings['state_copy_s'] += state_done - state_start
    timings['policy_sample_s'] += policy_done - policy_start
    timings['action_copy_s'] += action_done - policy_done
    timings['action_release_s'] += release_done - action_done
    timings['obs_wait_s'] += obs_done - release_done
    counters['invalid_actions'] += invalid
    counters['done'] += done
    counters['stockout'] += stockout
    counters['timeout'] += timeout
    counters['max_frame_reached'] += max_frame

  final_state_start = time.perf_counter()
  states.append(_to_numpy_tree(packed.game))
  resets.append(np.asarray(packed.needs_reset, dtype=np.bool_).copy())
  actions.append(action_queue[0])
  trajectory = _build_trajectory(
      actor=actor,
      states=states,
      actions=actions,
      resets=resets,
      initial_state=initial_state,
      delayed_actions=list(action_queue)[1:],
      name_code=name_code,
      rollout_length=rollout_length,
      total_batch=total_batch,
  )
  final_state_done = time.perf_counter()
  timings['trajectory_build_s'] += final_state_done - final_state_start
  return trajectory, {
      'timings_sec': dict(timings),
      'counters': dict(counters),
  }


def _build_trajectory(
    *,
    actor: jax_agents.BasicAgent,
    states: list,
    actions: list,
    resets: list[np.ndarray],
    initial_state,
    delayed_actions: list,
    name_code: np.ndarray,
    rollout_length: int,
    total_batch: int,
) -> Trajectory:
  time_major_states = utils.batch_nest_nt(states)
  encoded_states = actor._policy.network.encode_game(time_major_states)
  return Trajectory(
      states=encoded_states,
      name=np.broadcast_to(
          np.asarray(name_code, dtype=np.int32),
          [rollout_length + 1, total_batch * 2],
      ).copy(),
      actions=utils.batch_nest_nt(actions),
      rewards=np.zeros((rollout_length, total_batch * 2), dtype=np.float32),
      is_resetting=np.stack(resets, axis=0),
      initial_state=initial_state,
      delayed_actions=delayed_actions,
  )


def _to_numpy_tree(value):
  return utils.map_single_structure(lambda x: np.asarray(x).copy(), value)


def _block_until_ready(value):
  for leaf in jax.tree.leaves(value):
    if hasattr(leaf, 'block_until_ready'):
      leaf.block_until_ready()
    elif isinstance(leaf, np.ndarray):
      # NumPy arrays may wrap pending jax.copy_to_host_async results.
      np.asarray(leaf)


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
