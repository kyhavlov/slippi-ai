import argparse
import json
import time

import jax
import numpy as np
import tree

from slippi_ai import eval_lib
from slippi_ai.jax import agents as jax_agents
from slippi_ai.jax import tf_checkpoint


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model-path', default='models/rl_doubles_v27_11000.pkl')
  parser.add_argument('--batch-size', type=int, default=1024)
  parser.add_argument('--steps', type=int, default=2000)
  parser.add_argument('--warmup-steps', type=int, default=50)
  parser.add_argument('--sample-temperature', type=float, default=1.0)
  parser.add_argument(
      '--mode',
      choices=('agent', 'agent-device', 'device-logits'),
      default='agent')
  parser.add_argument('--no-pack-args', action='store_true')
  args = parser.parse_args()

  state = eval_lib.load_state(path=args.model_path)
  policy = tf_checkpoint.load_policy_from_tf_state(state)
  name_code = _name_code(state, args.batch_size)
  agent = jax_agents.BasicAgent(
      policy=policy,
      batch_size=args.batch_size,
      name_code=name_code,
      sample_kwargs=dict(temperature=args.sample_temperature),
      compile=True,
      pack_args=not args.no_pack_args,
  )

  if args.mode == 'agent':
    result = _bench_agent(agent, args.batch_size, args.steps, args.warmup_steps)
  elif args.mode == 'agent-device':
    result = _bench_agent_device(agent, args.batch_size, args.steps, args.warmup_steps)
  else:
    result = _bench_device(policy, name_code, args.batch_size, args.steps, args.warmup_steps)

  result['model_path'] = args.model_path
  result['batch_size'] = args.batch_size
  result['steps'] = args.steps
  result['warmup_steps'] = args.warmup_steps
  result['mode'] = args.mode
  result['pack_args'] = not args.no_pack_args
  print(json.dumps(result, indent=2, sort_keys=True))


def _name_code(state: dict, batch_size: int):
  names = eval_lib.get_name_from_rl_state(state)
  if names is None:
    return 0
  return [eval_lib.get_name_code(state, names[i % len(names)]) for i in range(batch_size)]


def _block(x):
  for leaf in tree.flatten(x):
    np.asarray(leaf)


def _bench_agent(agent, batch_size: int, steps: int, warmup_steps: int):
  game = agent._policy.network.dummy((batch_size,)).state
  needs_reset = np.zeros(batch_size, dtype=np.bool_)
  for _ in range(warmup_steps):
    _block(agent.step_controller_state(game, needs_reset))

  start = time.perf_counter()
  for _ in range(steps):
    _block(agent.step_controller_state(game, needs_reset))
  elapsed = time.perf_counter() - start
  return _rates(elapsed, batch_size, steps)


def _bench_agent_device(agent, batch_size: int, steps: int, warmup_steps: int):
  game = agent._policy.network.encode_game(agent._policy.network.dummy((batch_size,)).state)
  needs_reset = np.zeros(batch_size, dtype=np.bool_)
  prev_controller = agent._prev_controller
  hidden_state = agent.hidden_state()
  sample_fn = (
      agent._jitted_sample_controller_state
      if agent._compile
      else agent._sample_controller_state)

  for _ in range(warmup_steps):
    prev_controller, hidden_state = sample_fn(
        (game, needs_reset), agent._name_code, prev_controller, hidden_state)
  jax.block_until_ready(prev_controller)

  start = time.perf_counter()
  for _ in range(steps):
    prev_controller, hidden_state = sample_fn(
        (game, needs_reset), agent._name_code, prev_controller, hidden_state)
  jax.block_until_ready(prev_controller)
  elapsed = time.perf_counter() - start
  return _rates(elapsed, batch_size, steps)


def _bench_device(
    policy,
    name_code,
    batch_size: int,
    steps: int,
    warmup_steps: int,
):
  game = policy.network.encode_game(policy.network.dummy((batch_size,)).state)
  needs_reset = np.zeros(batch_size, dtype=np.bool_)
  name_code = np.asarray(name_code, dtype=np.int32)
  if name_code.ndim == 0:
    name_code = np.full(batch_size, int(name_code), dtype=np.int32)
  prev_controller = policy.controller_head.dummy_controller([batch_size])
  hidden_state = policy.initial_state(batch_size)

  needs_reset = jax.device_put(needs_reset)
  name_code = jax.device_put(name_code)
  prev_controller = jax.device_put(prev_controller)

  # Keep this explicit instead of using BasicAgent so this mode measures model
  # math and device-resident recurrent/action state, not Python arg packing or
  # CPU action transfer. It computes controller logits via teacher-forced
  # distance instead of sampling, so compare it to sampled agent mode as a lower
  # bound for model math rather than as an end-to-end rollout number.
  from slippi_ai.data import StateAction

  def one_step(prev_action, prev_state):
    state_action = StateAction(state=game, action=prev_action, name=name_code)
    output, next_state = policy.network.step_with_reset(
        state_action,
        needs_reset,
        prev_state,
    )
    distance_outputs = policy.controller_head.distance(output, prev_action, prev_action)
    return prev_action, next_state, distance_outputs.logits

  @jax.jit
  def run_steps(prev_action, prev_state, num_steps: int):
    def body(_, carry):
      prev, state, _ = carry
      return one_step(prev, state)
    logits = policy.controller_head.dummy_sample_outputs([batch_size]).logits
    return jax.lax.fori_loop(0, num_steps, body, (prev_action, prev_state, logits))

  prev_controller, hidden_state, logits = run_steps(
      prev_controller, hidden_state, int(warmup_steps))
  jax.block_until_ready(hidden_state)

  start = time.perf_counter()
  prev_controller, hidden_state, logits = run_steps(
      prev_controller, hidden_state, int(steps))
  jax.block_until_ready(logits)
  elapsed = time.perf_counter() - start
  return _rates(elapsed, batch_size, steps)


def _rates(elapsed: float, batch_size: int, steps: int):
  frames = int(batch_size) * int(steps)
  return {
      'elapsed_sec': elapsed,
      'agent_steps_per_sec': steps / elapsed,
      'player_frames_per_sec': frames / elapsed,
      'ns_per_player_frame': elapsed * 1e9 / frames,
  }


if __name__ == '__main__':
  main()
