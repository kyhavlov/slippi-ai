import argparse
import json
from collections import Counter

import jax
import melee
import numpy as np
import tree

from slippi_ai.sim_env import multiprocess_env
from slippi_ai import data
from slippi_ai import embed as tf_embed
from slippi_ai import eval_lib
from slippi_ai import sim_env
from slippi_ai import utils


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model-path', default='models/imitation_v19.pkl')
  parser.add_argument('--batch-size', type=int, default=8)
  parser.add_argument('--steps', type=int, default=512)
  parser.add_argument('--stage', default='final_destination')
  parser.add_argument('--jax-param-dtype', choices=('float32', 'bfloat16'), default='float32')
  parser.add_argument('--sample-temperature', type=float, default=1.0)
  args = parser.parse_args()

  state = eval_lib.load_state(path=args.model_path)
  spacing = multiprocess_env.default_controller_spacing(state)
  stage = getattr(melee.Stage, args.stage.upper())

  tf_agent = eval_lib.build_delayed_agent(
      state=state,
      batch_size=args.batch_size * 2,
      console_delay=0,
      platform='tf',
      compile=False,
      async_inference=False,
      name='',
      sample_temperature=args.sample_temperature,
  )
  jax_agent = eval_lib.build_delayed_agent(
      state=state,
      batch_size=args.batch_size * 2,
      console_delay=0,
      platform='jax',
      compile=False,
      jax_param_dtype=args.jax_param_dtype,
      name='',
      sample_temperature=args.sample_temperature,
  )

  env = sim_env.SimBatchedEnvironment(
      num_envs=args.batch_size,
      length=max(128, args.steps + 4),
      stage=stage,
      character_pool='fox,falco',
      max_frame_id=28800 - 123,
  )

  logits_max = 0.0
  output_max = 0.0
  hidden_max = 0.0
  per_component_logits = Counter()
  done_count = 0
  invalid_count = 0

  try:
    game_batch = env.current_game_batch(
        needs_reset=np.ones(args.batch_size, dtype=np.bool_))
    prev_controller = tf_agent._agent._prev_controller
    tf_hidden = tf_agent._agent.hidden_state
    jax_hidden = jax_agent._agent.hidden_state()

    for step in range(args.steps):
      game = game_batch.game
      needs_reset = game_batch.needs_reset

      tf_game = tf_agent._policy.embed_game.from_state(game)
      tf_state_action = tf_embed.StateAction(
          state=tf_game,
          action=prev_controller,
          name=tf_agent.name_code,
      )
      tf_embedded = tf_agent._policy.embed_state_action(tf_state_action)
      tf_input = tf_agent._policy._opponent_pooling(tf_embedded)
      tf_output, next_tf_hidden = tf_agent._policy.network.step_with_reset(
          tf_input, needs_reset, tf_hidden)
      tf_dist = tf_agent._policy.controller_head.distance(
          tf_output, prev_controller, prev_controller)
      tf_sample = tf_agent._policy.controller_head.sample(
          tf_output, prev_controller, temperature=args.sample_temperature)

      jax_game = jax_agent._policy.network.encode_game(game)
      jax_state_action = data.StateAction(
          state=jax_game,
          action=prev_controller,
          name=jax_agent.name_code,
      )
      jax_output, next_jax_hidden = jax_agent._policy.network.step_with_reset(
          jax_state_action, needs_reset, jax_hidden)
      jax_dist = jax_agent._policy.controller_head.distance(
          jax_output, prev_controller, prev_controller)

      output_max = max(output_max, _max_abs_tree(tf_output, jax_output))
      hidden_max = max(hidden_max, _max_abs_tree(next_tf_hidden, next_jax_hidden))
      logits_max = max(logits_max, _max_abs_tree(tf_dist.logits, jax_dist.logits))
      _accum_component_max(per_component_logits, tf_dist.logits, jax_dist.logits)

      invalid_count += _invalid_controller_count(tf_sample.controller_state, spacing)
      needs_reset = env.step_encoded(
          tf_sample.controller_state,
          axis_spacing=spacing[0],
          shoulder_spacing=spacing[1],
      )
      done_count += int(needs_reset.sum())
      game_batch = env.current_game_batch(needs_reset=needs_reset)
      prev_controller = utils.map_single_structure(
          lambda x: np.asarray(x), tf_sample.controller_state)
      tf_hidden = next_tf_hidden
      jax_hidden = next_jax_hidden

  finally:
    env.stop()

  result = {
      'model_path': args.model_path,
      'batch_size': args.batch_size,
      'steps': args.steps,
      'jax_param_dtype': args.jax_param_dtype,
      'output_max_abs': output_max,
      'hidden_max_abs': hidden_max,
      'logits_max_abs': logits_max,
      'per_component_logits_max_abs': dict(per_component_logits),
      'done_count': done_count,
      'invalid_controller_values': invalid_count,
  }
  print(json.dumps(result, indent=2, sort_keys=True))


def _max_abs_tree(a, b) -> float:
  out = 0.0
  for x, y in zip(tree.flatten(a), tree.flatten(b)):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
      raise ValueError(f'shape mismatch: {x.shape} != {y.shape}')
    if x.size:
      out = max(out, float(np.max(np.abs(x - y))))
  return out


def _accum_component_max(counter: Counter, a, b):
  def visit(prefix, x, y):
    if isinstance(x, tuple) and hasattr(x, '_fields'):
      for field in x._fields:
        visit(f'{prefix}.{field}' if prefix else field, getattr(x, field), getattr(y, field))
      return
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size:
      counter[prefix] = max(counter[prefix], float(np.max(np.abs(x - y))))

  visit('', a, b)


def _invalid_controller_count(controller, spacing: tuple[int, int]) -> int:
  axis_spacing, shoulder_spacing = spacing
  invalid = 0
  for values, limit in (
      (controller.main_stick.x, axis_spacing),
      (controller.main_stick.y, axis_spacing),
      (controller.c_stick.x, axis_spacing),
      (controller.c_stick.y, axis_spacing),
      (controller.shoulder, shoulder_spacing),
  ):
    values = np.asarray(values)
    invalid += int((values < 0).sum() + (values > limit).sum())
  return invalid


if __name__ == '__main__':
  main()
