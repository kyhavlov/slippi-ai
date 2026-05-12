import os
from pathlib import Path
import unittest

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import jax
import numpy as np
import tree

from slippi_ai import data
from slippi_ai import embed as tf_embed
from slippi_ai import eval_lib
from slippi_ai.jax import embed as jax_embed


MODEL_PATH = Path('models/rl_doubles_v27_11000.pkl')


def _max_abs_tree(a, b) -> float:
  return max(
      float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
      for x, y in zip(tree.flatten(a), tree.flatten(b))
  )


class JaxTfCheckpointTest(unittest.TestCase):

  def test_terminal_sentinel_categories_embed_as_empty(self):
    cases = [
        (jax_embed.embed_char, np.array([255], dtype=np.uint8)),
        (jax_embed.embed_stage, np.array([255], dtype=np.uint8)),
        (jax_embed.embed_item_type, np.array([999], dtype=np.int32)),
        (jax_embed.embed_item_state, np.array([255], dtype=np.uint8)),
    ]

    for embedding, value in cases:
      with self.subTest(embedding=embedding.name):
        encoded = embedding.from_state(value)
        one_hot = np.asarray(embedding(encoded))
        self.assertEqual(one_hot.shape, (1, embedding.size))
        self.assertFalse(np.any(one_hot))

  @unittest.skipUnless(
      MODEL_PATH.exists(),
      f'{MODEL_PATH} is a local model fixture and is not present')
  def test_tf_checkpoint_conversion_matches_controller_distribution(self):
    batch_size = 64
    state = eval_lib.load_state(path=str(MODEL_PATH))
    tf_agent = eval_lib.build_delayed_agent(
        state=state,
        batch_size=batch_size,
        console_delay=0,
        batch_steps=1,
        platform='tf',
        compile=False,
        async_inference=False,
    )
    jax_agent = eval_lib.build_delayed_agent(
        state=state,
        batch_size=batch_size,
        console_delay=0,
        batch_steps=1,
        platform='jax',
        compile=False,
    )

    game = tf_agent._policy.embed_game.dummy([batch_size])
    needs_reset = np.ones(batch_size, dtype=np.bool_)

    tf_prev = tf_agent._agent._prev_controller
    tf_state_action = tf_embed.StateAction(
        state=game,
        action=tf_prev,
        name=tf_agent.name_code,
    )
    tf_embedded = tf_agent._policy.embed_state_action(tf_state_action)
    tf_input = tf_agent._policy._opponent_pooling(tf_embedded)
    tf_output, _ = tf_agent._policy.network.step_with_reset(
        tf_input,
        needs_reset,
        tf_agent._agent.hidden_state,
    )
    tf_dist = tf_agent._policy.controller_head.distance(
        tf_output,
        tf_prev,
        tf_prev,
    )

    jax_prev = jax_agent._agent._prev_controller
    jax_game = jax_agent._policy.network.encode_game(game)
    jax_state_action = data.StateAction(
        state=jax_game,
        action=jax_prev,
        name=jax_agent.name_code,
    )
    jax_output, _ = jax_agent._policy.network.step_with_reset(
        jax_state_action,
        needs_reset,
        jax_agent._agent.hidden_state(),
    )
    jax_dist = jax_agent._policy.controller_head.distance(
        jax_output,
        jax_prev,
        jax_prev,
    )

    kl_tree = jax_agent.embed_controller.map(
        lambda embedding, p, q: embedding.kl_divergence(
            np.asarray(p),
            np.asarray(q),
        ),
        tf_dist.logits,
        jax_dist.logits,
    )
    kl_by_component = [
        np.asarray(x)
        for x in tree.flatten(jax.device_get(kl_tree))
    ]
    kl_sum_by_sample = np.sum(np.stack(kl_by_component, axis=0), axis=0)

    self.assertLess(_max_abs_tree(tf_output, jax_output), 1e-4)
    self.assertLess(_max_abs_tree(tf_dist.logits, jax_dist.logits), 1e-3)
    self.assertLess(float(np.max(kl_sum_by_sample)), 1e-9)


if __name__ == '__main__':
  unittest.main(failfast=True)
