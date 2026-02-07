import unittest

import numpy as np
import tensorflow as tf

from slippi_ai import embed
from slippi_ai import opponent_pooling


def _get_p2_p3_slices(embed_game: embed.StructEmbedding) -> tuple[slice, slice]:
  offset = 0
  p2_slice = None
  p3_slice = None
  for field, op in embed_game.embedding:
    size = int(op.size)
    if field == "p2":
      p2_slice = slice(offset, offset + size)
    elif field == "p3":
      p3_slice = slice(offset, offset + size)
    offset += size
  if p2_slice is None or p3_slice is None:
    raise ValueError("Could not locate p2/p3 slices")
  return p2_slice, p3_slice


def _swap_p2_p3_flat(
    x: tf.Tensor,
    *,
    state_size: int,
    p2_slice: slice,
    p3_slice: slice,
) -> tf.Tensor:
  state = x[..., :state_size]
  tail = x[..., state_size:]
  between = state[..., p2_slice.stop:p3_slice.start]
  swapped_state = tf.concat(
      [
          state[..., :p2_slice.start],
          state[..., p3_slice],
          between,
          state[..., p2_slice],
          state[..., p3_slice.stop:],
      ],
      axis=-1,
  )
  return tf.concat([swapped_state, tail], axis=-1)


class OpponentPoolingTest(unittest.TestCase):

  def test_disabled_is_identity(self):
    embed_game = embed.make_game_embedding()
    embed_state_action = embed.get_state_action_embedding(
        embed_game=embed_game,
        embed_action=embed.get_controller_embedding(),
        num_names=16,
    )
    x = tf.random.normal([2, 3, int(embed_state_action.size)])

    pool = opponent_pooling.OpponentPoolingPreprocessor(
        embed_game=embed_game,
        config=opponent_pooling.OpponentPoolingConfig(enabled=False),
    )
    y = pool(x)
    np.testing.assert_allclose(y.numpy(), x.numpy())
    self.assertEqual(len(pool.trainable_variables), 0)

  def test_invariant_to_swap(self):
    tf.random.set_seed(0)
    embed_game = embed.make_game_embedding()
    embed_state_action = embed.get_state_action_embedding(
        embed_game=embed_game,
        embed_action=embed.get_controller_embedding(),
        num_names=16,
    )
    x = tf.random.normal([2, 3, int(embed_state_action.size)])

    p2_slice, p3_slice = _get_p2_p3_slices(embed_game)
    x_swapped = _swap_p2_p3_flat(
        x,
        state_size=int(embed_game.size),
        p2_slice=p2_slice,
        p3_slice=p3_slice,
    )

    pool = opponent_pooling.OpponentPoolingPreprocessor(
        embed_game=embed_game,
        config=opponent_pooling.OpponentPoolingConfig(
            enabled=True,
            k=32,
            include_other=True,
        ),
    )
    y = pool(x)
    y_swapped = pool(x_swapped)

    np.testing.assert_allclose(y.numpy(), y_swapped.numpy(), atol=1e-6, rtol=1e-6)
    self.assertGreater(len(pool.trainable_variables), 0)

  def test_symmetrized_identity_and_swap_involution(self):
    tf.random.set_seed(0)
    embed_game = embed.make_game_embedding()
    embed_state_action = embed.get_state_action_embedding(
        embed_game=embed_game,
        embed_action=embed.get_controller_embedding(),
        num_names=16,
    )
    x = tf.random.normal([2, 3, int(embed_state_action.size)])

    p2_slice, p3_slice = _get_p2_p3_slices(embed_game)
    x_swapped = _swap_p2_p3_flat(
        x,
        state_size=int(embed_game.size),
        p2_slice=p2_slice,
        p3_slice=p3_slice,
    )

    pool = opponent_pooling.OpponentPoolingPreprocessor(
        embed_game=embed_game,
        config=opponent_pooling.OpponentPoolingConfig(
            enabled=True,
            mode="symmetrized",
            k=32,
        ),
    )
    y = pool(x)  # identity in symmetrized mode
    np.testing.assert_allclose(y.numpy(), x.numpy(), atol=1e-6, rtol=1e-6)

    swapped = pool.swap_opponents(x)
    double_swapped = pool.swap_opponents(swapped)
    np.testing.assert_allclose(swapped.numpy(), x_swapped.numpy(), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(double_swapped.numpy(), x.numpy(), atol=1e-6, rtol=1e-6)

    self.assertEqual(y.shape[-1], x.shape[-1])

  def test_symmetrized_average_is_invariant_for_nonlinear_fn(self):
    tf.random.set_seed(0)
    embed_game = embed.make_game_embedding()
    embed_state_action = embed.get_state_action_embedding(
        embed_game=embed_game,
        embed_action=embed.get_controller_embedding(),
        num_names=16,
    )
    x = tf.random.normal([2, 3, int(embed_state_action.size)])

    pool = opponent_pooling.OpponentPoolingPreprocessor(
        embed_game=embed_game,
        config=opponent_pooling.OpponentPoolingConfig(
            enabled=True,
            mode="symmetrized",
            k=32,
        ),
    )

    def nonlinear(z):
      return tf.square(z) + 0.1 * tf.tanh(z)

    swapped = pool.swap_opponents(x)
    y = 0.5 * (nonlinear(pool(x)) + nonlinear(swapped))
    y_swapped = 0.5 * (nonlinear(pool(swapped)) + nonlinear(pool.swap_opponents(swapped)))

    np.testing.assert_allclose(y.numpy(), y_swapped.numpy(), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
  unittest.main(failfast=True)
