import dataclasses

import sonnet as snt
import tensorflow as tf

from slippi_ai import embed


@dataclasses.dataclass
class OpponentPoolingConfig:
  """Config for permutation-invariant opponent pooling.

  If enabled, the model replaces raw p2/p3 embedded blocks with a pooled
  representation that is invariant to swapping the two opponents.
  """

  enabled: bool = False
  k: int = 128
  include_other: bool = True


@snt.allow_empty_variables
class OpponentPoolingPreprocessor(snt.Module):
  """Replace raw p2/p3 blocks with permutation-invariant pooled features.

  Operates on the *embedded* flat input (output of embed_state_action), by
  slicing the embedded game-state portion into p0/p1/p2/p3/globals.
  """

  def __init__(
      self,
      *,
      embed_game: embed.StructEmbedding,
      config: OpponentPoolingConfig,
      name: str = "OpponentPoolingPreprocessor",
  ):
    super().__init__(name=name)
    self._embed_game = embed_game
    self._config = config

    if config.enabled:
      if config.k <= 0:
        raise ValueError(f"OpponentPoolingConfig.k must be > 0, got {config.k}")
      self._p0_slice, self._p1_slice, self._p2_slice, self._p3_slice = self._player_slices()
    else:
      self._p0_slice = None
      self._p1_slice = None
      self._p2_slice = None
      self._p3_slice = None

    # Shared opponent encoder f.
    self._opp_enc = snt.nets.MLP([config.k], activate_final=True, name="opp_enc")
    # Query network g.
    self._query = snt.nets.MLP([config.k], activate_final=True, name="query")

  def _player_slices(self) -> tuple[slice, slice, slice, slice]:
    offset = 0
    slices: dict[str, slice] = {}
    for field, op in self._embed_game.embedding:
      size = int(op.size)
      if field in ("p0", "p1", "p2", "p3"):
        slices[field] = slice(offset, offset + size)
      offset += size
    missing = [k for k in ("p0", "p1", "p2", "p3") if k not in slices]
    if missing:
      raise ValueError(f"embed_game is missing fields: {missing}")
    return slices["p0"], slices["p1"], slices["p2"], slices["p3"]

  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    if not self._config.enabled:
      return inputs

    assert self._p0_slice is not None
    assert self._p1_slice is not None
    assert self._p2_slice is not None
    assert self._p3_slice is not None

    state_size = int(self._embed_game.size)
    state = inputs[..., :state_size]
    tail = inputs[..., state_size:]

    p0 = state[..., self._p0_slice]
    p1 = state[..., self._p1_slice]
    p2 = state[..., self._p2_slice]
    p3 = state[..., self._p3_slice]

    # Everything after p3 in the embedded game struct (stage/items/etc).
    globals_ = state[..., self._p3_slice.stop:]

    h2 = self._opp_enc(p2)
    h3 = self._opp_enc(p3)

    q_in = tf.concat([p0, p1, globals_], axis=-1)
    q = self._query(q_in)

    scale = tf.cast(tf.math.rsqrt(tf.cast(self._config.k, tf.float32)), q.dtype)
    score2 = tf.reduce_sum(q * h2, axis=-1) * scale
    score3 = tf.reduce_sum(q * h3, axis=-1) * scale
    scores = tf.stack([score2, score3], axis=-1)
    alpha = tf.nn.softmax(scores, axis=-1)
    a2 = alpha[..., 0:1]
    a3 = alpha[..., 1:2]

    h_attn = a2 * h2 + a3 * h3
    h_sum = h2 + h3
    parts = [p0, p1, globals_, h_sum, h_attn]
    if self._config.include_other:
      h_other = h_sum - h_attn
      parts.append(h_other)

    new_state = tf.concat(parts, axis=-1)
    return tf.concat([new_state, tail], axis=-1)
