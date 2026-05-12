import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx

from slippi_ai.jax import embed, jax_utils

Array = jax.Array


@dataclasses.dataclass
class OpponentPoolingConfig:
  enabled: bool = False
  mode: str = "summary"
  k: int = 128
  include_other: bool = True


class OpponentPoolingPreprocessor(nnx.Module):
  """Replace embedded p2/p3 blocks with permutation-invariant features."""

  def __init__(
      self,
      *,
      rngs: nnx.Rngs,
      embed_game: embed.StructEmbedding,
      config: OpponentPoolingConfig,
  ):
    self._embed_game = embed_game
    self._config = config
    self._mode = str(config.mode).lower()
    if self._mode not in ("summary", "symmetrized", "process_set"):
      raise ValueError(
          f"Unsupported opponent pooling mode: {config.mode}. "
          "Expected one of ['summary', 'symmetrized', 'process_set'].")

    self._p0_slice = None
    self._p1_slice = None
    self._p2_slice = None
    self._p3_slice = None
    self._output_size = int(embed_game.size)

    if config.enabled:
      if self._mode in ("summary", "process_set") and config.k <= 0:
        raise ValueError(f"OpponentPoolingConfig.k must be > 0, got {config.k}")
      self._p0_slice, self._p1_slice, self._p2_slice, self._p3_slice = (
          self._player_slices())

      if self._mode == "summary":
        self._opp_enc = jax_utils.MLP(
            rngs=rngs,
            input_size=self._p2_slice.stop - self._p2_slice.start,
            features=[config.k],
            activation=nnx.relu,
            activate_final=True,
        )
        query_input_size = (
            self._p1_slice.stop
            + (int(embed_game.size) - self._p3_slice.stop))
        self._query = jax_utils.MLP(
            rngs=rngs,
            input_size=query_input_size,
            features=[config.k],
            activation=nnx.relu,
            activate_final=True,
        )
        pooled_size = config.k * (3 if config.include_other else 2)
        self._output_size = (
            (self._p2_slice.start - 0)
            + (int(embed_game.size) - self._p3_slice.stop)
            + pooled_size)
      elif self._mode == "process_set":
        self._opp_enc = jax_utils.MLP(
            rngs=rngs,
            input_size=self._p2_slice.stop - self._p2_slice.start,
            features=[config.k],
            activation=nnx.relu,
            activate_final=True,
        )
        self._process_set_context = jax_utils.MLP(
            rngs=rngs,
            input_size=config.k * 3,
            features=[config.k],
            activation=nnx.relu,
            activate_final=True,
        )
        pooled_size = config.k * (3 if config.include_other else 2)
        self._output_size = (
            self._p2_slice.start
            + (int(embed_game.size) - self._p3_slice.stop)
            + pooled_size)

  @property
  def output_size(self) -> int:
    return self._output_size

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

  def is_symmetrized(self) -> bool:
    return self._config.enabled and self._mode == "symmetrized"

  def swap_opponents(self, inputs: Array) -> Array:
    if not self._config.enabled:
      return inputs

    assert self._p2_slice is not None
    assert self._p3_slice is not None

    state_size = int(self._embed_game.size)
    state = inputs[..., :state_size]
    tail = inputs[..., state_size:]

    between = state[..., self._p2_slice.stop:self._p3_slice.start]
    swapped_state = jnp.concatenate([
        state[..., :self._p2_slice.start],
        state[..., self._p3_slice],
        between,
        state[..., self._p2_slice],
        state[..., self._p3_slice.stop:],
    ], axis=-1)
    return jnp.concatenate([swapped_state, tail], axis=-1)

  def __call__(self, inputs: Array) -> Array:
    if not self._config.enabled or self._mode == "symmetrized":
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
    globals_ = state[..., self._p3_slice.stop:]

    h2 = self._opp_enc(p2)
    h3 = self._opp_enc(p3)

    if self._mode == "summary":
      q_in = jnp.concatenate([p0, p1, globals_], axis=-1)
      q = self._query(q_in)

      scale = jnp.asarray(1.0 / jnp.sqrt(float(self._config.k)), q.dtype)
      score2 = jnp.sum(q * h2, axis=-1) * scale
      score3 = jnp.sum(q * h3, axis=-1) * scale
      scores = jnp.stack([score2, score3], axis=-1)
      alpha = jax.nn.softmax(scores, axis=-1)
      a2 = alpha[..., 0:1]
      a3 = alpha[..., 1:2]

      h_attn = a2 * h2 + a3 * h3
      h_sum = h2 + h3
      parts = [p0, p1, globals_, h_sum, h_attn]
      if self._config.include_other:
        parts.append(h_sum - h_attn)
    else:
      stacked = jnp.stack([h2, h3], axis=-2)
      pooled = jnp.concatenate(
          [jnp.mean(stacked, axis=-2), jnp.max(stacked, axis=-2)],
          axis=-1)
      c2 = self._process_set_context(jnp.concatenate([h2, pooled], axis=-1))
      c3 = self._process_set_context(jnp.concatenate([h3, pooled], axis=-1))
      parts = [p0, p1, globals_, c2 + c3, c2 * c3]
      if self._config.include_other:
        parts.append(jnp.maximum(c2, c3))

    new_state = jnp.concatenate(parts, axis=-1)
    return jnp.concatenate([new_state, tail], axis=-1)
