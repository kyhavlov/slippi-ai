"""JAX RL learner implementing PPO."""

import dataclasses
import functools
import time
import typing as tp
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from slippi_ai import data, reward as reward_lib, utils
from slippi_ai.evaluators import Trajectory
from slippi_ai.jax import jax_utils, embed, rl_lib, tf_checkpoint
from slippi_ai.jax import value_function as vf_lib
from slippi_ai.jax.policies import Policy, LogitUnrollOutputs
from slippi_ai.jax.networks import RecurrentState

Array = jax.Array
field = lambda f: dataclasses.field(default_factory=f)


@dataclasses.dataclass
class PPOConfig:
  num_epochs: int = 1
  num_batches: int = 1
  minibatch_size: int = 0
  minibatch_scan_size: int = 4
  offload_minibatch_outputs: bool = False
  epsilon: float = 1e-2
  beta: float = 0
  max_mean_actor_kl: float = 1e-4
  post_update_eval_interval: int = 0
  revert_on_post_update_actor_kl: bool = False


@dataclasses.dataclass
class LearnerConfig:
  learning_rate: float = 1e-4
  policy_gradient_weight: float = 1
  kl_teacher_weight: float = 1e-1
  reverse_kl_teacher_weight: float = 0
  entropy_weight: float = 0
  value_cost: float = 0.5
  reward_halflife: float = 4  # measured in seconds
  reward: reward_lib.RewardConfig = field(reward_lib.RewardConfig)
  ppo: PPOConfig = field(PPOConfig)

  optimizer_burnin_epochs: int = 1
  value_burnin_epochs: int = 1

class LearnerState(tp.NamedTuple):
  teacher: RecurrentState
  value_function: RecurrentState


class LearnerOutputs(tp.NamedTuple):
  teacher: LogitUnrollOutputs
  value: vf_lib.ValueOutputs


def get_frames(trajectory: Trajectory) -> data.Frames:
  """Gives time-major frames with actions taken."""
  state_action = data.StateAction(
      state=trajectory.states,
      action=trajectory.actions.controller_state,
      name=trajectory.name,
  )
  return data.Frames(state_action, trajectory.is_resetting, trajectory.rewards)


def get_delayed_frames(trajectory: Trajectory) -> data.Frames:
  """Gives time-major frames with delayed actions, for teacher/policy unroll."""
  delay = len(trajectory.delayed_actions)

  if delay == 0:
    return get_frames(trajectory)

  # Extract controller states from delayed actions, each is [B, ...]
  delayed_cs = [sa.controller_state for sa in trajectory.delayed_actions]

  # Add time dimension: [B, ...] -> [1, B, ...]
  delayed_cs_with_time = [
      jax.tree.map(lambda t: t[np.newaxis], cs) for cs in delayed_cs
  ]

  # Concatenate: [T+1, B, ...] + D * [1, B, ...] -> [T+1+D, B, ...]
  # Then take [delay:] to align -> [T+1, B, ...]
  actions = jax.tree.map(
      lambda *ts: jnp.concatenate(ts, axis=0),
      trajectory.actions.controller_state,
      *delayed_cs_with_time,
  )
  actions = jax.tree.map(lambda t: t[delay:], actions)

  state_action = data.StateAction(
      state=trajectory.states,
      action=actions,
      name=trajectory.name,
  )
  return data.Frames(state_action, trajectory.is_resetting, trajectory.rewards)


def reset_frame_actions(
    frames: data.Frames,
    dummy_action,
) -> data.Frames:
  """Reset previous-action inputs on reset frames.

  `StateAction.action` is the controller from the previous frame. At runtime
  BasicAgent replaces that previous action with the dummy controller whenever a
  lane resets before sampling. Learner unrolls must do the same for network
  inputs without rewriting `Trajectory.actions`, which also stores old-policy
  logits/actions used by PPO.
  """
  reset = jnp.asarray(frames.is_resetting, dtype=jnp.bool_)

  def select(action_leaf, dummy_leaf):
    action_leaf = jnp.asarray(action_leaf)
    lane_reset = reset
    while lane_reset.ndim < action_leaf.ndim:
      lane_reset = lane_reset[..., None]
    return jnp.where(lane_reset, jnp.asarray(dummy_leaf), action_leaf)

  action = jax.tree.map(select, frames.state_action.action, dummy_action)
  state_action = data.StateAction(
      state=frames.state_action.state,
      action=action,
      name=frames.state_action.name,
  )
  return data.Frames(state_action, frames.is_resetting, frames.reward)


def update_rewards(
    trajectory: Trajectory,
    reward_config: reward_lib.RewardConfig,
) -> Trajectory:
  rewards = reward_lib.compute_rewards(
      trajectory.states, **dataclasses.asdict(reward_config))
  return trajectory._replace(rewards=rewards)


def trajectory_batch_size(trajectory: Trajectory) -> int:
  return int(trajectory.name.shape[1])


def minibatch_ranges(batch_size: int, minibatch_size: int):
  if minibatch_size <= 0 or minibatch_size >= batch_size:
    yield 0, batch_size
    return
  for start in range(0, batch_size, minibatch_size):
    yield start, min(start + minibatch_size, batch_size)


def slice_time_batch(tree, start: int, end: int):
  return jax.tree.map(lambda t: t[:, start:end], tree)


def slice_batch(tree, start: int, end: int):
  return jax.tree.map(lambda t: t[start:end], tree)


def slice_trajectory(
    trajectory: Trajectory,
    start: int,
    end: int,
) -> Trajectory:
  return Trajectory(
      states=slice_time_batch(trajectory.states, start, end),
      name=trajectory.name[:, start:end],
      actions=slice_time_batch(trajectory.actions, start, end),
      rewards=trajectory.rewards[:, start:end],
      is_resetting=trajectory.is_resetting[:, start:end],
      initial_state=slice_batch(trajectory.initial_state, start, end),
      delayed_actions=[
          slice_batch(action, start, end)
          for action in trajectory.delayed_actions
      ],
  )


def slice_learner_state(
    state: LearnerState,
    start: int,
    end: int,
) -> LearnerState:
  return LearnerState(
      teacher=slice_batch(state.teacher, start, end),
      value_function=slice_batch(state.value_function, start, end),
  )


def concat_learner_states(states: list[LearnerState]) -> LearnerState:
  return LearnerState(
      teacher=jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0),
                           *[state.teacher for state in states]),
      value_function=jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0),
                                  *[state.value_function for state in states]),
  )


def stack_trees(trees: list[tp.Any]):
  return jax.tree.map(lambda *xs: jnp.stack(xs), *trees)


def concat_trees(trees: list[tp.Any], axis: int = 0):
  return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=axis), *trees)


def split_trajectory_minibatches(
    trajectory: Trajectory,
    minibatch_size: int,
) -> Trajectory | None:
  """Reshape a trajectory batch into contiguous equal-size minibatches."""
  batch_size = trajectory_batch_size(trajectory)
  if minibatch_size <= 0 or batch_size % minibatch_size:
    return None
  num_minibatches = batch_size // minibatch_size

  def split_time_batch(t):
    t = jnp.asarray(t)
    shape = (t.shape[0], num_minibatches, minibatch_size) + t.shape[2:]
    return jnp.swapaxes(jnp.reshape(t, shape), 0, 1)

  def split_batch(t):
    t = jnp.asarray(t)
    shape = (num_minibatches, minibatch_size) + t.shape[1:]
    return jnp.reshape(t, shape)

  return Trajectory(
      states=jax.tree.map(split_time_batch, trajectory.states),
      name=split_time_batch(trajectory.name),
      actions=jax.tree.map(split_time_batch, trajectory.actions),
      rewards=split_time_batch(trajectory.rewards),
      is_resetting=split_time_batch(trajectory.is_resetting),
      initial_state=jax.tree.map(split_batch, trajectory.initial_state),
      delayed_actions=[
          jax.tree.map(split_batch, action)
          for action in trajectory.delayed_actions
      ],
  )


def split_learner_state_minibatches(
    state: LearnerState,
    minibatch_size: int,
) -> LearnerState | None:
  leaves = jax.tree.leaves(state)
  if not leaves:
    return state
  batch_size = int(leaves[0].shape[0])
  if minibatch_size <= 0 or batch_size % minibatch_size:
    return None
  num_minibatches = batch_size // minibatch_size

  def split_batch(t):
    t = jnp.asarray(t)
    shape = (num_minibatches, minibatch_size) + t.shape[1:]
    return jnp.reshape(t, shape)

  return jax.tree.map(split_batch, state)


def merge_learner_state_minibatches(state: LearnerState) -> LearnerState:
  return jax.tree.map(
      lambda t: (
          jnp.reshape(t, (t.shape[0] * t.shape[1],) + t.shape[2:])
          if t.ndim >= 2 else t),
      state)


def group_record_axis(tree, scan_size: int):
  """Reshape leading record axis into [num_chunks, scan_size, ...]."""
  return jax.tree.map(
      lambda t: jnp.reshape(
          t, (t.shape[0] // scan_size, scan_size) + t.shape[1:]),
      tree)


def flatten_batch_minibatch_axes(tree):
  """Flatten [ppo_batch, minibatch, ...] into one record axis."""
  return jax.tree.map(
      lambda t: jnp.reshape(t, (t.shape[0] * t.shape[1],) + t.shape[2:]),
      tree)


def summarize_policy_metrics(metrics_list: list[dict]) -> dict:
  if not metrics_list:
    return {}

  summarized = {}
  for key in metrics_list[0]:
    values = [metrics[key] for metrics in metrics_list]
    summarized[key] = dict(
        mean=jax_utils.add_n([jnp.mean(value) for value in values]) / len(values),
        min=functools.reduce(jnp.minimum, [jnp.min(value) for value in values]),
        max=functools.reduce(jnp.maximum, [jnp.max(value) for value in values]),
    )
  return summarized


def summarize_policy_metrics_jax(metrics: dict) -> dict:
  return {
      key: dict(
          mean=jnp.mean(value),
          min=jnp.min(value),
          max=jnp.max(value),
      )
      for key, value in metrics.items()
  }


def summarize_policy_array(value: Array) -> dict:
  return dict(
      mean=jnp.mean(value),
      min=jnp.min(value),
      max=jnp.max(value),
  )


def _broadcast_policy_mask(mask: Array, value: Array) -> Array:
  mask = jnp.asarray(mask, dtype=jnp.bool_)
  while mask.ndim < value.ndim:
    mask = mask[..., None]
  return mask


def masked_policy_mean(value: Array, mask: Array) -> Array:
  value = jnp.asarray(value)
  mask = _broadcast_policy_mask(mask, value)
  count = jnp.maximum(jnp.sum(mask), 1)
  return jnp.sum(jnp.where(mask, value, 0.0)) / count


def mask_policy_array(value: Array, mask: Array) -> Array:
  value = jnp.asarray(value)
  return jnp.where(_broadcast_policy_mask(mask, value), value, 0.0)


def summarize_policy_array_masked(value: Array, mask: Array) -> dict:
  value = jnp.asarray(value)
  mask = _broadcast_policy_mask(mask, value)
  any_valid = jnp.any(mask)
  masked = jnp.where(mask, value, 0.0)
  return dict(
      mean=masked_policy_mean(value, mask),
      min=jnp.where(any_valid, jnp.min(jnp.where(mask, value, jnp.inf)), 0.0),
      max=jnp.where(any_valid, jnp.max(jnp.where(mask, value, -jnp.inf)), 0.0),
  )


def valid_policy_step_mask(is_resetting: Array, delay: int) -> Array:
  # A policy sample from state[t] is executed after the observation delay. If a
  # reset occurs in [t + 1, t + D], that delayed controller is discarded before
  # it reaches the game. A reset at t + D + 1 is the terminal state after the
  # action was applied, so that sample is still valid.
  is_resetting = jnp.asarray(is_resetting, dtype=jnp.bool_)
  out_len = is_resetting.shape[0] - delay - 1
  invalid = jnp.zeros_like(is_resetting[1 + delay:])
  for offset in range(delay):
    invalid = jnp.logical_or(
        invalid,
        is_resetting[1 + offset:1 + offset + out_len],
    )
  return jnp.logical_not(invalid)


def summarize_policy_metric_summaries(metrics_list: list[dict]) -> dict:
  if not metrics_list:
    return {}

  summarized = {}
  for key in metrics_list[0]:
    values = [metrics[key] for metrics in metrics_list]
    summarized[key] = dict(
        mean=jax_utils.add_n([value['mean'] for value in values]) / len(values),
        min=functools.reduce(jnp.minimum, [value['min'] for value in values]),
        max=functools.reduce(jnp.maximum, [value['max'] for value in values]),
    )
  return summarized


def summarize_stacked_policy_metric_summaries(metrics: dict) -> dict:
  return {
      key: dict(
          mean=jnp.mean(value['mean']),
          min=jnp.min(value['min']),
          max=jnp.max(value['max']),
      )
      for key, value in metrics.items()
  }


def summarize_nested_means(metrics_list: list[dict]) -> dict:
  if not metrics_list:
    return {}
  return utils.map_nt(
      lambda *xs: jax_utils.add_n([jnp.mean(x) for x in xs]) / len(xs),
      *metrics_list,
  )


def block_until_ready(value):
  for leaf in jax.tree.leaves(value):
    if hasattr(leaf, 'block_until_ready'):
      leaf.block_until_ready()
    elif isinstance(leaf, np.ndarray):
      np.asarray(leaf)

def warmup_schedule(burnin_steps: int, base_value: float):
  burnin = optax.constant_schedule(0)
  normal = optax.constant_schedule(base_value)
  return optax.join_schedules([burnin, normal], [burnin_steps])


class Learner(nnx.Module):
  """Implements PPO for RL fine-tuning."""

  def __init__(
      self,
      config: LearnerConfig,
      policy: Policy,
      teacher: Policy,
      value_function: vf_lib.ValueFunction,
  ) -> None:
    self._config = config
    self.policy = policy
    self.teacher = teacher
    self.value_function = value_function

    self._controller_embedding = policy.controller_head.controller_embedding

    # The policy update is accumulated across the trajectories in one epoch.
    # For the first value_burnin_epochs we don't train the policy at all.
    # Then for the next optimizer_burnin_epochs we train the policy for a
    # single epoch with learning_rate=0 to initialize the optimizer state.

    self.policy_schedule = warmup_schedule(
        config.optimizer_burnin_epochs,
        config.learning_rate,
    )

    self.policy_optimizer = nnx.Optimizer(
        policy,
        optax.adam(self.policy_schedule, mu_dtype=jnp.float32),
        wrt=nnx.Param)
    self.value_optimizer = nnx.Optimizer(
        self.value_function,
        optax.adam(config.learning_rate, mu_dtype=jnp.float32),
        wrt=nnx.Param)

    self.discount = rl_lib.discount_from_halflife(config.reward_halflife)

    jit_unroll = nnx.jit(
        Learner._unroll_teacher_and_vf,
        donate_argnums=(0, 2),
        static_argnames=['train_value_function'],
    )
    self.unroll = jax_utils.cached_partial(jit_unroll, self)

    jit_unroll_grads = nnx.jit(Learner._unroll_teacher_and_vf_grads)
    self.unroll_value_grads = jax_utils.cached_partial(jit_unroll_grads, self)

    jit_stacked_unroll_grads = nnx.jit(
        Learner.unroll_stacked_minibatch_value_grads)
    self.jit_unroll_stacked_minibatch_value_grads = (
        jax_utils.cached_partial(jit_stacked_unroll_grads, self))

    jit_ppo_batch_unroll = nnx.jit(
        Learner.unroll_ppo_batch_stacked_minibatch_value_grads)
    self.jit_unroll_ppo_batch_stacked_minibatch_value_grads = (
        jax_utils.cached_partial(jit_ppo_batch_unroll, self))

    jit_equal_minibatch_update_train_and_eval = nnx.jit(
        Learner.ppo_equal_minibatch_update_train_and_eval)
    self.jit_ppo_equal_minibatch_update_train_and_eval = (
        jax_utils.cached_partial(
            jit_equal_minibatch_update_train_and_eval, self))

    jit_equal_minibatch_update_train = nnx.jit(
        Learner.ppo_equal_minibatch_update_train)
    self.jit_ppo_equal_minibatch_update_train = jax_utils.cached_partial(
        jit_equal_minibatch_update_train, self)

    jit_ppo_epoch = nnx.jit(
        Learner.ppo_epoch,
        donate_argnums=0,
        static_argnames=['train'])
    self.jit_ppo_epoch = jax_utils.cached_partial(jit_ppo_epoch, self)

    jit_ppo_stacked_minibatch_grads = nnx.jit(
        Learner.ppo_stacked_minibatch_grads)
    self.jit_ppo_stacked_minibatch_grads = jax_utils.cached_partial(
        jit_ppo_stacked_minibatch_grads, self)

    jit_ppo_stacked_minibatch_metrics = nnx.jit(
        Learner.ppo_stacked_minibatch_metrics)
    self.jit_ppo_stacked_minibatch_metrics = jax_utils.cached_partial(
        jit_ppo_stacked_minibatch_metrics, self)

    jit_ppo_stacked_minibatch_train_and_eval = nnx.jit(
        Learner.ppo_stacked_minibatch_train_and_eval)
    self.jit_ppo_stacked_minibatch_train_and_eval = jax_utils.cached_partial(
        jit_ppo_stacked_minibatch_train_and_eval, self)

    jit_ppo_chunked_minibatch_train_and_eval = nnx.jit(
        Learner.ppo_chunked_minibatch_train_and_eval)
    self.jit_ppo_chunked_minibatch_train_and_eval = jax_utils.cached_partial(
        jit_ppo_chunked_minibatch_train_and_eval, self)

    jit_ppo_chunked_minibatch_train = nnx.jit(
        Learner.ppo_chunked_minibatch_train)
    self.jit_ppo_chunked_minibatch_train = jax_utils.cached_partial(
        jit_ppo_chunked_minibatch_train, self)

  def initial_state(
      self, batch_size: int, rngs: tp.Optional[nnx.Rngs] = None,
  ) -> LearnerState:
    if rngs is None:
      rngs = nnx.Rngs(0)
    return LearnerState(
        teacher=self.teacher.initial_state(batch_size, rngs),
        value_function=self.value_function.initial_state(batch_size, rngs),
    )

  def policy_variables(self):
    """Returns policy state for actor update via evaluators.update_variables."""
    return self.policy.get_state()

  def _sum_leaves(self, embedding: embed.Embedding, struct) -> Array:
    return functools.reduce(jnp.add, embedding.flatten(struct))

  def _compute_kl_reference(self, logits_p, logits_q) -> Array:
    """Computes total KL(P||Q) summed over all controller components."""
    kls = self._controller_embedding.map(
        lambda e, lp, lq: e.kl_divergence(lp, lq),
        logits_p, logits_q)
    return self._sum_leaves(self._controller_embedding, kls)

  def _compute_entropy_reference(self, logits) -> Array:
    """Computes total entropy H(P) summed over all controller components."""
    entropies = self._controller_embedding.map(
        lambda e, l: e.entropy(l), logits)
    return self._sum_leaves(self._controller_embedding, entropies)

  def _get_log_prob_reference(self, logits, action) -> Array:
    """Computes log P(action | logits) summed over all controller components."""
    distances = self._controller_embedding.map(
        lambda e, l, a: e.distance(l, a), logits, action)
    return -self._sum_leaves(self._controller_embedding, distances)

  def _controller_reduce(self, embedding: embed.Embedding, f, *args) -> Array:
    if isinstance(embedding, embed.CompoundEmbedding):
      return self._controller_reduce(embedding._embed_mid, f, *args)
    if isinstance(embedding, embed.StructEmbedding):
      values = [
          self._controller_reduce(
              child,
              f,
              *(embedding.getter(arg, key) for arg in args),
          )
          for key, child in embedding.embedding
      ]
      return functools.reduce(jnp.add, values)
    return f(embedding, *args)

  def _leaf_log_prob(self, embedding: embed.Embedding, logits, action) -> Array:
    if isinstance(embedding, embed.BoolEmbedding):
      logits = jnp.squeeze(logits, axis=-1)
      return jnp.where(
          action,
          -jax.nn.softplus(-logits),
          -jax.nn.softplus(logits),
      )
    if isinstance(embedding, embed.OneHotEmbedding):
      log_probs = jax.nn.log_softmax(logits, axis=-1)
      return jnp.take_along_axis(
          log_probs, jnp.expand_dims(action.astype(jnp.int32), axis=-1),
          axis=-1,
      ).squeeze(axis=-1)
    return -embedding.distance(logits, action)

  def _leaf_entropy(self, embedding: embed.Embedding, logits) -> Array:
    if isinstance(embedding, embed.BoolEmbedding):
      logits = jnp.squeeze(logits, axis=-1)
      log_p1 = -jax.nn.softplus(-logits)
      log_p0 = -jax.nn.softplus(logits)
      p1 = jax.nn.sigmoid(logits)
      p0 = 1.0 - p1
      return -(p1 * log_p1 + p0 * log_p0)
    if isinstance(embedding, embed.OneHotEmbedding):
      log_probs = jax.nn.log_softmax(logits, axis=-1)
      probs = jnp.exp(log_probs)
      return -jnp.sum(probs * log_probs, axis=-1)
    return embedding.entropy(logits)

  def _leaf_kl(
      self,
      embedding: embed.Embedding,
      logits_p,
      logits_q,
  ) -> Array:
    if isinstance(embedding, embed.BoolEmbedding):
      logits_p = jnp.squeeze(logits_p, axis=-1)
      logits_q = jnp.squeeze(logits_q, axis=-1)
      log_p1 = -jax.nn.softplus(-logits_p)
      log_p0 = -jax.nn.softplus(logits_p)
      log_q1 = -jax.nn.softplus(-logits_q)
      log_q0 = -jax.nn.softplus(logits_q)
      p1 = jax.nn.sigmoid(logits_p)
      p0 = 1.0 - p1
      return p1 * (log_p1 - log_q1) + p0 * (log_p0 - log_q0)
    if isinstance(embedding, embed.OneHotEmbedding):
      log_p = jax.nn.log_softmax(logits_p, axis=-1)
      log_q = jax.nn.log_softmax(logits_q, axis=-1)
      p = jnp.exp(log_p)
      return jnp.sum(p * (log_p - log_q), axis=-1)
    return embedding.kl_divergence(logits_p, logits_q)

  def _compute_kl(self, logits_p, logits_q) -> Array:
    """Computes total KL(P||Q) summed over all controller components."""
    return self._controller_reduce(
        self._controller_embedding, self._leaf_kl, logits_p, logits_q)

  def _compute_entropy(self, logits) -> Array:
    """Computes total entropy H(P) summed over all controller components."""
    return self._controller_reduce(
        self._controller_embedding, self._leaf_entropy, logits)

  def _get_log_prob(self, logits, action) -> Array:
    """Computes log P(action | logits) summed over all controller components."""
    return self._controller_reduce(
        self._controller_embedding, self._leaf_log_prob, logits, action)

  def _sum_controller_terms(self, values: list[tuple[Array, ...]]):
    return tuple(
        functools.reduce(jnp.add, terms)
        for terms in zip(*values))

  def _controller_policy_terms_reduce(
      self,
      embedding: embed.Embedding,
      new_logits,
      actor_logits,
      teacher_logits,
      action,
  ) -> tuple[Array, Array, Array, Array, Array, Array]:
    if isinstance(embedding, embed.CompoundEmbedding):
      return self._controller_policy_terms_reduce(
          embedding._embed_mid, new_logits, actor_logits, teacher_logits, action)
    if isinstance(embedding, embed.StructEmbedding):
      values = [
          self._controller_policy_terms_reduce(
              child,
              embedding.getter(new_logits, key),
              embedding.getter(actor_logits, key),
              embedding.getter(teacher_logits, key),
              embedding.getter(action, key),
          )
          for key, child in embedding.embedding
      ]
      return self._sum_controller_terms(values)
    return self._leaf_policy_terms(
        embedding, new_logits, actor_logits, teacher_logits, action)

  def _leaf_policy_terms(
      self,
      embedding: embed.Embedding,
      new_logits,
      actor_logits,
      teacher_logits,
      action,
  ) -> tuple[Array, Array, Array, Array, Array, Array]:
    if isinstance(embedding, embed.BoolEmbedding):
      new_logits = jnp.squeeze(new_logits, axis=-1)
      actor_logits = jnp.squeeze(actor_logits, axis=-1)
      teacher_logits = jnp.squeeze(teacher_logits, axis=-1)

      new_log_p1 = -jax.nn.softplus(-new_logits)
      new_log_p0 = -jax.nn.softplus(new_logits)
      actor_log_p1 = -jax.nn.softplus(-actor_logits)
      actor_log_p0 = -jax.nn.softplus(actor_logits)
      teacher_log_p1 = -jax.nn.softplus(-teacher_logits)
      teacher_log_p0 = -jax.nn.softplus(teacher_logits)

      new_p1 = jax.nn.sigmoid(new_logits)
      new_p0 = 1.0 - new_p1
      actor_p1 = jax.nn.sigmoid(actor_logits)
      actor_p0 = 1.0 - actor_p1
      teacher_p1 = jax.nn.sigmoid(teacher_logits)
      teacher_p0 = 1.0 - teacher_p1

      new_log_prob = jnp.where(action, new_log_p1, new_log_p0)
      actor_log_prob = jnp.where(action, actor_log_p1, actor_log_p0)
      teacher_kl = (
          new_p1 * (new_log_p1 - teacher_log_p1)
          + new_p0 * (new_log_p0 - teacher_log_p0))
      actor_kl = (
          actor_p1 * (actor_log_p1 - new_log_p1)
          + actor_p0 * (actor_log_p0 - new_log_p0))
      reverse_teacher_kl = (
          teacher_p1 * (teacher_log_p1 - new_log_p1)
          + teacher_p0 * (teacher_log_p0 - new_log_p0))
      entropy = -(new_p1 * new_log_p1 + new_p0 * new_log_p0)
      return (
          actor_log_prob,
          new_log_prob,
          teacher_kl,
          actor_kl,
          reverse_teacher_kl,
          entropy,
      )

    if isinstance(embedding, embed.OneHotEmbedding):
      new_log_probs = jax.nn.log_softmax(new_logits, axis=-1)
      actor_log_probs = jax.nn.log_softmax(actor_logits, axis=-1)
      teacher_log_probs = jax.nn.log_softmax(teacher_logits, axis=-1)
      new_probs = jnp.exp(new_log_probs)
      actor_probs = jnp.exp(actor_log_probs)
      teacher_probs = jnp.exp(teacher_log_probs)
      action_indices = jnp.expand_dims(action.astype(jnp.int32), axis=-1)

      new_log_prob = jnp.take_along_axis(
          new_log_probs, action_indices, axis=-1).squeeze(axis=-1)
      actor_log_prob = jnp.take_along_axis(
          actor_log_probs, action_indices, axis=-1).squeeze(axis=-1)
      teacher_kl = jnp.sum(
          new_probs * (new_log_probs - teacher_log_probs), axis=-1)
      actor_kl = jnp.sum(
          actor_probs * (actor_log_probs - new_log_probs), axis=-1)
      reverse_teacher_kl = jnp.sum(
          teacher_probs * (teacher_log_probs - new_log_probs), axis=-1)
      entropy = -jnp.sum(new_probs * new_log_probs, axis=-1)
      return (
          actor_log_prob,
          new_log_prob,
          teacher_kl,
          actor_kl,
          reverse_teacher_kl,
          entropy,
      )

    return (
        self._leaf_log_prob(embedding, actor_logits, action),
        self._leaf_log_prob(embedding, new_logits, action),
        self._leaf_kl(embedding, new_logits, teacher_logits),
        self._leaf_kl(embedding, actor_logits, new_logits),
        self._leaf_kl(embedding, teacher_logits, new_logits),
        self._leaf_entropy(embedding, new_logits),
    )

  def _controller_policy_terms(
      self,
      new_logits,
      actor_logits,
      teacher_logits,
      action,
  ) -> tuple[Array, Array, Array, Array, Array, Array]:
    return self._controller_policy_terms_reduce(
        self._controller_embedding,
        new_logits,
        actor_logits,
        teacher_logits,
        action,
    )

  def _policy_loss_and_metrics(
      self,
      policy: Policy,
      outputs: LearnerOutputs,
      trajectory: Trajectory,
  ) -> tuple[Array, dict]:
    delay = self.policy.delay  # D
    remove_first = lambda t: t[delay:] if delay > 0 else t
    remove_last = lambda t: t[:t.shape[0] - delay] if delay > 0 else t

    # Advantages from [0, U]: take [D, U] -> U-D steps.
    advantages = jax.lax.stop_gradient(outputs.value.advantages[delay:])

    # Policy frames: states [0, U-D+1], actions [D, U+1].
    policy_frames = data.Frames(
        state_action=data.StateAction(
            state=jax.tree.map(remove_last, trajectory.states),
            action=jax.tree.map(remove_first, trajectory.actions.controller_state),
            name=remove_last(trajectory.name),
        ),
        is_resetting=remove_last(trajectory.is_resetting),
        reward=remove_first(trajectory.rewards),
    )
    policy_frames = reset_frame_actions(
        policy_frames,
        policy.controller_head.dummy_controller(policy_frames.is_resetting.shape),
    )

    # Actor (old policy) logits and log probs for steps [D+1, U+1].
    actor_outputs = utils.map_single_structure(
        lambda t: t[1 + delay:], trajectory.actions)
    actor_logits = actor_outputs.logits

    policy_outputs = policy.unroll_logits(policy_frames, trajectory.initial_state)
    new_logits = policy_outputs.logits

    # Teacher logits: [D, U+D] -> truncate last D -> [D, U].
    # Note: no stop_gradient needed since teacher has no trainable variables.
    teacher_logits = jax.tree.map(remove_last, outputs.teacher.logits)
    # KL divergences are computed over full output distribution, not just
    # sampled action. Forward KL to teacher incentivizes refining human
    # actions over covering all of them.
    (
        actor_log_probs,
        new_log_probs,
        teacher_kl,
        actor_kl,
        reverse_teacher_kl,
        entropy,
    ) = self._controller_policy_terms(
        new_logits, actor_logits, teacher_logits, actor_outputs.controller_state)
    actor_log_probs = jax.lax.stop_gradient(actor_log_probs)
    valid_steps = valid_policy_step_mask(trajectory.is_resetting, delay)

    # PPO clipped objective.
    log_rhos = new_log_probs - actor_log_probs
    rhos = jnp.exp(log_rhos)

    eps = self._config.ppo.epsilon
    clipped_log_rhos = jnp.clip(log_rhos, -eps, eps)
    clipped_rhos = jnp.exp(clipped_log_rhos)

    ppo_objective = jnp.minimum(rhos * advantages, clipped_rhos * advantages)

    weighted_loss = (
        - self._config.policy_gradient_weight * ppo_objective
        + self._config.ppo.beta * actor_kl
        + self._config.kl_teacher_weight * teacher_kl
        + self._config.reverse_kl_teacher_weight * reverse_teacher_kl
        - self._config.entropy_weight * entropy
    )
    loss = masked_policy_mean(weighted_loss, valid_steps)

    metrics = dict(
        total_loss=loss,
        ppo_objective=mask_policy_array(ppo_objective, valid_steps),
        teacher_kl=mask_policy_array(teacher_kl, valid_steps),
        entropy=mask_policy_array(entropy, valid_steps),
        actor_kl=mask_policy_array(actor_kl, valid_steps),
        reverse_teacher_kl=mask_policy_array(reverse_teacher_kl, valid_steps),
    )
    return loss, metrics

  def _policy_loss_and_metric_summary(
      self,
      policy: Policy,
      outputs: LearnerOutputs,
      trajectory: Trajectory,
  ) -> tuple[Array, dict]:
    delay = self.policy.delay  # D
    remove_first = lambda t: t[delay:] if delay > 0 else t
    remove_last = lambda t: t[:t.shape[0] - delay] if delay > 0 else t

    advantages = jax.lax.stop_gradient(outputs.value.advantages[delay:])

    policy_frames = data.Frames(
        state_action=data.StateAction(
            state=jax.tree.map(remove_last, trajectory.states),
            action=jax.tree.map(remove_first, trajectory.actions.controller_state),
            name=remove_last(trajectory.name),
        ),
        is_resetting=remove_last(trajectory.is_resetting),
        reward=remove_first(trajectory.rewards),
    )
    policy_frames = reset_frame_actions(
        policy_frames,
        policy.controller_head.dummy_controller(policy_frames.is_resetting.shape),
    )

    actor_outputs = utils.map_single_structure(
        lambda t: t[1 + delay:], trajectory.actions)
    actor_logits = actor_outputs.logits

    policy_outputs = policy.unroll_logits(policy_frames, trajectory.initial_state)
    new_logits = policy_outputs.logits

    teacher_logits = jax.tree.map(remove_last, outputs.teacher.logits)
    (
        actor_log_probs,
        new_log_probs,
        teacher_kl,
        actor_kl,
        reverse_teacher_kl,
        entropy,
    ) = self._controller_policy_terms(
        new_logits, actor_logits, teacher_logits, actor_outputs.controller_state)
    actor_log_probs = jax.lax.stop_gradient(actor_log_probs)
    valid_steps = valid_policy_step_mask(trajectory.is_resetting, delay)

    log_rhos = new_log_probs - actor_log_probs
    rhos = jnp.exp(log_rhos)

    eps = self._config.ppo.epsilon
    clipped_log_rhos = jnp.clip(log_rhos, -eps, eps)
    clipped_rhos = jnp.exp(clipped_log_rhos)

    ppo_objective = jnp.minimum(rhos * advantages, clipped_rhos * advantages)

    weighted_loss = (
        - self._config.policy_gradient_weight * ppo_objective
        + self._config.ppo.beta * actor_kl
        + self._config.kl_teacher_weight * teacher_kl
        + self._config.reverse_kl_teacher_weight * reverse_teacher_kl
        - self._config.entropy_weight * entropy
    )
    loss = masked_policy_mean(weighted_loss, valid_steps)

    metrics = dict(
        total_loss=summarize_policy_array_masked(weighted_loss, valid_steps),
        ppo_objective=summarize_policy_array_masked(
            ppo_objective, valid_steps),
        teacher_kl=summarize_policy_array_masked(teacher_kl, valid_steps),
        entropy=summarize_policy_array_masked(entropy, valid_steps),
        actor_kl=summarize_policy_array_masked(actor_kl, valid_steps),
        reverse_teacher_kl=summarize_policy_array_masked(
            reverse_teacher_kl, valid_steps),
    )
    return loss, metrics

  def _unroll_teacher_and_vf(
      self,
      trajectory: Trajectory,
      initial_state: LearnerState,
      *,
      train_value_function: bool = False,
  ):
    teacher_frames = get_delayed_frames(trajectory)
    teacher_frames = reset_frame_actions(
        teacher_frames,
        self.teacher.controller_head.dummy_controller(
            teacher_frames.is_resetting.shape),
    )
    teacher_outputs = self.teacher.unroll_logits(
        teacher_frames, initial_state.teacher)

    # Run value function (with or without gradient update).
    value_frames = get_frames(trajectory)
    value_frames = reset_frame_actions(
        value_frames,
        self.policy.controller_head.dummy_controller(
            value_frames.is_resetting.shape),
    )

    if train_value_function:
      def value_loss_fn(vf: vf_lib.ValueFunction):
        outputs, final_state = vf.loss(
            value_frames, initial_state.value_function, self.discount)
        return jnp.mean(outputs.loss), (outputs, final_state)

      grads, (value_outputs, value_final_state) = jax_utils.grad_with_aux(
          value_loss_fn)(self.value_function)
      self.value_optimizer.update(self.value_function, grads)
    else:
      value_outputs, value_final_state = self.value_function.loss(
          value_frames, initial_state.value_function, self.discount)

    final_state = LearnerState(
        teacher=teacher_outputs.final_state,
        value_function=value_final_state,
    )
    outputs = LearnerOutputs(
        teacher=teacher_outputs,
        value=value_outputs,
    )
    return outputs, final_state

  def _unroll_teacher_and_vf_grads(
      self,
      trajectory: Trajectory,
      initial_state: LearnerState,
  ):
    """Unroll teacher and compute value-function grads without applying them."""
    teacher_frames = get_delayed_frames(trajectory)
    teacher_frames = reset_frame_actions(
        teacher_frames,
        self.teacher.controller_head.dummy_controller(
            teacher_frames.is_resetting.shape),
    )
    teacher_outputs = self.teacher.unroll_logits(
        teacher_frames, initial_state.teacher)

    value_frames = get_frames(trajectory)
    value_frames = reset_frame_actions(
        value_frames,
        self.policy.controller_head.dummy_controller(
            value_frames.is_resetting.shape),
    )

    def value_loss_fn(vf: vf_lib.ValueFunction):
      outputs, final_state = vf.loss(
          value_frames, initial_state.value_function, self.discount)
      return jnp.mean(outputs.loss), (outputs, final_state)

    grads, (value_outputs, value_final_state) = jax_utils.grad_with_aux(
        value_loss_fn)(self.value_function)

    final_state = LearnerState(
        teacher=teacher_outputs.final_state,
        value_function=value_final_state,
    )
    outputs = LearnerOutputs(
        teacher=teacher_outputs,
        value=value_outputs,
    )
    return grads, outputs, final_state

  def unroll_stacked_minibatch_value_grads(
      self,
      trajectories: Trajectory,
      initial_states: LearnerState,
  ) -> tuple[jax_utils.Grads, LearnerOutputs, LearnerState]:
    """Unroll teacher/value and sum value grads over stacked minibatches."""
    @nnx.scan(
        in_axes=(None, 0, 0),
        out_axes=(0, 0, 0),
    )
    def scan_fn(
        learner: Learner,
        trajectory: Trajectory,
        initial_state: LearnerState,
    ) -> tuple[LearnerOutputs, LearnerState, jax_utils.Grads]:
      value_grads, outputs, final_state = (
          learner._unroll_teacher_and_vf_grads(trajectory, initial_state))
      return outputs, final_state, value_grads

    outputs, final_states, value_grads = scan_fn(
        self, trajectories, initial_states)
    value_grads_sum = jax.tree.map(lambda g: jnp.sum(g, axis=0), value_grads)
    return value_grads_sum, outputs, final_states

  def unroll_ppo_batch_stacked_minibatch_value_grads(
      self,
      trajectories: Trajectory,
      initial_state: LearnerState,
  ) -> tuple[LearnerOutputs, LearnerState]:
    """Unroll PPO batches in one compiled scan.

    The scan carries recurrent teacher/value state between PPO batches and
    applies the value optimizer once per batch, preserving the original update
    order while avoiding separate host dispatches for each PPO batch.
    """
    @nnx.scan(
        in_axes=(None, 0, nnx.Carry),
        out_axes=(0, nnx.Carry),
    )
    def batch_scan(
        learner: Learner,
        trajectory_minibatches: Trajectory,
        hidden_state: LearnerState,
    ) -> tuple[LearnerOutputs, LearnerState]:
      state_minibatches = split_learner_state_minibatches(
          hidden_state, learner._config.ppo.minibatch_size)
      value_grads_sum, outputs, final_states = (
          learner.unroll_stacked_minibatch_value_grads(
              trajectory_minibatches, state_minibatches))
      num_minibatches = trajectory_minibatches.rewards.shape[0]
      value_grads = jax.tree.map(
          lambda g: g / num_minibatches, value_grads_sum)
      learner.value_optimizer.update(learner.value_function, value_grads)
      next_hidden_state = merge_learner_state_minibatches(final_states)
      return outputs, next_hidden_state

    outputs, final_state = batch_scan(self, trajectories, initial_state)
    return outputs, final_state

  def ppo_grads(
      self,
      outputs: LearnerOutputs,
      trajectory: Trajectory,
  ) -> tp.Tuple[tp.Any, dict]:
    """Computes policy gradients for one PPO step.

    Value function outputs are for [0, U] while policy outputs are for
    [D, U+D]. This means we can only train on steps [D, U].

    Args:
      outputs: Pre-computed teacher and value function outputs.
      trajectory: The collected trajectory.

    Returns:
      Tuple of (gradients, metrics dict).
    """
    def policy_loss_fn(policy: Policy):
      return self._policy_loss_and_metrics(policy, outputs, trajectory)

    grads, metrics = jax_utils.grad_with_aux(policy_loss_fn)(self.policy)
    return grads, metrics

  def ppo_grads_summary(
      self,
      outputs: LearnerOutputs,
      trajectory: Trajectory,
  ) -> tp.Tuple[tp.Any, dict]:
    """Like ppo_grads, but returns scalar metric summaries for minibatching."""
    def policy_loss_fn(policy: Policy):
      return self._policy_loss_and_metric_summary(policy, outputs, trajectory)

    grads, metrics = jax_utils.grad_with_aux(policy_loss_fn)(self.policy)
    return grads, metrics

  def ppo_metrics(
      self,
      outputs: LearnerOutputs,
      trajectory: Trajectory,
  ) -> dict:
    _, metrics = self._policy_loss_and_metrics(
        self.policy, outputs, trajectory)
    return metrics

  def ppo_metric_summary(
      self,
      outputs: LearnerOutputs,
      trajectory: Trajectory,
  ) -> dict:
    _, metrics = self._policy_loss_and_metric_summary(
        self.policy, outputs, trajectory)
    return metrics

  def ppo_epoch(
      self,
      learner_outputs: list[LearnerOutputs],
      trajectories: list[Trajectory],
      train: bool = True,
  ) -> dict:
    """One epoch of PPO: accumulate gradients over all trajectories."""
    if not train:
      metrics = [
          self.ppo_metrics(outputs, trajectory)
          for outputs, trajectory in zip(learner_outputs, trajectories)
      ]
      metrics = summarize_policy_metrics(metrics)
      metrics['grads'] = dict(
          norms={},
          max_norm=jnp.array(0.0),
      )
      return metrics

    grad_shapes, _ = nnx.eval_shape(
        Learner.ppo_grads, self, learner_outputs[0], trajectories[0])
    zero_grads = jax.tree.map(jnp.zeros_like, grad_shapes)

    batched_learner_outputs = jax.tree.map(
      lambda *xs: jnp.stack(xs), *learner_outputs)
    batched_trajectories = jax.tree.map(
      lambda *xs: jnp.stack(xs), *trajectories)

    @nnx.scan(
        in_axes=(None, 0, 0, nnx.Carry),
        out_axes=(0, nnx.Carry),
    )
    def scan_fn(
        learner: Learner,
        learner_outputs: LearnerOutputs,
        trajectory: Trajectory,
        grads_acc: jax_utils.Grads,
    ) -> tuple[dict, jax_utils.Grads]:
      grads, metrics = learner.ppo_grads(learner_outputs, trajectory)
      new_grads_acc = jax.tree.map(jnp.add, grads_acc, grads)
      return metrics, new_grads_acc

    metrics, grads_sum = scan_fn(
        self, batched_learner_outputs, batched_trajectories, zero_grads)
    n = len(trajectories)
    grads = jax.tree.map(lambda g: g / n, grads_sum)

    grads_dict = nnx.to_pure_dict(grads)
    grad_norms = jax.tree.map(jnp.linalg.norm, grads_dict)
    max_grad_norm = jax.tree.reduce(jnp.maximum, grad_norms)

    metrics['grads'] = dict(
        norms=grad_norms,
        max_norm=max_grad_norm,
    )

    if train:
      self.policy_optimizer.update(self.policy, grads)

    actor_kl = metrics['actor_kl']
    metrics['actor_kl'] = dict(
        mean=jnp.mean(actor_kl),
        max=jnp.max(actor_kl),
    )
    return metrics

  def ppo_epoch_minibatched(
      self,
      learner_outputs: list[list[tuple[int, int, LearnerOutputs]]],
      trajectories: list[Trajectory],
      train: bool = True,
      profile: dict | None = None,
  ) -> dict:
    """One PPO epoch with gradient accumulation over batch-axis minibatches."""
    metrics_list = []
    grads_sum = None
    total_minibatches = 0
    scan_size = max(1, self._config.ppo.minibatch_scan_size)
    chunk_outputs = []
    chunk_trajectories = []

    def flush_chunk():
      nonlocal grads_sum, total_minibatches
      if not chunk_outputs:
        return
      stacked_outputs = jax.tree.map(lambda *xs: jnp.stack(xs), *chunk_outputs)
      stacked_trajectories = jax.tree.map(
          lambda *xs: jnp.stack(xs), *chunk_trajectories)
      chunk_count = len(chunk_outputs)
      if train:
        grad_start = time.perf_counter()
        chunk_grads, metrics = self.jit_ppo_stacked_minibatch_grads(
            stacked_outputs, stacked_trajectories)
        if profile is not None:
          block_until_ready((chunk_grads, metrics))
          profile['ppo_grad_s'] += time.perf_counter() - grad_start
          profile['ppo_grad_minibatches'] += chunk_count
          profile['ppo_grad_chunks'] += 1
        if grads_sum is None:
          grads_sum = chunk_grads
        else:
          accum_start = time.perf_counter()
          grads_sum = jax.tree.map(jnp.add, grads_sum, chunk_grads)
          if profile is not None:
            block_until_ready(grads_sum)
            profile['ppo_grad_accum_s'] += time.perf_counter() - accum_start
        total_minibatches += chunk_count
      else:
        eval_start = time.perf_counter()
        metrics = self.jit_ppo_stacked_minibatch_metrics(
            stacked_outputs, stacked_trajectories)
        if profile is not None:
          block_until_ready(metrics)
          profile['ppo_eval_s'] += time.perf_counter() - eval_start
          profile['ppo_eval_minibatches'] += chunk_count
          profile['ppo_eval_chunks'] += 1
      metrics_list.append(metrics)
      chunk_outputs.clear()
      chunk_trajectories.clear()

    for output_slices, trajectory in zip(learner_outputs, trajectories):
      batch_size = trajectory_batch_size(trajectory)
      for start, end, outputs in output_slices:
        weight = end - start
        if weight != self._config.ppo.minibatch_size:
          raise ValueError(
              'chunked PPO minibatching requires equal-sized minibatches')
        slice_start = time.perf_counter()
        trajectory_slice = slice_trajectory(trajectory, start, end)
        if profile is not None:
          profile['ppo_slice_s'] += time.perf_counter() - slice_start
        chunk_outputs.append(outputs)
        chunk_trajectories.append(trajectory_slice)
        if profile is not None:
          key = 'ppo_grad_frames' if train else 'ppo_eval_frames'
          profile[key] += (
              weight * (trajectory.rewards.shape[0] - self.policy.delay))
        if len(chunk_outputs) >= scan_size:
          flush_chunk()
      if output_slices[-1][1] != batch_size:
        raise ValueError('learner output slices did not cover trajectory batch')
    flush_chunk()

    if train:
      assert grads_sum is not None
      avg_start = time.perf_counter()
      grads = jax.tree.map(lambda g: g / total_minibatches, grads_sum)
      if profile is not None:
        block_until_ready(grads)
        profile['ppo_grad_average_s'] += time.perf_counter() - avg_start
      opt_start = time.perf_counter()
      self.policy_optimizer.update(self.policy, grads)
      if profile is not None:
        block_until_ready(self.policy)
        profile['policy_optimizer_s'] += time.perf_counter() - opt_start

      norm_start = time.perf_counter()
      grads_dict = nnx.to_pure_dict(grads)
      grad_norms = jax.tree.map(jnp.linalg.norm, grads_dict)
      max_grad_norm = jax.tree.reduce(jnp.maximum, grad_norms)
      if profile is not None:
        block_until_ready(max_grad_norm)
        profile['grad_norm_s'] += time.perf_counter() - norm_start
    else:
      grad_norms = {}
      max_grad_norm = jnp.array(0.0)

    summarize_start = time.perf_counter()
    metrics = summarize_policy_metric_summaries(metrics_list)
    if profile is not None:
      block_until_ready(metrics)
      profile['policy_metrics_s'] += time.perf_counter() - summarize_start
    metrics['grads'] = dict(
        norms=grad_norms,
        max_norm=max_grad_norm,
    )
    return metrics

  def ppo_stacked_minibatch_grads(
      self,
      learner_outputs: LearnerOutputs,
      trajectories: Trajectory,
  ) -> tp.Tuple[tp.Any, dict]:
    """Accumulate PPO grads over a small stack of equal-size minibatches."""
    first_outputs = jax.tree.map(lambda t: t[0], learner_outputs)
    first_trajectory = jax.tree.map(lambda t: t[0], trajectories)
    grad_shapes, _ = nnx.eval_shape(
        Learner.ppo_grads_summary, self, first_outputs, first_trajectory)
    zero_grads = jax.tree.map(jnp.zeros_like, grad_shapes)

    @nnx.scan(
        in_axes=(None, 0, 0, nnx.Carry),
        out_axes=(0, nnx.Carry),
    )
    def scan_fn(
        learner: Learner,
        learner_outputs: LearnerOutputs,
        trajectory: Trajectory,
        grads_acc: jax_utils.Grads,
    ) -> tuple[dict, jax_utils.Grads]:
      grads, metrics = learner.ppo_grads_summary(learner_outputs, trajectory)
      new_grads_acc = jax.tree.map(jnp.add, grads_acc, grads)
      return metrics, new_grads_acc

    metrics, grads_sum = scan_fn(
        self, learner_outputs, trajectories, zero_grads)
    return grads_sum, summarize_stacked_policy_metric_summaries(metrics)

  def ppo_stacked_minibatch_metrics(
      self,
      learner_outputs: LearnerOutputs,
      trajectories: Trajectory,
  ) -> dict:
    """Compute PPO metric summaries over a small stack of minibatches."""
    @nnx.scan(
        in_axes=(None, 0, 0),
        out_axes=0,
    )
    def scan_fn(
        learner: Learner,
        learner_outputs: LearnerOutputs,
        trajectory: Trajectory,
    ) -> dict:
      return learner.ppo_metric_summary(learner_outputs, trajectory)

    metrics = scan_fn(self, learner_outputs, trajectories)
    return summarize_stacked_policy_metric_summaries(metrics)

  def ppo_stacked_minibatch_train_and_eval(
      self,
      learner_outputs: LearnerOutputs,
      trajectories: Trajectory,
  ) -> tuple[dict, dict]:
    """Apply one PPO update and compute post-update metrics in one JIT."""
    grads_sum, train_metrics = self.ppo_stacked_minibatch_grads(
        learner_outputs, trajectories)
    chunk_count = trajectories.rewards.shape[0]
    grads = jax.tree.map(lambda g: g / chunk_count, grads_sum)
    self.policy_optimizer.update(self.policy, grads)

    grads_dict = nnx.to_pure_dict(grads)
    grad_norms = jax.tree.map(jnp.linalg.norm, grads_dict)
    max_grad_norm = jax.tree.reduce(jnp.maximum, grad_norms)
    train_metrics['grads'] = dict(
        norms=grad_norms,
        max_norm=max_grad_norm,
    )

    final_metrics = self.ppo_stacked_minibatch_metrics(
        learner_outputs, trajectories)
    final_metrics['grads'] = dict(
        norms={},
        max_norm=jnp.array(0.0),
    )
    return train_metrics, final_metrics

  def ppo_chunked_minibatch_train_and_eval(
      self,
      learner_outputs: LearnerOutputs,
      trajectories: Trajectory,
  ) -> tuple[dict, dict]:
    """Apply one PPO update and post-update eval over stacked chunks.

    `learner_outputs` and `trajectories` are shaped
    [num_chunks, minibatches_per_chunk, ...]. Keeping both the gradient pass and
    post-update metrics in one compiled call avoids Python dispatch and repeated
    host/device restacking between minibatch chunks.
    """
    first_outputs = jax.tree.map(lambda t: t[0, 0], learner_outputs)
    first_trajectory = jax.tree.map(lambda t: t[0, 0], trajectories)
    grad_shapes, _ = nnx.eval_shape(
        Learner.ppo_grads_summary, self, first_outputs, first_trajectory)
    zero_grads = jax.tree.map(jnp.zeros_like, grad_shapes)

    @nnx.scan(
        in_axes=(None, 0, 0, nnx.Carry),
        out_axes=(0, nnx.Carry),
    )
    def grad_scan(
        learner: Learner,
        learner_outputs: LearnerOutputs,
        trajectory: Trajectory,
        grads_acc: jax_utils.Grads,
    ) -> tuple[dict, jax_utils.Grads]:
      chunk_grads, chunk_metrics = learner.ppo_stacked_minibatch_grads(
          learner_outputs, trajectory)
      new_grads_acc = jax.tree.map(jnp.add, grads_acc, chunk_grads)
      return chunk_metrics, new_grads_acc

    train_metrics, grads_sum = grad_scan(
        self, learner_outputs, trajectories, zero_grads)
    total_minibatches = (
        trajectories.rewards.shape[0] * trajectories.rewards.shape[1])
    grads = jax.tree.map(lambda g: g / total_minibatches, grads_sum)
    self.policy_optimizer.update(self.policy, grads)

    grads_dict = nnx.to_pure_dict(grads)
    grad_norms = jax.tree.map(jnp.linalg.norm, grads_dict)
    max_grad_norm = jax.tree.reduce(jnp.maximum, grad_norms)
    train_metrics = summarize_stacked_policy_metric_summaries(train_metrics)
    train_metrics['grads'] = dict(
        norms=grad_norms,
        max_norm=max_grad_norm,
    )

    @nnx.scan(
        in_axes=(None, 0, 0),
        out_axes=0,
    )
    def eval_scan(
        learner: Learner,
        learner_outputs: LearnerOutputs,
        trajectory: Trajectory,
    ) -> dict:
      return learner.ppo_stacked_minibatch_metrics(
          learner_outputs, trajectory)

    final_metrics = eval_scan(self, learner_outputs, trajectories)
    final_metrics = summarize_stacked_policy_metric_summaries(final_metrics)
    final_metrics['grads'] = dict(
        norms={},
        max_norm=jnp.array(0.0),
    )
    return train_metrics, final_metrics

  def ppo_chunked_minibatch_train(
      self,
      learner_outputs: LearnerOutputs,
      trajectories: Trajectory,
  ) -> dict:
    """Apply one PPO update over stacked chunks without post-update eval."""
    first_outputs = jax.tree.map(lambda t: t[0, 0], learner_outputs)
    first_trajectory = jax.tree.map(lambda t: t[0, 0], trajectories)
    grad_shapes, _ = nnx.eval_shape(
        Learner.ppo_grads_summary, self, first_outputs, first_trajectory)
    zero_grads = jax.tree.map(jnp.zeros_like, grad_shapes)

    @nnx.scan(
        in_axes=(None, 0, 0, nnx.Carry),
        out_axes=(0, nnx.Carry),
    )
    def grad_scan(
        learner: Learner,
        learner_outputs: LearnerOutputs,
        trajectory: Trajectory,
        grads_acc: jax_utils.Grads,
    ) -> tuple[dict, jax_utils.Grads]:
      chunk_grads, chunk_metrics = learner.ppo_stacked_minibatch_grads(
          learner_outputs, trajectory)
      new_grads_acc = jax.tree.map(jnp.add, grads_acc, chunk_grads)
      return chunk_metrics, new_grads_acc

    train_metrics, grads_sum = grad_scan(
        self, learner_outputs, trajectories, zero_grads)
    total_minibatches = (
        trajectories.rewards.shape[0] * trajectories.rewards.shape[1])
    grads = jax.tree.map(lambda g: g / total_minibatches, grads_sum)
    self.policy_optimizer.update(self.policy, grads)

    grads_dict = nnx.to_pure_dict(grads)
    grad_norms = jax.tree.map(jnp.linalg.norm, grads_dict)
    max_grad_norm = jax.tree.reduce(jnp.maximum, grad_norms)
    train_metrics = summarize_stacked_policy_metric_summaries(train_metrics)
    train_metrics['grads'] = dict(
        norms=grad_norms,
        max_norm=max_grad_norm,
    )
    return train_metrics

  def ppo_equal_minibatch_update_train_and_eval(
      self,
      trajectories: Trajectory,
      initial_state: LearnerState,
  ) -> tuple[LearnerState, dict, dict, dict]:
    """Run teacher/value and PPO for stacked equal-minibatch PPO batches."""
    learner_outputs, hidden_state = (
        self.unroll_ppo_batch_stacked_minibatch_value_grads(
            trajectories, initial_state))
    value_metrics = utils.map_nt(
        lambda x: jnp.mean(x), learner_outputs.value.metrics)

    scan_size = self._config.ppo.minibatch_scan_size
    flat_outputs = flatten_batch_minibatch_axes(learner_outputs)
    flat_trajectories = flatten_batch_minibatch_axes(trajectories)
    chunked_outputs = group_record_axis(flat_outputs, scan_size)
    chunked_trajectories = group_record_axis(flat_trajectories, scan_size)
    train_metrics, final_metrics = self.ppo_chunked_minibatch_train_and_eval(
        chunked_outputs, chunked_trajectories)
    return hidden_state, value_metrics, train_metrics, final_metrics

  def ppo_equal_minibatch_update_train(
      self,
      trajectories: Trajectory,
      initial_state: LearnerState,
  ) -> tuple[LearnerState, dict, dict]:
    """Run teacher/value and PPO train pass for stacked equal minibatches."""
    learner_outputs, hidden_state = (
        self.unroll_ppo_batch_stacked_minibatch_value_grads(
            trajectories, initial_state))
    value_metrics = utils.map_nt(
        lambda x: jnp.mean(x), learner_outputs.value.metrics)

    scan_size = self._config.ppo.minibatch_scan_size
    flat_outputs = flatten_batch_minibatch_axes(learner_outputs)
    flat_trajectories = flatten_batch_minibatch_axes(trajectories)
    chunked_outputs = group_record_axis(flat_outputs, scan_size)
    chunked_trajectories = group_record_axis(flat_trajectories, scan_size)
    train_metrics = self.ppo_chunked_minibatch_train(
        chunked_outputs, chunked_trajectories)
    return hidden_state, value_metrics, train_metrics

  def build_equal_minibatch_chunks(
      self,
      learner_outputs: list[list[tuple[int, int, LearnerOutputs]]],
      trajectories: list[Trajectory],
      profile: dict | None = None,
  ) -> tuple[LearnerOutputs, Trajectory, int] | None:
    """Stack uniform minibatches into [chunk, scan, ...] trees if possible."""
    scan_size = max(1, self._config.ppo.minibatch_scan_size)
    minibatch_size = self._config.ppo.minibatch_size
    output_records = []
    trajectory_records = []
    for output_slices, trajectory in zip(learner_outputs, trajectories):
      batch_size = trajectory_batch_size(trajectory)
      if not output_slices or output_slices[-1][1] != batch_size:
        return None
      if batch_size % minibatch_size:
        return None
      split_trajectory = split_trajectory_minibatches(
          trajectory, minibatch_size)
      if split_trajectory is None:
        return None
      for start, end, outputs in output_slices:
        if end - start != minibatch_size:
          return None
        output_records.append(outputs)
      trajectory_records.append(split_trajectory)

    record_count = len(output_records)
    if not output_records or record_count % scan_size:
      return None

    stack_start = time.perf_counter()
    output_stack_start = time.perf_counter()
    output_records = stack_trees(output_records)
    if profile is not None:
      profile['ppo_output_stack_s'] += time.perf_counter() - output_stack_start
    trajectory_stack_start = time.perf_counter()
    trajectory_records = concat_trees(trajectory_records, axis=0)
    if profile is not None:
      profile['ppo_trajectory_stack_s'] += time.perf_counter() - trajectory_stack_start
    chunked_outputs = group_record_axis(output_records, scan_size)
    chunked_trajectories = group_record_axis(trajectory_records, scan_size)
    block_until_ready((chunked_outputs, chunked_trajectories))
    if profile is not None:
      profile['ppo_chunk_stack_s'] += time.perf_counter() - stack_start
      profile['ppo_chunks'] += record_count // scan_size
      profile['ppo_chunked_minibatches'] += record_count
    return chunked_outputs, chunked_trajectories, record_count

  def unroll_trajectory_minibatched(
      self,
      trajectory: Trajectory,
      initial_state: LearnerState,
      *,
      train_value_function: bool,
      profile: dict | None = None,
  ) -> tuple[list[tuple[int, int, LearnerOutputs]], LearnerState]:
    if train_value_function:
      fast_result = self.unroll_trajectory_equal_minibatches(
          trajectory, initial_state, profile=profile)
      if fast_result is not None:
        return fast_result

    output_slices = []
    final_states = []
    value_grads_sum = None
    total_weight = 0
    batch_size = trajectory_batch_size(trajectory)
    for start, end in minibatch_ranges(
        batch_size, self._config.ppo.minibatch_size):
      weight = end - start
      total_weight += weight
      slice_start = time.perf_counter()
      trajectory_slice = slice_trajectory(trajectory, start, end)
      state_slice = slice_learner_state(initial_state, start, end)
      if profile is not None:
        profile['unroll_slice_s'] += time.perf_counter() - slice_start
      unroll_start = time.perf_counter()
      if train_value_function:
        value_grads, outputs, final_state = self.unroll_value_grads(
            trajectory_slice,
            state_slice,
        )
      else:
        value_grads = None
        outputs, final_state = self.unroll(
            trajectory_slice,
            state_slice,
            train_value_function=False,
        )
      if profile is not None:
        block_until_ready((value_grads, outputs, final_state))
        profile['teacher_value_unroll_s'] += time.perf_counter() - unroll_start
        profile['teacher_value_minibatches'] += 1
        profile['teacher_value_frames'] += (end - start) * trajectory.rewards.shape[0]
      if value_grads is not None:
        accum_start = time.perf_counter()
        weighted_grads = jax.tree.map(lambda g: g * weight, value_grads)
        if value_grads_sum is None:
          value_grads_sum = weighted_grads
        else:
          value_grads_sum = jax.tree.map(jnp.add, value_grads_sum, weighted_grads)
        if profile is not None:
          block_until_ready(value_grads_sum)
          profile['value_grad_accum_s'] += time.perf_counter() - accum_start
      if self._config.ppo.offload_minibatch_outputs:
        offload_start = time.perf_counter()
        # Keep cached learner outputs off device so rollout batch size does not
        # determine peak accelerator memory.
        outputs = jax.device_get(outputs)
        final_state = jax.device_get(final_state)
        if profile is not None:
          profile['minibatch_output_offload_s'] += time.perf_counter() - offload_start
      output_slices.append((start, end, outputs))
      final_states.append(final_state)

    concat_start = time.perf_counter()
    final_state = concat_learner_states(final_states)
    if profile is not None:
      block_until_ready(final_state)
      profile['learner_state_concat_s'] += time.perf_counter() - concat_start
    if value_grads_sum is not None:
      average_start = time.perf_counter()
      value_grads = jax.tree.map(lambda g: g / total_weight, value_grads_sum)
      if profile is not None:
        block_until_ready(value_grads)
        profile['value_grad_average_s'] += time.perf_counter() - average_start
      opt_start = time.perf_counter()
      self.value_optimizer.update(self.value_function, value_grads)
      if profile is not None:
        block_until_ready(self.value_function)
        profile['value_optimizer_s'] += time.perf_counter() - opt_start
    return output_slices, final_state

  def unroll_trajectory_equal_minibatches(
      self,
      trajectory: Trajectory,
      initial_state: LearnerState,
      profile: dict | None = None,
  ) -> tuple[list[tuple[int, int, LearnerOutputs]], LearnerState] | None:
    """Fast path for equal contiguous minibatches."""
    minibatch_size = self._config.ppo.minibatch_size
    trajectory_minibatches = split_trajectory_minibatches(
        trajectory, minibatch_size)
    state_minibatches = split_learner_state_minibatches(
        initial_state, minibatch_size)
    if trajectory_minibatches is None or state_minibatches is None:
      return None

    batch_size = trajectory_batch_size(trajectory)
    num_minibatches = batch_size // minibatch_size
    unroll_start = time.perf_counter()
    value_grads_sum, outputs, final_states = (
        self.jit_unroll_stacked_minibatch_value_grads(
            trajectory_minibatches, state_minibatches))
    if profile is not None:
      block_until_ready((value_grads_sum, outputs, final_states))
      profile['teacher_value_unroll_s'] += time.perf_counter() - unroll_start
      profile['teacher_value_minibatches'] += num_minibatches
      profile['teacher_value_chunks'] += 1
      profile['teacher_value_frames'] += batch_size * trajectory.rewards.shape[0]

    output_slices = [
        (
            i * minibatch_size,
            (i + 1) * minibatch_size,
            jax.tree.map(lambda t, i=i: t[i], outputs),
        )
        for i in range(num_minibatches)
    ]
    final_state = merge_learner_state_minibatches(final_states)

    average_start = time.perf_counter()
    value_grads = jax.tree.map(lambda g: g / num_minibatches, value_grads_sum)
    if profile is not None:
      block_until_ready(value_grads)
      profile['value_grad_average_s'] += time.perf_counter() - average_start
    opt_start = time.perf_counter()
    self.value_optimizer.update(self.value_function, value_grads)
    if profile is not None:
      block_until_ready(self.value_function)
      profile['value_optimizer_s'] += time.perf_counter() - opt_start
    return output_slices, final_state

  def unroll_ppo_batches_equal_minibatches(
      self,
      trajectories: list[Trajectory],
      initial_state: LearnerState,
      profile: dict | None = None,
  ) -> tuple[list[list[tuple[int, int, LearnerOutputs]]], LearnerState] | None:
    """Fast path for equal minibatches across PPO batches.

    Stacking PPO batches raises peak memory, so keep the original per-batch
    path for the single-batch case where this does not remove any dispatch.
    """
    if len(trajectories) < 2:
      return None
    minibatch_size = self._config.ppo.minibatch_size
    split_trajectories = []
    num_minibatches = None
    for trajectory in trajectories:
      split_trajectory = split_trajectory_minibatches(
          trajectory, minibatch_size)
      if split_trajectory is None:
        return None
      current_num_minibatches = (
          trajectory_batch_size(trajectory) // minibatch_size)
      if num_minibatches is None:
        num_minibatches = current_num_minibatches
      elif current_num_minibatches != num_minibatches:
        return None
      split_trajectories.append(split_trajectory)

    if num_minibatches is None:
      return None

    stack_start = time.perf_counter()
    stacked_trajectories = stack_trees(split_trajectories)
    if profile is not None:
      profile['teacher_value_batch_stack_s'] += time.perf_counter() - stack_start

    unroll_start = time.perf_counter()
    outputs, final_state = (
        self.jit_unroll_ppo_batch_stacked_minibatch_value_grads(
            stacked_trajectories, initial_state))
    if profile is not None:
      block_until_ready((outputs, final_state))
      profile['teacher_value_unroll_s'] += time.perf_counter() - unroll_start
      profile['teacher_value_minibatches'] += (
          len(trajectories) * num_minibatches)
      profile['teacher_value_chunks'] += 1
      profile['teacher_value_batches'] += len(trajectories)
      profile['teacher_value_frames'] += sum(
          trajectory_batch_size(t) * t.rewards.shape[0] for t in trajectories)

    output_slices = []
    for batch_index in range(len(trajectories)):
      batch_slices = []
      for minibatch_index in range(num_minibatches):
        batch_slices.append((
            minibatch_index * minibatch_size,
            (minibatch_index + 1) * minibatch_size,
            jax.tree.map(
                lambda t, b=batch_index, m=minibatch_index: t[b, m],
                outputs),
        ))
      output_slices.append(batch_slices)
    return output_slices, final_state

  def ppo_update_equal_minibatches(
      self,
      trajectories: list[Trajectory],
      initial_state: LearnerState,
      *,
      evaluate_post_update: bool,
      profile: dict | None = None,
  ) -> tuple[LearnerState, dict, dict, dict, int, bool] | None:
    """Fast path for a full equal-minibatch PPO update."""
    if len(trajectories) < 2:
      return None
    minibatch_size = self._config.ppo.minibatch_size
    scan_size = max(1, self._config.ppo.minibatch_scan_size)
    split_trajectories = []
    num_minibatches = None
    for trajectory in trajectories:
      split_trajectory = split_trajectory_minibatches(
          trajectory, minibatch_size)
      if split_trajectory is None:
        return None
      current_num_minibatches = trajectory_batch_size(trajectory) // minibatch_size
      if num_minibatches is None:
        num_minibatches = current_num_minibatches
      elif current_num_minibatches != num_minibatches:
        return None
      split_trajectories.append(split_trajectory)

    if num_minibatches is None:
      return None
    record_count = len(trajectories) * num_minibatches
    if record_count % scan_size:
      return None

    stack_start = time.perf_counter()
    stacked_trajectories = stack_trees(split_trajectories)
    if profile is not None:
      profile['equal_minibatch_stack_s'] += time.perf_counter() - stack_start

    update_start = time.perf_counter()
    if evaluate_post_update:
      hidden_state, value_metrics, epoch_metrics, final_metrics = (
          self.jit_ppo_equal_minibatch_update_train_and_eval(
              stacked_trajectories, initial_state))
      ready = (hidden_state, value_metrics, epoch_metrics, final_metrics)
      post_update_evaluated = True
    else:
      hidden_state, value_metrics, epoch_metrics = (
          self.jit_ppo_equal_minibatch_update_train(
              stacked_trajectories, initial_state))
      final_metrics = epoch_metrics
      ready = (hidden_state, value_metrics, epoch_metrics)
      post_update_evaluated = False
    if profile is not None:
      block_until_ready(ready)
      elapsed = time.perf_counter() - update_start
      profile['equal_minibatch_update_s'] += elapsed
      profile['teacher_value_batches'] += len(trajectories)
      profile['teacher_value_minibatches'] += record_count
      profile['teacher_value_frames'] += sum(
          trajectory_batch_size(t) * t.rewards.shape[0] for t in trajectories)
      profile['ppo_train_epochs'] += 1
      profile['ppo_grad_minibatches'] += record_count
      profile['ppo_grad_chunks'] += record_count // scan_size
      frame_count = (
          record_count
          * minibatch_size
          * (trajectories[0].rewards.shape[0] - self.policy.delay))
      profile['ppo_grad_frames'] += frame_count
      if post_update_evaluated:
        profile['ppo_eval_minibatches'] += record_count
        profile['ppo_eval_chunks'] += record_count // scan_size
        profile['ppo_eval_frames'] += frame_count
    return (
        hidden_state,
        value_metrics,
        epoch_metrics,
        final_metrics,
        record_count,
        post_update_evaluated)

  def ppo(
      self,
      trajectories: list[Trajectory],
      initial_state: LearnerState,
      step: int,
      jit: bool = True,
      recompute_rewards: bool = True,
      profile: bool = False,
  ) -> tp.Tuple[LearnerState, dict]:
    """Multi-epoch PPO update.

    Args:
      trajectories: List of trajectories (one per PPO batch).
      initial_state: Initial recurrent states for teacher and value function.
      num_epochs: Number of PPO epochs. Defaults to config value.

    Returns:
      Tuple of (new hidden state, metrics dict).
    """
    assert len(trajectories) == self._config.ppo.num_batches
    profile_timings = defaultdict(float) if profile else None

    if recompute_rewards:
      reward_start = time.perf_counter()
      trajectories = [
          update_rewards(t, self._config.reward) for t in trajectories
      ]
      if profile_timings is not None:
        profile_timings['reward_update_s'] += time.perf_counter() - reward_start

    use_minibatches = self._config.ppo.minibatch_size > 0
    if step < self._config.value_burnin_epochs:
      num_epochs = 0
    elif step < self._config.value_burnin_epochs + self._config.optimizer_burnin_epochs:
      num_epochs = 1
    else:
      num_epochs = self._config.ppo.num_epochs

    if use_minibatches:
      ppo_epoch = self.ppo_epoch_minibatched
    else:
      ppo_epoch = self.jit_ppo_epoch if jit else self.ppo_epoch

    should_eval_post_update = (
        self._config.ppo.post_update_eval_interval > 0
        and step % self._config.ppo.post_update_eval_interval == 0)
    should_checkpoint_for_revert = (
        should_eval_post_update
        and self._config.ppo.revert_on_post_update_actor_kl)
    if should_checkpoint_for_revert:
      checkpoint = dict(
          policy=jax_utils.get_module_state(self.policy, to_numpy=False),
          policy_optimizer=jax_utils.get_module_state(
              self.policy_optimizer, to_numpy=False),
      )
    else:
      checkpoint = None

    per_epoch_metrics = []
    ppo_already_ran = False
    post_update_evaluated = False

    # Unroll teacher + value function, training value function.
    hidden_state = initial_state
    unroll_total_start = time.perf_counter()
    if use_minibatches:
      fast_update = None
      if num_epochs == 1:
        fast_update = self.ppo_update_equal_minibatches(
            trajectories,
            hidden_state,
            evaluate_post_update=should_eval_post_update,
            profile=profile_timings)
      if fast_update is not None:
        (
            hidden_state,
            value_metrics,
            epoch_metrics,
            final_metrics,
            _,
            post_update_evaluated,
        ) = fast_update
        per_epoch_metrics = [epoch_metrics]
        ppo_already_ran = True
      else:
        learner_outputs: list[list[tuple[int, int, LearnerOutputs]]] = []
        fast_unroll = self.unroll_ppo_batches_equal_minibatches(
            trajectories, hidden_state, profile=profile_timings)
        if fast_unroll is not None:
          learner_outputs, hidden_state = fast_unroll
        else:
          for trajectory in trajectories:
            output_slices, hidden_state = self.unroll_trajectory_minibatched(
                trajectory, hidden_state, train_value_function=True,
                profile=profile_timings)
            learner_outputs.append(output_slices)
        if profile_timings is not None:
          profile_timings['teacher_value_unroll_total_s'] += (
              time.perf_counter() - unroll_total_start)
        value_metrics_start = time.perf_counter()
        value_metrics_list = [
            outputs.value.metrics
            for output_slices in learner_outputs
            for _, _, outputs in output_slices
        ]
        value_metrics = summarize_nested_means(value_metrics_list)
        if profile_timings is not None:
          profile_timings['value_metrics_s'] += time.perf_counter() - value_metrics_start
    else:
      learner_outputs: list[LearnerOutputs] = []
      for trajectory in trajectories:
        outputs, hidden_state = self.unroll(
            trajectory, hidden_state, train_value_function=True)
        if profile_timings is not None:
          block_until_ready((outputs, hidden_state))
        learner_outputs.append(outputs)
      if profile_timings is not None:
        profile_timings['teacher_value_unroll_total_s'] += (
            time.perf_counter() - unroll_total_start)

      # Collect value function metrics.
      value_metrics_start = time.perf_counter()
      value_metrics_list = [o.value.metrics for o in learner_outputs]
      value_metrics = summarize_nested_means(value_metrics_list)
      if profile_timings is not None:
        profile_timings['value_metrics_s'] += time.perf_counter() - value_metrics_start

    # PPO epochs with gradient updates.
    chunked_minibatches = None
    if (
        not ppo_already_ran
        and use_minibatches
        and num_epochs == 1):
      chunked_minibatches = self.build_equal_minibatch_chunks(
          learner_outputs, trajectories, profile=profile_timings)

    if ppo_already_ran:
      pass
    elif chunked_minibatches is not None:
      fuse_start = time.perf_counter()
      stacked_outputs, stacked_trajectories, minibatch_count = chunked_minibatches
      if should_eval_post_update:
        epoch_metrics, final_metrics = (
            self.jit_ppo_chunked_minibatch_train_and_eval(
                stacked_outputs, stacked_trajectories))
        post_update_evaluated = True
        ready = (epoch_metrics, final_metrics)
      else:
        epoch_metrics = self.jit_ppo_chunked_minibatch_train(
            stacked_outputs, stacked_trajectories)
        final_metrics = epoch_metrics
        ready = epoch_metrics
      if profile_timings is not None:
        block_until_ready(ready)
        elapsed = time.perf_counter() - fuse_start
        if post_update_evaluated:
          profile_timings['ppo_chunked_train_eval_s'] += elapsed
        else:
          profile_timings['ppo_chunked_train_s'] += elapsed
        profile_timings['ppo_train_epoch_s'] += elapsed
        profile_timings['final_eval_total_s'] += 0.0
        profile_timings['ppo_train_epochs'] += 1
        profile_timings['ppo_grad_minibatches'] += minibatch_count
        profile_timings['ppo_grad_chunks'] += stacked_trajectories.rewards.shape[0]
        frame_count = (
            minibatch_count
            * self._config.ppo.minibatch_size
            * (trajectories[0].rewards.shape[0] - self.policy.delay))
        profile_timings['ppo_grad_frames'] += frame_count
        if post_update_evaluated:
          profile_timings['ppo_eval_minibatches'] += minibatch_count
          profile_timings['ppo_eval_chunks'] += stacked_trajectories.rewards.shape[0]
          profile_timings['ppo_eval_frames'] += frame_count
      per_epoch_metrics.append(epoch_metrics)
    else:
      for _ in range(num_epochs):
        epoch_start = time.perf_counter()
        if use_minibatches:
          epoch_metrics = ppo_epoch(
              learner_outputs, trajectories, train=True, profile=profile_timings)
        else:
          epoch_metrics = ppo_epoch(learner_outputs, trajectories, train=True)
        if profile_timings is not None:
          block_until_ready(epoch_metrics)
          profile_timings['ppo_train_epoch_s'] += time.perf_counter() - epoch_start
          profile_timings['ppo_train_epochs'] += 1
        per_epoch_metrics.append(epoch_metrics)

      if should_eval_post_update or not per_epoch_metrics:
        # Final eval epoch (no gradient update) to measure exact post-update KL.
        eval_start = time.perf_counter()
        if use_minibatches:
          final_metrics = ppo_epoch(
              learner_outputs, trajectories, train=False,
              profile=profile_timings)
        else:
          final_metrics = ppo_epoch(learner_outputs, trajectories, train=False)
        if profile_timings is not None:
          block_until_ready(final_metrics)
          profile_timings['final_eval_total_s'] += time.perf_counter() - eval_start
        post_update_evaluated = True
      else:
        final_metrics = per_epoch_metrics[-1]

    # Optional legacy rollback when exact post-update actor KL is evaluated.
    reverted = False
    if (
        post_update_evaluated
        and self._config.ppo.revert_on_post_update_actor_kl
        and final_metrics['actor_kl']['mean'] > self._config.ppo.max_mean_actor_kl):
      assert checkpoint is not None
      jax_utils.set_module_state(self.policy, checkpoint['policy'])
      jax_utils.set_module_state(
          self.policy_optimizer, checkpoint['policy_optimizer'])
      reverted = True

    metrics = dict(
        ppo_step={str(i): m for i, m in enumerate(per_epoch_metrics)},
        post_update=final_metrics,
        value=value_metrics,
        reverted=reverted,
        post_update_evaluated=post_update_evaluated,
    )
    if profile_timings is not None:
      metrics['profile_sec'] = dict(profile_timings)

    return hidden_state, metrics

  def get_state(self) -> dict:
    return jax_utils.get_module_state(self)

  def restore_from_imitation(self, state_dict: dict, param_dtype: str = 'float32'):
    # Legacy TF checkpoints store policy/value leaves as tuples and optimizer
    # state under an incompatible `optimizers` tuple. Restore the model weights
    # and keep the freshly initialized JAX optimizer state in that case.
    state_dict = dict(state_dict)
    state_dict.pop('step', None)
    if 'policy' in state_dict and not isinstance(state_dict['policy'], dict):
      policy_state = tf_checkpoint.convert_policy_params(
          self.policy, state_dict.pop('policy'))
      policy_state = tf_checkpoint.cast_floating_state(policy_state, param_dtype)
      jax_utils.set_module_state(self.policy, policy_state)
      tf_checkpoint.set_compute_dtype(self.policy, param_dtype)
    if (
        'value_function' in state_dict
        and not isinstance(state_dict['value_function'], dict)
    ):
      value_state = tf_checkpoint.convert_value_function_params(
          self.value_function, state_dict.pop('value_function'))
      value_state = tf_checkpoint.cast_floating_state(value_state, param_dtype)
      jax_utils.set_module_state(self.value_function, value_state)
      tf_checkpoint.set_compute_dtype(self.value_function, param_dtype)
    if 'optimizers' in state_dict and 'policy_optimizer' not in state_dict:
      state_dict.pop('optimizers')
    if state_dict:
      jax_utils.set_module_state(self, state_dict)
