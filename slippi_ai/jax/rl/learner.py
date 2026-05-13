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
from slippi_ai.jax.policies import Policy, UnrollOutputs
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
  teacher: UnrollOutputs
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


def update_rewards(
    trajectory: Trajectory,
    reward_config: reward_lib.RewardConfig,
) -> Trajectory:
  rewards = reward_lib.compute_rewards(
      trajectory.states, **dataclasses.asdict(reward_config))
  rewards = np.where(trajectory.is_resetting[1:], 0.0, rewards)
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

  def _compute_kl(self, logits_p, logits_q) -> Array:
    """Computes total KL(P||Q) summed over all controller components."""
    kls = self._controller_embedding.map(
        lambda e, lp, lq: e.kl_divergence(lp, lq),
        logits_p, logits_q)
    return self._sum_leaves(self._controller_embedding, kls)

  def _compute_entropy(self, logits) -> Array:
    """Computes total entropy H(P) summed over all controller components."""
    entropies = self._controller_embedding.map(
        lambda e, l: e.entropy(l), logits)
    return self._sum_leaves(self._controller_embedding, entropies)

  def _get_log_prob(self, logits, action) -> Array:
    """Computes log P(action | logits) summed over all controller components."""
    distances = self._controller_embedding.map(
        lambda e, l, a: e.distance(l, a), logits, action)
    return -self._sum_leaves(self._controller_embedding, distances)

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

    # Actor (old policy) logits and log probs for steps [D+1, U+1].
    actor_outputs = utils.map_single_structure(
        lambda t: t[1 + delay:], trajectory.actions)
    actor_logits = actor_outputs.logits
    actor_log_probs = jax.lax.stop_gradient(
        self._get_log_prob(actor_logits, actor_outputs.controller_state))

    policy_outputs = policy.unroll(policy_frames, trajectory.initial_state)
    new_logits = policy_outputs.distances.logits
    new_log_probs = policy_outputs.log_probs

    # Teacher logits: [D, U+D] -> truncate last D -> [D, U].
    # Note: no stop_gradient needed since teacher has no trainable variables.
    teacher_logits = jax.tree.map(remove_last, outputs.teacher.distances.logits)
    # KL divergences are computed over full output distribution, not just
    # sampled action. Forward KL to teacher incentivizes refining human
    # actions over covering all of them.
    teacher_kl = self._compute_kl(new_logits, teacher_logits)
    # KL of old actor from new policy: used for monitoring / reverting bad
    # updates.
    actor_kl = self._compute_kl(actor_logits, new_logits)
    reverse_teacher_kl = self._compute_kl(teacher_logits, new_logits)
    entropy = self._compute_entropy(new_logits)

    # PPO clipped objective.
    log_rhos = new_log_probs - actor_log_probs
    rhos = jnp.exp(log_rhos)

    eps = self._config.ppo.epsilon
    clipped_log_rhos = jnp.clip(log_rhos, -eps, eps)
    clipped_rhos = jnp.exp(clipped_log_rhos)

    ppo_objective = jnp.minimum(rhos * advantages, clipped_rhos * advantages)

    loss = jnp.mean(
        - self._config.policy_gradient_weight * ppo_objective
        + self._config.ppo.beta * actor_kl
        + self._config.kl_teacher_weight * teacher_kl
        + self._config.reverse_kl_teacher_weight * reverse_teacher_kl
        - self._config.entropy_weight * entropy
    )

    metrics = dict(
        total_loss=loss,
        ppo_objective=ppo_objective,
        teacher_kl=teacher_kl,
        entropy=entropy,
        actor_kl=actor_kl,
        reverse_teacher_kl=reverse_teacher_kl,
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
    teacher_outputs = self.teacher.unroll(
        teacher_frames, initial_state.teacher)

    # Run value function (with or without gradient update).
    value_frames = get_frames(trajectory)

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
    teacher_outputs = self.teacher.unroll(
        teacher_frames, initial_state.teacher)

    value_frames = get_frames(trajectory)

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
      loss, metrics = self._policy_loss_and_metrics(policy, outputs, trajectory)
      return loss, summarize_policy_metrics_jax(metrics)

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
    return summarize_policy_metrics_jax(
        self.ppo_metrics(outputs, trajectory))

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

  def unroll_trajectory_minibatched(
      self,
      trajectory: Trajectory,
      initial_state: LearnerState,
      *,
      train_value_function: bool,
      profile: dict | None = None,
  ) -> tuple[list[tuple[int, int, LearnerOutputs]], LearnerState]:
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

  def ppo(
      self,
      trajectories: list[Trajectory],
      initial_state: LearnerState,
      step: int,
      jit: bool = True,
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

    # Compute rewards from game states.
    reward_start = time.perf_counter()
    trajectories = [
        update_rewards(t, self._config.reward) for t in trajectories
    ]
    if profile_timings is not None:
      profile_timings['reward_update_s'] += time.perf_counter() - reward_start

    use_minibatches = self._config.ppo.minibatch_size > 0
    # Unroll teacher + value function, training value function.
    hidden_state = initial_state
    unroll_total_start = time.perf_counter()
    if use_minibatches:
      learner_outputs: list[list[tuple[int, int, LearnerOutputs]]] = []
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

    # Checkpoint policy update state for potential actor-KL reversion.
    checkpoint = dict(
        policy=jax_utils.get_module_state(self.policy, to_numpy=False),
        policy_optimizer=jax_utils.get_module_state(
            self.policy_optimizer, to_numpy=False),
    )

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

    # PPO epochs with gradient updates.
    per_epoch_metrics = []
    if (
        use_minibatches
        and num_epochs == 1
        and len(trajectories) == 1
        and len(learner_outputs) == 1
        and len(learner_outputs[0]) <= self._config.ppo.minibatch_scan_size
        and len({end - start for start, end, _ in learner_outputs[0]}) == 1
    ):
      fuse_start = time.perf_counter()
      output_slices = learner_outputs[0]
      stacked_outputs = jax.tree.map(
          lambda *xs: jnp.stack(xs),
          *[outputs for _, _, outputs in output_slices])
      stacked_trajectories = jax.tree.map(
          lambda *xs: jnp.stack(xs),
          *[
              slice_trajectory(trajectories[0], start, end)
              for start, end, _ in output_slices
          ])
      epoch_metrics, final_metrics = (
          self.jit_ppo_stacked_minibatch_train_and_eval(
              stacked_outputs, stacked_trajectories))
      if profile_timings is not None:
        block_until_ready((epoch_metrics, final_metrics))
        elapsed = time.perf_counter() - fuse_start
        profile_timings['ppo_fused_train_eval_s'] += elapsed
        profile_timings['ppo_train_epoch_s'] += elapsed
        profile_timings['final_eval_total_s'] += 0.0
        profile_timings['ppo_train_epochs'] += 1
        profile_timings['ppo_grad_minibatches'] += len(output_slices)
        profile_timings['ppo_eval_minibatches'] += len(output_slices)
        profile_timings['ppo_grad_chunks'] += 1
        profile_timings['ppo_eval_chunks'] += 1
        frame_count = sum(
            (end - start) * (trajectories[0].rewards.shape[0] - self.policy.delay)
            for start, end, _ in output_slices)
        profile_timings['ppo_grad_frames'] += frame_count
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

      # Final eval epoch (no gradient update) to measure post-update KL.
      eval_start = time.perf_counter()
      if use_minibatches:
        final_metrics = ppo_epoch(
            learner_outputs, trajectories, train=False, profile=profile_timings)
      else:
        final_metrics = ppo_epoch(learner_outputs, trajectories, train=False)
      if profile_timings is not None:
        block_until_ready(final_metrics)
        profile_timings['final_eval_total_s'] += time.perf_counter() - eval_start
    per_epoch_metrics.append(final_metrics)

    # Revert if the policy moved too far from the actor.
    reverted = False
    if final_metrics['actor_kl']['mean'] > self._config.ppo.max_mean_actor_kl:
      jax_utils.set_module_state(self.policy, checkpoint['policy'])
      jax_utils.set_module_state(
          self.policy_optimizer, checkpoint['policy_optimizer'])
      reverted = True

    metrics = dict(
        ppo_step={str(i): m for i, m in enumerate(per_epoch_metrics)},
        post_update=final_metrics,
        value=value_metrics,
        reverted=reverted,
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
