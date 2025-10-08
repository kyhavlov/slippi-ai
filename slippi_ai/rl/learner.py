import dataclasses
import logging
import typing as tp

import numpy as np
import sonnet as snt
import tensorflow as tf

from slippi_ai.data import Frames
from slippi_ai.embed import StateAction
from slippi_ai.policies import Policy, UnrollOutputs, SampleOutputs
from slippi_ai.evaluators import Trajectory
from slippi_ai.networks import RecurrentState
from slippi_ai.controller_heads import ControllerType
from slippi_ai import value_function as vf_lib
from slippi_ai import tf_utils, utils, reward as reward_lib

field = lambda f: dataclasses.field(default_factory=f)

@dataclasses.dataclass
class PPOConfig:
  num_epochs: int = 1
  num_batches: int = 1
  epsilon: float = 1e-2
  beta: float = 0
  minibatched: bool = False
  # target_kl: float = 1e-3
  max_mean_actor_kl: float = 1e-4

@dataclasses.dataclass
class LearnerConfig:
  # TODO: unify this with the imitation config?
  learning_rate: float = 1e-4
  compile: bool = True
  jit_compile: bool = False
  policy_gradient_weight: float = 1
  kl_teacher_weight: float = 1e-1
  reverse_kl_teacher_weight: float = 0
  entropy_weight: float = 0
  value_cost: float = 0.5
  reward_halflife: float = 4  # measured in seconds
  discount_on_death: tp.Optional[float] = None
  reward: reward_lib.RewardConfig = field(reward_lib.RewardConfig)
  ppo: PPOConfig = field(PPOConfig)

class LearnerState(tp.NamedTuple):
  teacher: RecurrentState
  value_function: RecurrentState

class LearnerOutputs(tp.NamedTuple):
  teacher: UnrollOutputs
  value: vf_lib.ValueOutputs

def get_frames(trajectory: Trajectory) -> Frames:
  """Gives frames with actions taken."""
  state_action = StateAction(
      state=trajectory.states,
      action=trajectory.actions.controller_state,
      name=trajectory.name,
  )
  return Frames(state_action, trajectory.is_resetting, trajectory.rewards)

def get_delayed_frames(trajectory: Trajectory) -> Frames:
  """Gives frames with delayed actions, for policy unroll."""
  delay = len(trajectory.delayed_actions)

  delayed_actions = [
      sample_output.controller_state
      for sample_output in trajectory.delayed_actions
  ]
  # Add time dimension
  delayed_actions = tf.nest.map_structure(
      lambda t: tf.expand_dims(t, 0), delayed_actions)

  # Concatenate everything together.
  actions = tf.nest.map_structure(
      lambda *ts: tf.concat(ts, 0),
      trajectory.actions.controller_state, *delayed_actions)
  # Chop off the beginning _after_ concatenation to handle the case where
  # trajectory length < delay, which happens during initialization.
  actions = tf.nest.map_structure(lambda t: t[delay:], actions)

  state_action = StateAction(
      state=trajectory.states,
      action=actions,
      name=trajectory.name,
  )

  # Trajectory.rewards is technically wrong, but it's fine because
  # we don't use the policy's builtin value function anyways.
  return Frames(state_action, trajectory.is_resetting, trajectory.rewards)

def combine_grads(x: tp.Optional[tf.Tensor], y: tp.Optional[tf.Tensor]):
  if x is None or y is None:
    return None
  return x + y

def update_rewards(
    trajectory: Trajectory,
    reward_config: reward_lib.RewardConfig,
) -> Trajectory:
  rewards = reward_lib.compute_rewards(
      trajectory.states, **dataclasses.asdict(reward_config))
  return trajectory._replace(rewards=rewards)

class Learner:
  """Implements A2C."""

  def __init__(
      self,
      config: LearnerConfig,
      policy: Policy,
      teacher: Policy,
      learning_rate: tp.Optional[tf.Variable] = None,
      value_function: tp.Optional[vf_lib.ValueFunction] = None,
  ) -> None:
    self._config = config
    self._policy = policy
    self._teacher = teacher
    self._use_separate_vf = value_function is not None
    self._value_function = value_function or vf_lib.FakeValueFunction()

    if learning_rate is None:
      learning_rate = tf.Variable(config.learning_rate, trainable=False)
    self.learning_rate = learning_rate
    self.policy_optimizer = snt.optimizers.Adam(learning_rate)
    self.value_optimizer = snt.optimizers.Adam(learning_rate)

    self.discount = 0.5 ** (1 / (config.reward_halflife * 60))

    if config.jit_compile:
      logging.warning('jit_compile may lead to instability')

    if config.compile:
      maybe_compile = tf.function(
          jit_compile=config.jit_compile, autograph=False)
    else:
      maybe_compile = lambda f: f

    self.compiled_unroll = maybe_compile(self.unroll)
    self.compiled_ppo_grads = maybe_compile(self.ppo_grads)
    self.compiled_ppo_grads_acc = maybe_compile(self.ppo_grads_acc)
    self.compiled_ppo = maybe_compile(self.ppo)

  def _slice_time_major(
      self,
      value: tp.Any,
      mask_size: tp.Optional[int],
      indices,
  ) -> tp.Any:

    def slicer(arr):
      if tf.is_tensor(arr):
        tensor = tf.convert_to_tensor(arr)
        rank = tf.rank(tensor)

        def gather_axis1():
          return tf.gather(tensor, indices, axis=1)

        def gather_axis0():
          return tf.gather(tensor, indices, axis=0)

        return tf.cond(
            tf.equal(rank, 0),
            lambda: tensor,
            lambda: tf.cond(tf.equal(rank, 1), gather_axis0, gather_axis1))

      arr_np = np.asarray(arr)
      if arr_np.ndim == 0:
        return arr_np
      if arr_np.ndim == 1:
        if mask_size is not None and arr_np.shape[0] != mask_size:
          raise ValueError('Mask length does not match array length.')
        return np.take(arr_np, indices, axis=0)
      if mask_size is not None and arr_np.shape[1] != mask_size:
        raise ValueError('Mask length does not match environment axis.')
      return np.take(arr_np, indices, axis=1)

    return utils.map_single_structure(slicer, value)

  def _slice_batch_major(
      self,
      value: tp.Any,
      mask_size: tp.Optional[int],
      indices,
  ) -> tp.Any:

    def slicer(arr):
      if tf.is_tensor(arr):
        tensor = tf.convert_to_tensor(arr)
        rank = tf.rank(tensor)

        def gather_axis0():
          return tf.gather(tensor, indices, axis=0)

        return tf.cond(tf.equal(rank, 0), lambda: tensor, gather_axis0)

      arr_np = np.asarray(arr)
      if arr_np.ndim == 0:
        return arr_np
      if mask_size is not None and arr_np.shape[0] != mask_size:
        raise ValueError('Mask length does not match batch axis.')
      return np.take(arr_np, indices, axis=0)

    return utils.map_single_structure(slicer, value)

  def _mask_trajectory(self, trajectory: Trajectory) -> Trajectory:
    mask = trajectory.active_mask
    if tf.is_tensor(mask):
      mask_tensor = tf.reshape(tf.cast(mask, tf.bool), [-1])
      active_total = tf.reduce_sum(tf.cast(mask_tensor, tf.int32))
      with tf.control_dependencies([
          tf.debugging.assert_positive(
              active_total, message='Active mask must contain at least one entry.'),
      ]):
        mask_tensor = tf.identity(mask_tensor)
      indices = tf.cast(tf.reshape(tf.where(mask_tensor), [-1]), tf.int32)
      mask_size = None

      states = self._slice_time_major(trajectory.states, mask_size, indices)
      name = self._slice_time_major(trajectory.name, mask_size, indices)
      actions = SampleOutputs(
          controller_state=self._slice_time_major(
              trajectory.actions.controller_state, mask_size, indices),
          logits=self._slice_time_major(
              trajectory.actions.logits, mask_size, indices),
      )
      rewards = self._slice_time_major(trajectory.rewards, mask_size, indices)
      is_resetting = self._slice_time_major(
          trajectory.is_resetting, mask_size, indices)
      initial_state = self._slice_batch_major(
          trajectory.initial_state, mask_size, indices)
      delayed_actions = [
          SampleOutputs(
              controller_state=self._slice_batch_major(
                  sample.controller_state, mask_size, indices),
              logits=self._slice_batch_major(sample.logits, mask_size, indices),
          )
          for sample in trajectory.delayed_actions
      ]
      active_mask = tf.ones_like(indices, dtype=tf.bool)

    else:
      mask_np = np.asarray(mask, dtype=np.bool_)
      if mask_np.ndim != 1:
        raise ValueError('Active mask must be one-dimensional.')
      mask_size = mask_np.size
      if mask_size == 0:
        raise ValueError('Active mask must contain at least one entry.')
      indices = np.nonzero(mask_np)[0]
      states = self._slice_time_major(trajectory.states, mask_size, indices)
      name = self._slice_time_major(trajectory.name, mask_size, indices)
      actions = SampleOutputs(
          controller_state=self._slice_time_major(
              trajectory.actions.controller_state, mask_size, indices),
          logits=self._slice_time_major(
              trajectory.actions.logits, mask_size, indices),
      )
      rewards = self._slice_time_major(trajectory.rewards, mask_size, indices)
      is_resetting = self._slice_time_major(
          trajectory.is_resetting, mask_size, indices)
      initial_state = self._slice_batch_major(
          trajectory.initial_state, mask_size, indices)
      delayed_actions = [
          SampleOutputs(
              controller_state=self._slice_batch_major(
                  sample.controller_state, mask_size, indices),
              logits=self._slice_batch_major(sample.logits, mask_size, indices),
          )
          for sample in trajectory.delayed_actions
      ]
      active_mask = np.ones(indices.size, dtype=np.bool_)

    return trajectory._replace(
        states=states,
        name=name,
        actions=actions,
        rewards=rewards,
        is_resetting=is_resetting,
        initial_state=initial_state,
        delayed_actions=delayed_actions,
        active_mask=active_mask,
    )

  def initial_state(self, batch_size: int) -> LearnerState:
    return LearnerState(
        teacher=self._teacher.initial_state(batch_size),
        value_function=self._value_function.initial_state(batch_size),
    )

  def policy_variables(self) -> tp.Sequence[tf.Variable]:
    return self._policy.variables

  def _get_distribution(self, logits: ControllerType):
    """Returns a Controller-shaped structure of distributions."""
    # TODO: return an actual JointDistribution instead?
    return self._policy.controller_embedding.map(
        lambda e, t: e.distribution(t), logits)

  def _compute_kl(self, dist1: ControllerType, dist2: ControllerType):
    kls = self._policy.controller_embedding.map(
        lambda _, d1, d2: d1.kl_divergence(d2),
        dist1, dist2)
    return tf.add_n(list(self._policy.controller_embedding.flatten(kls)))

  def _get_log_prob(self, logits: ControllerType, action: ControllerType):
    controller_embedding = self._policy.controller_embedding
    distances = controller_embedding.map(
        lambda e, t, a: e.distance(t, a), logits, action)
    return - tf.add_n(list(controller_embedding.flatten(distances)))

  def _compute_entropy(self, dist: ControllerType):
    controller_embedding = self._policy.controller_embedding
    entropies = controller_embedding.map(lambda _, d: d.entropy(), dist)
    return tf.add_n(list(controller_embedding.flatten(entropies)))

  def _init_mode_accumulators(self) -> dict[str, dict[str, float]]:
    def make_bucket():
      return dict(
          reward_sum=0.0,
          reward_sq_sum=0.0,
          reward_count=0,
          actor_kl_sum=0.0,
          actor_kl_count=0,
          teacher_kl_sum=0.0,
          teacher_kl_count=0,
          uev_sum=0.0,
          uev_count=0,
          columns=0,
      )

    return {
        'singles': make_bucket(),
        'doubles': make_bucket(),
    }

  def _accumulate_mode_metrics(
      self,
      buckets: dict[str, dict[str, float]],
      trajectory: Trajectory,
      policy_metrics: dict,
      value_metrics: dict,
  ) -> None:
    is_teams = np.asarray(trajectory.states.is_teams[0]).astype(bool)
    masks = {
        'singles': ~is_teams,
        'doubles': is_teams,
    }

    rewards = np.asarray(trajectory.rewards)
    actor_kl = np.asarray(policy_metrics['actor_kl'])
    teacher_kl = np.asarray(policy_metrics['teacher_kl'])
    uev = np.asarray(value_metrics.get('uev', ()))

    for mode, mask in masks.items():
      indices = np.nonzero(mask)[0]
      if indices.size == 0:
        continue

      bucket = buckets[mode]
      bucket['columns'] += int(indices.size)

      mode_rewards = rewards[:, indices].reshape(-1)
      bucket['reward_sum'] += float(mode_rewards.sum())
      bucket['reward_sq_sum'] += float(np.square(mode_rewards).sum())
      bucket['reward_count'] += int(mode_rewards.size)

      mode_actor_kl = actor_kl[:, indices].reshape(-1)
      bucket['actor_kl_sum'] += float(mode_actor_kl.sum())
      bucket['actor_kl_count'] += int(mode_actor_kl.size)

      mode_teacher_kl = teacher_kl[:, indices].reshape(-1)
      bucket['teacher_kl_sum'] += float(mode_teacher_kl.sum())
      bucket['teacher_kl_count'] += int(mode_teacher_kl.size)

      if uev.size:
        mode_uev = uev[:, indices].reshape(-1)
        bucket['uev_sum'] += float(mode_uev.sum())
        bucket['uev_count'] += int(mode_uev.size)

  def _finalize_mode_metrics(
      self,
      buckets: dict[str, dict[str, float]],
  ) -> dict[str, tp.Any]:
    result = {}
    total_columns = 0

    def finalize_bucket(bucket: dict[str, float]) -> dict[str, tp.Any]:
      stats = {}
      count = bucket['reward_count']
      if count:
        mean = bucket['reward_sum'] / count
        variance = max(bucket['reward_sq_sum'] / count - mean ** 2, 0.0)
        stats['reward'] = dict(
            mean=float(mean),
            std=float(np.sqrt(variance)),
            count=int(count),
        )
      else:
        stats['reward'] = dict(mean=0.0, std=0.0, count=0)

      for key in [('actor_kl', 'actor_kl_sum', 'actor_kl_count'),
                  ('teacher_kl', 'teacher_kl_sum', 'teacher_kl_count')]:
        metric, total_key, count_key = key
        denom = bucket[count_key]
        value = bucket[total_key] / denom if denom else 0.0
        stats[metric] = dict(mean=float(value), count=int(denom))

      denom = bucket['uev_count']
      value = bucket['uev_sum'] / denom if denom else 0.0
      stats['uev'] = dict(mean=float(value), count=int(denom))

      stats['active_columns'] = int(bucket['columns'])
      return stats

    for mode, bucket in buckets.items():
      stats = finalize_bucket(bucket)
      result[mode] = stats
      total_columns += stats['active_columns']

    singles_columns = result['singles']['active_columns']
    ratio = singles_columns / total_columns if total_columns else 0.0
    result['totals'] = dict(
        active_columns=int(total_columns),
        singles_ratio=float(ratio),
    )
    return result

  def _match_state_dtypes(self, states, template_states):
    def cast_value(value, template):
      if tf.is_tensor(value):
        if value.dtype.is_integer and value.dtype not in (tf.int32, tf.uint8):
          return tf.cast(value, tf.int32)
        return value
      if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.integer):
        if value.dtype not in (np.int32, np.uint8):
          return value.astype(np.int32)
      return value

    return tf.nest.map_structure(cast_value, states, template_states)

  def unroll(
      self,
      trajectory: Trajectory,
      initial_state: LearnerState,
      train_value_function: bool = False,
  ) -> tp.Tuple[LearnerOutputs, LearnerState]:
    assert len(trajectory.delayed_actions) == self._policy.delay

    original_states = trajectory.states
    trajectory = self._mask_trajectory(trajectory)
    trajectory = trajectory._replace(
        states=self._match_state_dtypes(trajectory.states, original_states))

    teacher_outputs = self._teacher.unroll(
        # TODO: use the teacher's name instead?
        frames=get_delayed_frames(trajectory),
        initial_state=initial_state.teacher,
        discount=self.discount,
    )

    with tf.GradientTape() as tape:
      value_ouputs, final_value_state = self._value_function.loss(
          frames=get_frames(trajectory),
          initial_state=initial_state.value_function,
          discount=self.discount,
          discount_on_death=self._config.discount_on_death,
      )
      if train_value_function:
        grads = tape.gradient(value_ouputs.loss, self._value_vars)
        self.value_optimizer.apply(grads, self._value_vars)

    final_state = LearnerState(
        teacher=teacher_outputs.final_state,
        value_function=final_value_state,
    )

    outputs = LearnerOutputs(
        teacher=teacher_outputs,
        value=value_ouputs,
    )

    return outputs, final_state

  def ppo_grads(self, outputs: LearnerOutputs, trajectory: Trajectory):
    # Value function outputs are for [0, U] while policy outputs are for
    # [D, U+D]. This means we can only train on steps [D, U].

    original_states = trajectory.states
    trajectory = self._mask_trajectory(trajectory)
    trajectory = trajectory._replace(
        states=self._match_state_dtypes(trajectory.states, original_states))

    delay = self._policy.delay  # "D"
    remove_first = lambda t: t[delay:]

    def remove_last(arr):
      if delay == 0:
        return arr
      if tf.is_tensor(arr):
        time_len = tf.shape(arr, out_type=tf.int32)[0]
        return arr[:time_len - delay]
      arr_np = np.asarray(arr)
      return arr_np[:arr_np.shape[0] - delay]

    advantages = outputs.value.advantages[delay:]  # [0, U] -> [D, U]

    # Teacher logits are between [D, U+D]; truncate to [D, U]
    # Note: no stop_gradient needed as the teacher's variables aren't trainable.
    teacher_logits = tf.nest.map_structure(
        remove_last, outputs.teacher.distances.logits)
    teacher_distribution = self._get_distribution(teacher_logits)

    del outputs

    # We also have to reset the policy state; this isn't visible to
    # the actor as it happens inside the agent (eval_lib.BasicAgent).
    is_resetting = trajectory.is_resetting[0]  # [B]
    batch_size = tf.shape(is_resetting, out_type=tf.int32)[0]
    initial_policy_state = tf.nest.map_structure(
        lambda x, y: tf_utils.where(is_resetting, x, y),
        self._policy.initial_state(batch_size), trajectory.initial_state)

    policy_frames = Frames(
        state_action=StateAction(
            state=tf.nest.map_structure(remove_last, trajectory.states),
            action=tf.nest.map_structure(
                remove_first, trajectory.actions.controller_state),
            name=remove_last(trajectory.name),
        ),
        is_resetting=remove_last(trajectory.is_resetting),
        reward=remove_first(trajectory.rewards),
    )

    # For the actor, also drop the first action which precedes the first frame.
    actions: SampleOutputs = tf.nest.map_structure(
        lambda t: t[1+delay:], trajectory.actions)
    actor_distribution = self._get_distribution(actions.logits)
    actor_log_probs = self._get_log_prob(
        actions.logits, actions.controller_state)
    del trajectory

    with tf.GradientTape() as tape:
      policy_outputs = self._policy.unroll(
          frames=policy_frames,
          initial_state=initial_policy_state,
          discount=self.discount,
      )
      policy_distribution = self._get_distribution(policy_outputs.distances.logits)
      entropy = self._compute_entropy(policy_distribution)

      # We take the "forward" KL to the teacher, which a) is more correct as the
      # trajectory and autoregressive actions are sampled according to the
      # learned policy and b) incentivizes the agent to refine what humans do as
      # opposed to the usual "reverse" KL from supervised learning which forces
      # the policy to imitate all behaviors of the teacher, including mistakes.
      teacher_kl = self._compute_kl(policy_distribution, teacher_distribution)
      actor_kl = self._compute_kl(actor_distribution, policy_distribution)
      reverse_teacher_kl = self._compute_kl(teacher_distribution, policy_distribution)

      log_rhos = policy_outputs.log_probs - actor_log_probs
      rhos = tf.exp(log_rhos)

      eps = self._config.ppo.epsilon
      clipped_log_rhos = tf.clip_by_value(log_rhos, -eps, eps)
      clipped_rhos = tf.exp(clipped_log_rhos)

      ppo_objective = tf.minimum(rhos * advantages, clipped_rhos * advantages)

      weighted_losses = [
          - self._config.policy_gradient_weight * ppo_objective,
          self._config.ppo.beta * actor_kl,
          self._config.kl_teacher_weight * teacher_kl,
          self._config.reverse_kl_teacher_weight * reverse_teacher_kl,
          -self._config.entropy_weight * entropy,
      ]
      loss = tf.reduce_mean(tf.add_n(weighted_losses))
      grads = tape.gradient(loss, self._policy_vars)
      # tf.while_loop doesn't like None's in the loop vars
      grads = [
          tf.zeros_like(v) if g is None else g
          for g, v in zip(grads, self._policy_vars)]

    metrics = dict(
        total_loss=loss,
        ppo_objective=ppo_objective,
        teacher_kl=teacher_kl,
        entropy=entropy,
        actor_kl=actor_kl,
        reverse_teacher_kl=reverse_teacher_kl,
    )

    return grads, metrics

  def ppo_grads_acc(self, outputs: LearnerOutputs, trajectory: Trajectory, grads_acc: list):
    grads, metrics = self.compiled_ppo_grads(outputs, trajectory)
    grads_acc = [a + g for a, g in zip(grads_acc, grads)]
    return metrics, grads_acc

  @tf.function
  def apply_grads(self, grads, scale: float = 1):
    grads = [g * scale for g in grads]
    self.policy_optimizer.apply(grads, self._policy_vars)

  def ppo_epoch_full(
      self,
      learner_outputs: list[LearnerOutputs],
      trajectories: list[Trajectory],
      train: bool,
  ) -> tuple[dict, dict[str, tp.Any]]:
    # Could cache this?
    grads_acc = [np.zeros(v.shape, dtype=v.dtype.as_numpy_dtype()) for v in self._policy_vars]
    metrics_acc = []

    metrics_acc = []
    mode_acc = self._init_mode_accumulators()
    for outputs, trajectory in zip(learner_outputs, trajectories):
      metrics, grads_acc = self.compiled_ppo_grads_acc(outputs, trajectory, grads_acc)
      metrics_acc.append(metrics)
      policy_metrics = tf.nest.map_structure(tf.identity, metrics)
      self._accumulate_mode_metrics(
          mode_acc,
          trajectory,
          policy_metrics,
          tf.nest.map_structure(tf.identity, outputs.value.metrics),
      )

    if train:
      self.apply_grads(grads_acc, scale=1 / len(learner_outputs))

    metrics_acc = tf.nest.map_structure(lambda t: t.numpy(), metrics_acc)
    metrics = utils.batch_nest(metrics_acc)

    # Make sure to take max over whole epoch, not just over the minibatch.
    actor_kl = metrics['actor_kl']
    metrics['actor_kl'] = dict(
        mean=np.mean(actor_kl),
        max=np.amax(actor_kl),
    )
    return metrics, self._finalize_mode_metrics(mode_acc)

  @tf.function(autograph=False)
  def ppo_epoch_full_tf(
      self,
      learner_outputs: list[LearnerOutputs],
      trajectories: list[Trajectory],
      train: bool,
  ):
    # Accumulate gradients across the entire batch.
    grads_acc = [tf.zeros_like(v) for v in self._policy_vars]

    # def body(inputs, grads_acc: list):
    #   learner_output, trajectory = inputs
    #   grads, metrics = self.ppo_grads(learner_output, trajectory)
    #   grads_acc = [combine_grads(a, g) for a, g in zip(grads_acc, grads)]
    #   return metrics, grads_acc

    # metrics, grads = tf_utils.dynamic_rnn(
    #     body, (learner_outputs, trajectories), grads_acc)

    metrics_acc = []
    for outputs, trajectory in zip(learner_outputs, trajectories):
      with tf.control_dependencies(grads_acc):
        metrics, grads_acc = self.compiled_ppo_grads_acc(outputs, trajectory, grads_acc)
        metrics_acc.append(metrics)

    metrics = tf.nest.map_structure(lambda *xs: tf.stack(xs), *metrics_acc)

    if train:
      self.policy_optimizer.apply(grads_acc, self._policy_vars)

    # Make sure to take max over whole epoch, not just over the minibatch.
    actor_kl = metrics['actor_kl']
    metrics['actor_kl'] = dict(
        mean=tf.reduce_mean(actor_kl),
        max=tf.reduce_max(actor_kl),
    )
    return metrics

  @tf.function
  def ppo_batch(self, outputs: LearnerOutputs, trajectory: Trajectory, train: bool):
    grads, metrics = self.compiled_ppo_grads(outputs, trajectory)
    if train:
      self.policy_optimizer.apply(grads, self._policy_vars)
    return metrics

  def ppo_epoch_batched(
      self,
      learner_outputs: list[LearnerOutputs],
      trajectories: list[Trajectory],
      train: bool,
  ) -> tuple[dict, dict[str, tp.Any]]:
    """Per-minibatch gradients."""
    metrics = []
    mode_acc = self._init_mode_accumulators()
    for outputs, trajectory in zip(learner_outputs, trajectories):
      metrics.append(self.ppo_batch(outputs, trajectory, train))
      self._accumulate_mode_metrics(
          mode_acc,
          trajectory,
          tf.nest.map_structure(tf.identity, metrics[-1]),
          tf.nest.map_structure(tf.identity, outputs.value.metrics),
      )

    metrics = tf.nest.map_structure(lambda t: t.numpy(), metrics)
    metrics = utils.batch_nest(metrics)

    # Make sure to take max over whole epoch, not just over the minibatch.
    actor_kl = metrics['actor_kl']
    metrics['actor_kl'] = dict(
        mean=np.mean(actor_kl),
        max=np.amax(actor_kl),
    )
    return metrics, self._finalize_mode_metrics(mode_acc)

  def ppo(
      self,
      trajectories: list[Trajectory],
      initial_state: LearnerState,
      num_epochs: int = None,
  ) -> tuple[LearnerState, dict]:
    assert self._use_separate_vf

    trajectories = [self._mask_trajectory(t) for t in trajectories]
    trajectories = [
        update_rewards(t, self._config.reward)
        for t in trajectories]

    learner_outputs: list[LearnerOutputs] = []

    hidden_state = initial_state
    for trajectory in trajectories:
      outputs, hidden_state = self.compiled_unroll(
          trajectory, hidden_state, train_value_function=True)
      learner_outputs.append(outputs)

    value_metrics = [outputs.value.metrics for outputs in learner_outputs]
    value_metrics = utils.map_single_structure(
        lambda t: t.numpy(), value_metrics)
    value_metrics = utils.batch_nest(value_metrics)

    if num_epochs is None:
      num_epochs = self._config.ppo.num_epochs
    # learner_outputs = utils.batch_nest(learner_outputs)
    # trajectories = utils.batch_nest(trajectories)

    if self._config.ppo.minibatched:
      ppo_epoch = self.ppo_epoch_batched
    else:
      ppo_epoch = self.ppo_epoch_full

    checkpoint_vars = tf.nest.map_structure(tf.identity, self.get_vars())

    per_epoch_metrics = []
    per_epoch_mode_stats = []
    for _ in range(num_epochs):
      epoch_metrics, epoch_modes = ppo_epoch(learner_outputs, trajectories, train=True)
      per_epoch_metrics.append(epoch_metrics)
      per_epoch_mode_stats.append(epoch_modes)
    eval_metrics, eval_modes = ppo_epoch(learner_outputs, trajectories, train=False)
    per_epoch_metrics.append(eval_metrics)
    per_epoch_mode_stats.append(eval_modes)

    # If the step was too big, revert to the previous parameters.
    # TODO: if this happens frequently, reduce the learning rate.
    reverted = False
    if per_epoch_metrics[-1]['actor_kl']['mean'] > self._config.ppo.max_mean_actor_kl:
      tf.nest.map_structure(
          lambda v, c: v.assign(c), self.get_vars(), checkpoint_vars)
      reverted = True

    metrics = dict(
        ppo_step={str(i): d for i, d in enumerate(per_epoch_metrics)},
        post_update=per_epoch_metrics[-1],
        value=value_metrics,
        reverted=reverted,
        per_mode=per_epoch_mode_stats[-1],
    )

    return hidden_state, metrics

  @property
  def trainable_variables(self) -> tp.Sequence[tf.Variable]:
    return (self._policy.trainable_variables +
            self._value_function.trainable_variables)

  def initialize(self, trajectory: Trajectory):
    """Initialize model and optimizer variables."""
    # Note that optimizers need to be initialized with variables in the same
    # order as during imitation learning.
    original_states = trajectory.states
    trajectory = self._mask_trajectory(trajectory)
    trajectory = trajectory._replace(
        states=self._match_state_dtypes(trajectory.states, original_states))
    batch_size = trajectory.is_resetting.shape[1]
    self.unroll(trajectory, self.initial_state(batch_size))
    self._value_vars = self._value_function.variables
    self.value_optimizer._initialize(self._value_vars)

    frames = get_frames(trajectory)
    self._policy.unroll(frames, trajectory.initial_state)
    self._policy_vars = self._policy.variables
    self.policy_optimizer._initialize(self._policy_vars)

  def restore_from_imitation(self, imitation_state: dict):
    tf_state = self.get_vars()
    state = {k: imitation_state[k] for k in tf_state}
    tf.nest.map_structure(
        lambda var, val: var.assign(val),
        tf_state, state)

    # Unfortunately the optimizer state includes the learning rate, so it will
    # be overridden by the imitation learning rate.
    self.learning_rate.assign(self._config.learning_rate)

  def get_vars(self) -> dict:
    # For restoration, this structure needs to conform to imitation learning.
    return dict(
        policy=self._policy.variables,
        value_function=self._value_function.variables,
        optimizers=dict(
            policy=self.policy_optimizer.variables,
            value=self.value_optimizer.variables,
        ),
    )

  def get_state(self) -> dict:
    return tf.nest.map_structure(lambda t: t.numpy(), self.get_vars())
