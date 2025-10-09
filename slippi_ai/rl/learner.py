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


class StatsAccumulator:

  def __init__(self):
    self.count = 0
    self.total = 0.0
    self.total_sq = 0.0
    self._min = None
    self._max = None

  def add(self, values: np.ndarray):
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
      return
    self.count += flat.size
    self.total += float(flat.sum())
    self.total_sq += float(np.square(flat).sum())
    current_min = float(flat.min())
    current_max = float(flat.max())
    if self._min is None or current_min < self._min:
      self._min = current_min
    if self._max is None or current_max > self._max:
      self._max = current_max

  def finalize(self) -> dict | None:
    if self.count == 0:
      return None
    mean = self.total / self.count
    variance = max(self.total_sq / self.count - mean * mean, 0.0)
    return {
        'mean': float(mean),
        'variance': float(variance),
        'stddev': float(np.sqrt(variance)),
        'min': float(self._min),
        'max': float(self._max),
    }


class ModeAggregator:

  def __init__(self):
    def _bucket():
      return {
          'envs': 0,
          'ports': 0,
          'reward': StatsAccumulator(),
          'ppo_objective': StatsAccumulator(),
          'teacher_kl': StatsAccumulator(),
          'actor_kl': StatsAccumulator(),
          'entropy': StatsAccumulator(),
      }

    self._buckets: dict[str, dict] = {
        'singles': _bucket(),
        'doubles': _bucket(),
    }

  def add(self, mode: str, mask: np.ndarray, reward: np.ndarray, metrics: dict[str, np.ndarray]):
    slot_mask = np.asarray(mask, dtype=bool)
    slot_count = int(slot_mask.sum())
    if slot_count == 0:
      return

    bucket = self._buckets[mode]
    bucket['ports'] = max(bucket['ports'], slot_count)

    total_ports = slot_mask.size
    env_mask = None
    ports_per_env = 4 if reward.shape[1] == total_ports and total_ports % 4 == 0 else None
    if ports_per_env:
      try:
        env_mask = slot_mask.reshape(ports_per_env, -1).any(axis=0)
      except ValueError:
        env_mask = None

    if env_mask is not None:
      env_count = int(env_mask.sum())
      bucket['envs'] = max(bucket['envs'], env_count)

      reward_reshaped = reward.reshape(reward.shape[0], ports_per_env, -1)
      ally_reward = reward_reshaped[:, 0, env_mask]
      bucket['reward'].add(ally_reward)
    else:
      bucket['envs'] = max(bucket['envs'], slot_count)
      bucket['reward'].add(reward[:, slot_mask])

    for name in ('ppo_objective', 'teacher_kl', 'actor_kl', 'entropy'):
      bucket[name].add(metrics[name][:, slot_mask])

  def finalize(self) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for mode, bucket in self._buckets.items():
      if bucket['envs'] <= 0:
        continue
      mode_metrics = {
          'count': int(bucket['envs']),
          'ports': int(bucket['ports']),
      }
      for name in ('reward', 'ppo_objective', 'teacher_kl', 'actor_kl', 'entropy'):
        stats = bucket[name].finalize()
        if stats:
          mode_metrics[name] = stats
      result[mode] = mode_metrics
    return result

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

  def unroll(
      self,
      trajectory: Trajectory,
      initial_state: LearnerState,
      train_value_function: bool = False,
  ) -> tp.Tuple[LearnerOutputs, LearnerState]:
    assert len(trajectory.delayed_actions) == self._policy.delay

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

    delay = self._policy.delay  # "D"
    remove_first = lambda t: t[delay:]
    remove_last = lambda t: t[:t.shape[0] - delay]

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
    batch_size = is_resetting.shape[0]
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
      capture_timeseries: bool = False,
  ):
    # Could cache this?
    grads_acc = [np.zeros(v.shape, dtype=v.dtype.as_numpy_dtype()) for v in self._policy_vars]
    metrics_acc = []

    metrics_acc = []
    for outputs, trajectory in zip(learner_outputs, trajectories):
      metrics, grads_acc = self.compiled_ppo_grads_acc(outputs, trajectory, grads_acc)
      metrics_acc.append(metrics)

    if train:
      self.apply_grads(grads_acc, scale=1 / len(learner_outputs))

    metrics_acc = tf.nest.map_structure(lambda t: t.numpy(), metrics_acc)
    metrics = utils.batch_nest(metrics_acc)

    timeseries = None
    if capture_timeseries:
      timeseries = {
          'ppo_objective': np.array(metrics['ppo_objective']),
          'teacher_kl': np.array(metrics['teacher_kl']),
          'entropy': np.array(metrics['entropy']),
          'actor_kl': np.array(metrics['actor_kl']),
      }

    # Make sure to take max over whole epoch, not just over the minibatch.
    actor_kl = metrics['actor_kl']
    metrics['actor_kl'] = dict(
        mean=np.mean(actor_kl),
        max=np.amax(actor_kl),
    )
    if capture_timeseries:
      metrics['_timeseries'] = timeseries
    return metrics

  @tf.function(autograph=False)
  def ppo_epoch_full_tf(
      self,
      learner_outputs: list[LearnerOutputs],
      trajectories: list[Trajectory],
      train: bool,
      capture_timeseries: bool = False,
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

    timeseries = None
    if capture_timeseries:
      timeseries = {
          'ppo_objective': tf.convert_to_tensor(metrics['ppo_objective']).numpy(),
          'teacher_kl': tf.convert_to_tensor(metrics['teacher_kl']).numpy(),
          'entropy': tf.convert_to_tensor(metrics['entropy']).numpy(),
          'actor_kl': tf.convert_to_tensor(metrics['actor_kl']).numpy(),
      }

    if train:
      self.policy_optimizer.apply(grads_acc, self._policy_vars)

    # Make sure to take max over whole epoch, not just over the minibatch.
    actor_kl = metrics['actor_kl']
    metrics['actor_kl'] = dict(
        mean=tf.reduce_mean(actor_kl),
        max=tf.reduce_max(actor_kl),
    )
    if capture_timeseries:
      metrics['_timeseries'] = timeseries
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
      capture_timeseries: bool = False,
  ):
    """Per-minibatch gradients."""
    metrics = []
    for outputs, trajectory in zip(learner_outputs, trajectories):
      metrics.append(self.ppo_batch(outputs, trajectory, train))

    metrics = tf.nest.map_structure(lambda t: t.numpy(), metrics)
    metrics = utils.batch_nest(metrics)

    timeseries = None
    if capture_timeseries:
      timeseries = {
          'ppo_objective': np.array(metrics['ppo_objective']),
          'teacher_kl': np.array(metrics['teacher_kl']),
          'entropy': np.array(metrics['entropy']),
          'actor_kl': np.array(metrics['actor_kl']),
      }

    # Make sure to take max over whole epoch, not just over the minibatch.
    actor_kl = metrics['actor_kl']
    metrics['actor_kl'] = dict(
        mean=np.mean(actor_kl),
        max=np.amax(actor_kl),
    )
    if capture_timeseries:
      metrics['_timeseries'] = timeseries
    return metrics

  def ppo(
      self,
      trajectories: list[Trajectory],
      initial_state: LearnerState,
      num_epochs: int = None,
  ) -> tuple[LearnerState, dict]:
    assert self._use_separate_vf

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
    for _ in range(num_epochs):
      per_epoch_metrics.append(ppo_epoch(
          learner_outputs, trajectories, train=True, capture_timeseries=False))
    per_epoch_metrics.append(ppo_epoch(
        learner_outputs, trajectories, train=False, capture_timeseries=True))

    # If the step was too big, revert to the previous parameters.
    # TODO: if this happens frequently, reduce the learning rate.
    reverted = False
    if per_epoch_metrics[-1]['actor_kl']['mean'] > self._config.ppo.max_mean_actor_kl:
      tf.nest.map_structure(
          lambda v, c: v.assign(c), self.get_vars(), checkpoint_vars)
      reverted = True

    post_update = per_epoch_metrics[-1]
    timeseries = post_update.pop('_timeseries', None)
    mode_metrics = self._compute_mode_breakdown(trajectories, timeseries)

    metrics = dict(
        ppo_step={str(i): d for i, d in enumerate(per_epoch_metrics)},
        post_update=post_update,
        value=value_metrics,
        mode=mode_metrics,
        reverted=reverted,
    )

    return hidden_state, metrics

  @property
  def trainable_variables(self) -> tp.Sequence[tf.Variable]:
    return (self._policy.trainable_variables +
            self._value_function.trainable_variables)

  def _compute_mode_breakdown(
      self,
      trajectories: list[Trajectory],
      timeseries: tp.Optional[dict[str, np.ndarray]],
  ) -> dict[str, dict]:
    if not timeseries:
      return {}

    aggregator = ModeAggregator()

    objective = np.asarray(timeseries['ppo_objective'])
    teacher_kl = np.asarray(timeseries['teacher_kl'])
    entropy = np.asarray(timeseries['entropy'])
    actor_kl = np.asarray(timeseries['actor_kl'])

    for idx, trajectory in enumerate(trajectories):
      mode_flags = np.asarray(trajectory.states.is_teams[0], dtype=bool)
      reward = np.asarray(trajectory.rewards)

      metrics = {
          'ppo_objective': objective[idx],
          'teacher_kl': teacher_kl[idx],
          'entropy': entropy[idx],
          'actor_kl': actor_kl[idx],
      }

      aggregator.add('singles', np.logical_not(mode_flags), reward, metrics)
      aggregator.add('doubles', mode_flags, reward, metrics)

    return aggregator.finalize()

  def initialize(self, trajectory: Trajectory):
    """Initialize model and optimizer variables."""
    # Note that optimizers need to be initialized with variables in the same
    # order as during imitation learning.
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
