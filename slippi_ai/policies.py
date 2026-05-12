import dataclasses
import enum
from typing import Any, Tuple
import typing as tp

import sonnet as snt
import tensorflow as tf

from slippi_ai.controller_heads import (
    ControllerHead,
    DistanceOutputs,
    SampleOutputs,
)
from slippi_ai.rl_lib import discounted_returns
from slippi_ai import data, networks, embed, opponent_pooling as opponent_pooling_lib, types, tf_utils, utils
from slippi_ai.flag_utils import dataclass_from_dict
from slippi_ai.value_function import ValueOutputs

Outputs = tf_utils.Outputs
RecurrentState = networks.RecurrentState
ControllerType = tp.TypeVar('ControllerType')
RecurrentStateT = tp.TypeVar('RecurrentStateT')


class Platform(enum.Enum):
  TF = 'tf'
  JAX = 'jax'

def _mean_nest(x, y):
  return tf.nest.map_structure(lambda a, b: 0.5 * (a + b), x, y)

class UnrollOutputs(tp.NamedTuple):
  log_probs: tf.Tensor  # [T, B]
  distances: DistanceOutputs  # Struct of [T, B]
  value_outputs: ValueOutputs
  final_state: RecurrentState  # [B]
  metrics: dict  # mixed

class UnrollWithOutputs(tp.NamedTuple):
  imitation_loss: tf.Tensor  # [T, B]
  distances: DistanceOutputs  # Struct of [T, B]
  outputs: tf.Tensor  # [T, B]
  final_state: RecurrentState  # [B]
  metrics: dict  # mixed

class Policy(snt.Module, tp.Generic[ControllerType, RecurrentStateT]):

  def __init__(
      self,
      network: networks.Network,
      controller_head: ControllerHead,
      embed_game: embed.StructEmbedding[data.Game],
      num_names: int,
      train_value_head: bool = True,
      delay: int = 0,
      opponent_pooling: tp.Optional[tp.Union[opponent_pooling_lib.OpponentPoolingConfig, dict]] = None,
  ):
    super().__init__(name='Policy')
    self.network = network
    self.controller_head = controller_head
    self.embed_game = embed_game
    self.embed_state_action = embed.get_state_action_embedding(
        embed_game=embed_game,
        embed_action=self.controller_embedding,
        num_names=num_names,
    )

    if opponent_pooling is None:
      opp_pool_cfg = opponent_pooling_lib.OpponentPoolingConfig()
    elif isinstance(opponent_pooling, dict):
      opp_pool_cfg = dataclass_from_dict(
          opponent_pooling_lib.OpponentPoolingConfig, opponent_pooling)
    else:
      opp_pool_cfg = opponent_pooling
    self._opponent_pooling = opponent_pooling_lib.OpponentPoolingPreprocessor(
        embed_game=embed_game,
        config=opp_pool_cfg,
        name="opponent_pooling",
    )

    self.initial_state = self.network.initial_state
    self.train_value_head = train_value_head
    self.delay = delay

    self.value_head = snt.Linear(1, name='value_head')
    if not train_value_head:
      self.value_head = snt.Sequential([tf.stop_gradient, self.value_head])

  @property
  def platform(self) -> Platform:
    return Platform.TF

  def build_agent(self, batch_size: int, **kwargs):
    from slippi_ai import eval_lib  # avoid circular import
    return eval_lib.BasicAgent(self, batch_size, **kwargs)

  @property
  def controller_embedding(self) -> embed.Embedding[embed.Controller, embed.Action]:
    return self.controller_head.controller_embedding()

  def initialize_variables(self):
    T = 2 + self.delay
    B = 1
    dummy_state_action = self.embed_state_action.dummy([T, B])
    dummy_reward = tf.zeros([T-1, B], tf.float32)
    is_resetting = tf.fill([T, B], False)
    dummy_frames = data.Frames(dummy_state_action, is_resetting, dummy_reward)
    initial_state = self.initial_state(B)

    # imitation_loss also initializes value function
    self.imitation_loss(dummy_frames, initial_state)

  def _value_outputs(
      self,
      outputs,
      last_input,
      is_resetting,
      final_state,
      rewards,
      discount,
      swapped_last_input=None,
      swapped_final_state=None,
  ):
    values = tf.squeeze(self.value_head(outputs), -1)
    last_output, _ = self.network.step_with_reset(
        last_input, is_resetting, final_state)
    if swapped_last_input is not None:
      assert swapped_final_state is not None
      swapped_last_output, _ = self.network.step_with_reset(
          swapped_last_input, is_resetting, swapped_final_state)
      last_output = 0.5 * (last_output + swapped_last_output)
    last_value = tf.squeeze(self.value_head(last_output), -1)
    discounts = tf.fill(tf.shape(rewards), tf.cast(discount, tf.float32))
    value_targets = discounted_returns(
        rewards=rewards,
        discounts=discounts,
        bootstrap=last_value)
    value_targets = tf.stop_gradient(value_targets)
    advantages = value_targets - values
    value_loss = tf.square(advantages)

    _, value_variance = tf_utils.mean_and_variance(value_targets)
    uev = value_loss / (value_variance + 1e-8)

    metrics = {
        'reward': tf_utils.get_stats(rewards),
        'loss': value_loss,
        'return': value_targets,
        'uev': uev,  # unexplained variance
    }

    return ValueOutputs(
        returns=value_targets,
        advantages=advantages,
        loss=value_loss,
        metrics=metrics,
    )

  def unroll(
      self,
      frames: data.Frames,
      initial_state: RecurrentState,
      discount: float = 0.99,
  ) -> UnrollOutputs:
    """Computes prediction loss on a batch of frames.

    Assumes that actions and rewards are delayed, and that one extra
    "overlap" frame is tacked on at the end.

    Args:
      frames: Time-major batch of states, actions, and rewards.
      initial_state: Batch of initial recurrent states.
      value_cost: Weighting of value function loss.
      discount: Per-frame discount factor for returns.
    """
    embedded_inputs = self.embed_state_action(frames.state_action)
    if self._opponent_pooling.is_symmetrized():
      all_inputs = embedded_inputs
      swapped_all_inputs = self._opponent_pooling.swap_opponents(embedded_inputs)
    else:
      all_inputs = self._opponent_pooling(embedded_inputs)
      swapped_all_inputs = None

    inputs, last_input = all_inputs[:-1], all_inputs[-1]
    outputs, branch_final_state = self.network.unroll(
        inputs, frames.is_resetting[:-1], initial_state)

    swapped_last_input = None
    swapped_branch_final_state = None
    final_state = branch_final_state
    if swapped_all_inputs is not None:
      swapped_inputs, swapped_last_input = swapped_all_inputs[:-1], swapped_all_inputs[-1]
      swapped_outputs, swapped_branch_final_state = self.network.unroll(
          swapped_inputs, frames.is_resetting[:-1], initial_state)
      outputs = _mean_nest(outputs, swapped_outputs)
      final_state = _mean_nest(branch_final_state, swapped_branch_final_state)

    # Predict next action.
    action = frames.state_action.action
    prev_action = tf.nest.map_structure(lambda t: t[:-1], action)
    next_action = tf.nest.map_structure(lambda t: t[1:], action)

    distance_outputs = self.controller_head.distance(
        outputs, prev_action, next_action)
    distances = distance_outputs.distance
    policy_loss = tf.add_n(tf.nest.flatten(distances))
    log_probs = -policy_loss

    metrics = dict(
        loss=policy_loss,
        controller=dict(
            types.nt_to_nest(distances),
        )
    )

    value_outputs = self._value_outputs(
        outputs,
        last_input,
        frames.is_resetting[-1],
        branch_final_state,
        frames.reward,
        discount,
        swapped_last_input=swapped_last_input,
        swapped_final_state=swapped_branch_final_state,
    )
    metrics['value'] = value_outputs.metrics

    return UnrollOutputs(
        log_probs=log_probs,
        distances=distance_outputs,
        value_outputs=value_outputs,
        final_state=final_state,
        metrics=metrics)

  def imitation_loss(
      self,
      frames: data.Frames,
      initial_state: RecurrentState,
      discount: float = 0.99,
      value_cost: float = 0.5,
  ) -> tp.Tuple[tf.Tensor, RecurrentState, dict]:
    # Let's say that delay is D and total unroll-length is U + D + 1 (overlap
    # is D + 1). Then the first trajectory has game states [0, U + D] and the
    # second trajectory has game states [U, 2U + D]. That means that we want to
    # use states [0, U-1] to predict actions [D + 1, U + D] (with previous
    # actions being [D, U + D - 1]). The final hidden state should be the one
    # preceding timestep U, meaning we compute it from game states [0, U-1]. We
    # will use game state U to bootstrap the value function.

    state_action = frames.state_action
    # Includes "overlap" frame.
    unroll_length = state_action.state.stage.shape[0] - self.delay

    frames = data.Frames(
        state_action=embed.StateAction(
            state=tf.nest.map_structure(
                lambda t: t[:unroll_length], state_action.state),
            action=tf.nest.map_structure(
                lambda t: t[self.delay:], state_action.action),
            name=state_action.name[self.delay:],
        ),
        is_resetting=frames.is_resetting[:unroll_length],
        # Only use rewards that follow actions.
        reward=frames.reward[self.delay:],
    )

    unroll_outputs = self.unroll(
        frames, initial_state,
        discount=discount,
    )

    metrics = unroll_outputs.metrics

    total_loss = -tf.reduce_mean(unroll_outputs.log_probs)
    if self.train_value_head:
      value_loss = tf.reduce_mean(unroll_outputs.value_outputs.loss)
      total_loss += value_cost * value_loss

    metrics.update(
        total_loss=total_loss,
    )

    return total_loss, unroll_outputs.final_state, metrics

  def unroll_with_outputs(
      self,
      frames: data.Frames,
      initial_state: RecurrentState,
      discount: float = 0.99,
  ):
    embedded_inputs = self.embed_state_action(frames.state_action)
    if self._opponent_pooling.is_symmetrized():
      all_inputs = embedded_inputs
      swapped_all_inputs = self._opponent_pooling.swap_opponents(embedded_inputs)
    else:
      all_inputs = self._opponent_pooling(embedded_inputs)
      swapped_all_inputs = None

    inputs, last_input = all_inputs[:-1], all_inputs[-1]
    outputs, branch_final_state = self.network.unroll(
        inputs, frames.is_resetting[:-1], initial_state)

    swapped_last_input = None
    swapped_branch_final_state = None
    final_state = branch_final_state
    if swapped_all_inputs is not None:
      swapped_inputs, swapped_last_input = swapped_all_inputs[:-1], swapped_all_inputs[-1]
      swapped_outputs, swapped_branch_final_state = self.network.unroll(
          swapped_inputs, frames.is_resetting[:-1], initial_state)
      outputs = _mean_nest(outputs, swapped_outputs)
      final_state = _mean_nest(branch_final_state, swapped_branch_final_state)

    # Predict next action.
    action = frames.state_action.action
    prev_action = tf.nest.map_structure(lambda t: t[:-1], action)
    next_action = tf.nest.map_structure(lambda t: t[1:], action)

    distance_outputs = self.controller_head.distance(
        outputs, prev_action, next_action)
    distances = distance_outputs.distance
    policy_loss = tf.add_n(tf.nest.flatten(distances))

    metrics = dict(
        loss=policy_loss,
        controller=dict(
            types.nt_to_nest(distances),
        )
    )

    # We're only really doing this to initialize the value_head...
    value_outputs = self._value_outputs(
        outputs,
        last_input,
        frames.is_resetting[-1],
        branch_final_state,
        frames.reward,
        discount,
        swapped_last_input=swapped_last_input,
        swapped_final_state=swapped_branch_final_state,
    )
    metrics['value'] = value_outputs.metrics

    return UnrollWithOutputs(
        imitation_loss=policy_loss,
        distances=distances,
        outputs=outputs,
        final_state=final_state,
        metrics=metrics,
    )

  def sample(
      self,
      state_action: embed.StateAction,
      initial_state: RecurrentState,
      is_resetting: tp.Optional[tf.Tensor] = None,
      **kwargs,
  ) -> tp.Tuple[SampleOutputs, RecurrentState]:
    embedded_input = self.embed_state_action(state_action)
    if self._opponent_pooling.is_symmetrized():
      input = embedded_input
      swapped_input = self._opponent_pooling.swap_opponents(embedded_input)
    else:
      input = self._opponent_pooling(embedded_input)
      swapped_input = None

    if is_resetting is None:
      batch_size = input.shape[0]
      is_resetting = tf.fill([batch_size], False)

    output, final_state = self.network.step_with_reset(
        input, is_resetting, initial_state)
    if swapped_input is not None:
      swapped_output, swapped_final_state = self.network.step_with_reset(
          swapped_input, is_resetting, initial_state)
      output = _mean_nest(output, swapped_output)
      final_state = _mean_nest(final_state, swapped_final_state)

    prev_action = state_action.action
    next_action = self.controller_head.sample(
        output, prev_action, **kwargs)
    return next_action, final_state

  def multi_sample(
      self,
      states: list[embed.Game],  # time-indexed
      prev_action: embed.Action,  # only for first step
      name_code: int,
      initial_state: RecurrentState,
      **kwargs,
  ) -> Tuple[list[SampleOutputs], RecurrentState]:
    actions = []
    hidden_state = initial_state
    for game in range(states):
      state_action = embed.StateAction(
          state=game,
          action=prev_action,
          name=name_code,
      )
      next_action, hidden_state = self.sample(
          state_action, hidden_state, **kwargs)
      actions.append(next_action)
      prev_action = next_action

    return actions, hidden_state

@dataclasses.dataclass
class PolicyConfig:
  train_value_head: bool = True
  delay: int = 0
  opponent_pooling: opponent_pooling_lib.OpponentPoolingConfig = utils.field(opponent_pooling_lib.OpponentPoolingConfig)
