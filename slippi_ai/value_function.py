import typing as tp

import tensorflow as tf
import sonnet as snt

from melee.enums import Action

from slippi_ai import data, embed, networks, opponent_pooling as opponent_pooling_lib, tf_utils, types
from slippi_ai.flag_utils import dataclass_from_dict
from slippi_ai.rl_lib import discounted_returns
from slippi_ai.networks import RecurrentState

def _mean_nest(x, y):
  return tf.nest.map_structure(lambda a, b: 0.5 * (a + b), x, y)

class ValueOutputs(tp.NamedTuple):
  returns: tf.Tensor  # [T, B]
  advantages: tf.Tensor  # [T, B]
  loss: tf.Tensor
  metrics: dict

def player_respawn(player: types.Player) -> tf.Tensor:
  """Returns a boolean mask indicating when a player respawns."""
  actions = player.action
  # Note: players seem to be able to skip ON_HALO_WAIT
  respawn = Action.ON_HALO_DESCENT.value
  return tf.logical_and(actions[:-1] != respawn, actions[1:] == respawn)

class ValueFunction(snt.Module):

  def __init__(
      self,
      network_config: dict,
      embed_game: embed.StructEmbedding[data.Game],
      embed_state_action: embed.StructEmbedding[embed.StateAction],
      opponent_pooling: tp.Optional[tp.Union[opponent_pooling_lib.OpponentPoolingConfig, dict]] = None,
  ):
    super().__init__(name='ValueFunction')
    self.network = networks.construct_network(**network_config)

    if opponent_pooling is None:
      opp_pool_cfg = opponent_pooling_lib.OpponentPoolingConfig()
    elif isinstance(opponent_pooling, dict):
      opp_pool_cfg = dataclass_from_dict(
          opponent_pooling_lib.OpponentPoolingConfig, opponent_pooling)
    else:
      opp_pool_cfg = opponent_pooling

    self.embed_state_action = embed_state_action
    self._opponent_pooling = opponent_pooling_lib.OpponentPoolingPreprocessor(
        embed_game=embed_game,
        config=opp_pool_cfg,
        name="opponent_pooling",
    )
    self.value_head = snt.Linear(1, name='value_head')
    self.initial_state = self.network.initial_state

  def loss(
      self,
      frames: data.Frames,
      initial_state: RecurrentState,
      discount: float,
      discount_on_death: tp.Optional[float] = None,
  ) -> tp.Tuple[ValueOutputs, RecurrentState]:
    """Computes prediction loss on a batch of frames.

    Args:
      frames: Time-major batch of states, actions, and rewards.
        Assumed to have one frame of overlap.
      initial_state: Batch of initial recurrent states.
      discount: Per-frame discount factor for returns.
      discount_on_death: Discount factor to use when either player *respawns*.
        The reward for KOs comes on the frame of death, which precedes respawn.
    """
    rewards = frames.reward

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

    # Includes "overlap" frame.
    # unroll_length = state_action.state.stage.shape[0] - delay

    values = tf.squeeze(self.value_head(outputs), -1)
    last_output, _ = self.network.step_with_reset(
        last_input, frames.is_resetting[-1], branch_final_state)
    if swapped_last_input is not None:
      assert swapped_branch_final_state is not None
      swapped_last_output, _ = self.network.step_with_reset(
          swapped_last_input, frames.is_resetting[-1], swapped_branch_final_state)
      last_output = _mean_nest(last_output, swapped_last_output)
    last_value = tf.squeeze(self.value_head(last_output), -1)
    discounts = tf.fill(tf.shape(rewards), tf.cast(discount, tf.float32))

    if discount_on_death is not None:
      respawn_happened = tf.logical_or(
          player_respawn(frames.state_action.state.p0),
          player_respawn(frames.state_action.state.p1))

      discounts_on_death = tf.fill(
          discounts.shape, tf.cast(discount_on_death, tf.float32))

      discounts = tf.where(respawn_happened, discounts_on_death, discounts)
      assert discounts.shape == rewards.shape

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
        'return': tf_utils.get_stats(value_targets),
        'loss': value_loss,
        'variance': value_variance,
        'uev': uev,  # unexplained variance
    }

    outputs = ValueOutputs(
        returns=value_targets,
        advantages=advantages,
        loss=value_loss,
        metrics=metrics,
    )

    return outputs, final_state


@snt.allow_empty_variables
class FakeValueFunction(snt.Module):

  def initial_state(self, batch_size: int) -> RecurrentState:
    del batch_size
    return ()

  def loss(self, frames: data.Frames, initial_state, discount):
    del discount

    outputs = ValueOutputs(
        returns=tf.zeros_like(frames.reward),
        loss=tf.constant(0, dtype=tf.float32),
        advantages=tf.zeros_like(frames.reward),
        metrics={},
    )

    return outputs, initial_state
