import unittest
import concurrent.futures
from collections import deque

import melee
import numpy as np

from slippi_ai.sim_env import jax_rollout
from slippi_ai import data
from slippi_ai import reward
from slippi_ai import types
from slippi_ai import utils
from slippi_ai.controller_heads import SampleOutputs
from slippi_ai.evaluators import Trajectory
from slippi_ai.jax.rl import learner as jax_learner
from slippi_ai.rl import learner as tf_learner


def _controller(shape):
  return types.Controller(
      main_stick=types.Stick(
          x=np.full(shape, 0.5, dtype=np.float32),
          y=np.full(shape, 0.5, dtype=np.float32),
      ),
      c_stick=types.Stick(
          x=np.full(shape, 0.5, dtype=np.float32),
          y=np.full(shape, 0.5, dtype=np.float32),
      ),
      shoulder=np.zeros(shape, dtype=np.float32),
      buttons=types.Buttons(**{
          name: np.zeros(shape, dtype=np.bool_)
          for name in types.Buttons._fields
      }),
  )


def _nana(shape):
  return types.Nana(
      exists=np.zeros(shape, dtype=np.bool_),
      percent=np.zeros(shape, dtype=np.uint16),
      facing=np.ones(shape, dtype=np.bool_),
      x=np.zeros(shape, dtype=np.float32),
      y=np.zeros(shape, dtype=np.float32),
      action=np.full(shape, melee.Action.STANDING.value, dtype=np.uint16),
      invulnerable=np.zeros(shape, dtype=np.bool_),
      character=np.zeros(shape, dtype=np.uint8),
      jumps_left=np.zeros(shape, dtype=np.uint8),
      shield_strength=np.full(shape, 60.0, dtype=np.float32),
      on_ground=np.ones(shape, dtype=np.bool_),
  )


def _player(shape, character, actions, stocks, percent=0):
  return types.Player(
      percent=np.full(shape, percent, dtype=np.uint16),
      facing=np.ones(shape, dtype=np.bool_),
      x=np.zeros(shape, dtype=np.float32),
      y=np.zeros(shape, dtype=np.float32),
      action=np.asarray(actions, dtype=np.uint16).reshape(shape),
      invulnerable=np.zeros(shape, dtype=np.bool_),
      character=np.full(shape, character.value, dtype=np.uint8),
      jumps_left=np.ones(shape, dtype=np.uint8),
      shield_strength=np.full(shape, 60.0, dtype=np.float32),
      on_ground=np.ones(shape, dtype=np.bool_),
      is_dead=np.asarray(stocks, dtype=np.uint8).reshape(shape) == 0,
      stocks_left=np.asarray(stocks, dtype=np.uint8).reshape(shape),
      controller=_controller(shape),
      nana=_nana(shape),
  )


def _empty_player(shape):
  return _player(
      shape,
      melee.Character.FOX,
      np.full(shape, melee.Action.STANDING.value, dtype=np.uint16),
      np.zeros(shape, dtype=np.uint8),
  )


def _empty_items(shape):
  return types.Items(**{
      f'item_{i}': types.Item(
          exists=np.zeros(shape, dtype=np.bool_),
          type=np.zeros(shape, dtype=np.uint16),
          state=np.zeros(shape, dtype=np.uint8),
          x=np.zeros(shape, dtype=np.float32),
          y=np.zeros(shape, dtype=np.float32),
      )
      for i in range(types.MAX_ITEMS)
  })


def _terminal_stock_loss_game():
  shape = (3, 1)
  standing = melee.Action.STANDING.value
  dead = melee.Action.DEAD_DOWN.value
  return types.Game(
      p0=_player(
          shape,
          melee.Character.FOX,
          actions=[standing, standing, standing],
          stocks=[1, 1, 1],
      ),
      p1=_empty_player(shape),
      p2=_player(
          shape,
          melee.Character.FOX,
          actions=[standing, standing, dead],
          stocks=[1, 1, 0],
          percent=75,
      ),
      p3=_empty_player(shape),
      stage=np.full(shape, melee.Stage.FINAL_DESTINATION.value, dtype=np.uint8),
      randall_phase=np.zeros(shape, dtype=np.float32),
      randall=types.Randall(
          x=np.zeros(shape, dtype=np.float32),
          y=np.zeros(shape, dtype=np.float32),
      ),
      items=_empty_items(shape),
      is_teams=np.zeros(shape, dtype=np.bool_),
  )


def _game_from_players(shape, p0, p2):
  return types.Game(
      p0=p0,
      p1=_empty_player(shape),
      p2=p2,
      p3=_empty_player(shape),
      stage=np.full(shape, melee.Stage.FINAL_DESTINATION.value, dtype=np.uint8),
      randall_phase=np.zeros(shape, dtype=np.float32),
      randall=types.Randall(
          x=np.zeros(shape, dtype=np.float32),
          y=np.zeros(shape, dtype=np.float32),
      ),
      items=_empty_items(shape),
      is_teams=np.zeros(shape, dtype=np.bool_),
  )


class JaxRlLearnerTest(unittest.TestCase):

  def test_update_rewards_matches_legacy_tf_on_reset_boundary_stock_loss(self):
    is_resetting = np.array([[False], [False], [True]], dtype=np.bool_)
    trajectory = Trajectory(
        states=_terminal_stock_loss_game(),
        name=np.zeros((3, 1), dtype=np.int32),
        actions=None,
        rewards=np.zeros((2, 1), dtype=np.float32),
        is_resetting=is_resetting,
        initial_state=None,
        delayed_actions=[],
    )
    config = reward.RewardConfig()

    expected = tf_learner.update_rewards(trajectory, config).rewards
    actual = jax_learner.update_rewards(trajectory, config).rewards

    np.testing.assert_allclose(actual, expected)
    self.assertGreater(actual[1, 0], 0.0)

  def test_batched_transition_rewards_match_per_step_terminal_correction(self):
    shape = (2,)
    standing = melee.Action.STANDING.value
    dead = melee.Action.DEAD_DOWN.value
    p0 = _player(
        shape,
        melee.Character.FOX,
        actions=[standing, standing],
        stocks=[1, 1],
    )
    state0 = _game_from_players(
        shape,
        p0,
        _player(
            shape,
            melee.Character.FOX,
            actions=[standing, standing],
            stocks=[1, 1],
            percent=25,
        ),
    )
    state1 = _game_from_players(
        shape,
        p0,
        _player(
            shape,
            melee.Character.FOX,
            actions=[standing, standing],
            stocks=[1, 1],
            percent=50,
        ),
    )
    reset_state2 = _game_from_players(
        shape,
        p0,
        _player(
            shape,
            melee.Character.FOX,
            actions=[standing, standing],
            stocks=[1, 4],
            percent=0,
        ),
    )
    terminal_state2 = _game_from_players(
        shape,
        p0,
        _player(
            shape,
            melee.Character.FOX,
            actions=[standing, dead],
            stocks=[1, 0],
            percent=75,
        ),
    )
    reset_mask = np.array([False, True], dtype=np.bool_)
    time_major = utils.batch_nest_nt([
        state0,
        state1,
        reset_state2,
    ])
    uncorrected = jax_rollout.batched_transition_rewards(
        time_major,
        terminal_reward_overrides=[],
        reward_config=reward.RewardConfig(),
    )
    actual = jax_rollout.batched_transition_rewards(
        time_major,
        terminal_reward_overrides=[
            jax_rollout.TerminalRewardOverride(
                transition_index=1,
                reset_mask=reset_mask,
                terminal_game=jax_rollout.masked_numpy_tree(
                    terminal_state2,
                    reset_mask,
                ),
            ),
        ],
        reward_config=reward.RewardConfig(),
    )

    self.assertEqual(actual.shape, (2, 2))
    np.testing.assert_allclose(actual[0], uncorrected[0])
    self.assertAlmostEqual(float(actual[1, 0]), float(uncorrected[1, 0]))
    self.assertGreater(float(actual[1, 1]), float(uncorrected[1, 1]))

  def test_reset_does_not_rewrite_historical_learner_delay_outputs(self):
    queue = deque([
        SampleOutputs(
            controller_state=np.array([10 + i, 20 + i], dtype=np.int32),
            logits=np.array([50 + i, 60 + i], dtype=np.float32),
        )
        for i in range(3)
    ])
    dummy = SampleOutputs(
        controller_state=np.array([-1, -1], dtype=np.int32),
        logits=np.array([-3.0, -3.0], dtype=np.float32),
    )

    # The env input-delay queue may be flushed on reset, but the learner queue
    # stores actor outputs for already-collected frames. Rewriting it corrupts
    # PPO's old-policy logits/actions exactly around terminal boundaries.
    reset_mask = np.array([False, True], dtype=np.bool_)
    before = [np.asarray(v.controller_state).copy() for v in queue]

    env_queue = deque([v.controller_state.copy() for v in queue])
    jax_rollout.reset_delay_queues(
        env_action_queue=env_queue,
        learner_action_queue=queue,
        dummy_outputs=dummy,
        reset_mask=reset_mask,
    )

    self.assertEqual([v.tolist() for v in before], [[10, 20], [11, 21], [12, 22]])
    self.assertEqual(
        [v.controller_state.tolist() for v in queue],
        [v.tolist() for v in before],
    )
    self.assertEqual(
        [v.tolist() for v in env_queue],
        [[10, -1], [11, -1], [12, -1]],
    )

  def test_chunked_env_delay_queue_matches_single_step_resets(self):
    queue_start = [
        np.array([10 + i, 20 + i], dtype=np.int32)
        for i in range(4)
    ]
    samples = [
        SampleOutputs(
            controller_state=np.array([100 + i, 200 + i], dtype=np.int32),
            logits=np.array([300 + i, 400 + i], dtype=np.float32),
        )
        for i in range(3)
    ]
    reset_masks = [
        np.array([False, False], dtype=np.bool_),
        np.array([False, True], dtype=np.bool_),
        np.array([False, False], dtype=np.bool_),
    ]
    dummy = SampleOutputs(
        controller_state=np.array([-1, -2], dtype=np.int32),
        logits=np.array([-3.0, -4.0], dtype=np.float32),
    )

    single_step_queue = deque([v.copy() for v in queue_start])
    for sample, reset_mask in zip(samples, reset_masks):
      if np.any(reset_mask):
        jax_rollout.reset_delay_queue_lanes(
            single_step_queue,
            dummy.controller_state,
            reset_mask,
        )
      single_step_queue.append(sample.controller_state.copy())
      single_step_queue.popleft()

    chunked_queue = deque([v.copy() for v in queue_start])
    jax_rollout.replace_env_action_queue_after_chunk(
        env_action_queue=chunked_queue,
        queue_start=[v.copy() for v in queue_start],
        sample_outputs_list=samples,
        reset_masks=reset_masks,
        dummy_outputs=dummy,
    )

    self.assertEqual(
        [v.tolist() for v in chunked_queue],
        [v.tolist() for v in single_step_queue],
    )
    # The sample from frame 0 is still pending after the reset lane, so it must
    # be neutralized for that lane just like the per-frame rollout path.
    self.assertEqual(chunked_queue[1].tolist(), [100, -2])

  def test_pending_env_action_applies_reset_before_resolution(self):
    future = concurrent.futures.Future()
    future.set_result([
        SampleOutputs(
            controller_state=np.array([10, 20], dtype=np.int32),
            logits=np.array([1.0, 2.0], dtype=np.float32),
        )
    ])
    pending = jax_rollout.PendingEnvAction(future=future, index=0)
    queue = deque([pending])
    dummy = SampleOutputs(
        controller_state=np.array([-1, -2], dtype=np.int32),
        logits=np.array([-3.0, -4.0], dtype=np.float32),
    )

    jax_rollout.reset_delay_queue_lanes(
        queue,
        dummy.controller_state,
        np.array([False, True], dtype=np.bool_),
    )
    resolved = jax_rollout.resolve_env_action_entry(
        queue[0],
        dummy_outputs=dummy,
    )

    self.assertEqual(resolved.tolist(), [10, -2])

  def test_reset_frame_actions_splits_network_input_from_actor_outputs(self):
    controller_state = np.array([
        [10, 11],
        [20, 21],
        [30, 31],
        [40, 41],
    ], dtype=np.int32)
    logits = controller_state.astype(np.float32) + 100.0
    actions = SampleOutputs(controller_state=controller_state, logits=logits)
    is_resetting = np.array([
        [False, False],
        [False, True],
        [False, False],
        [False, False],
    ], dtype=np.bool_)
    trajectory = Trajectory(
        states=None,
        name=np.zeros((4, 2), dtype=np.int32),
        actions=actions,
        rewards=np.zeros((3, 2), dtype=np.float32),
        is_resetting=is_resetting,
        initial_state=None,
        delayed_actions=[],
    )

    # This mirrors the policy-frame construction for delay=1: the action leaf
    # is the previous-controller input to the network, not the old-policy
    # SampleOutputs tree used for actor KL/logprobs.
    policy_frames = data.Frames(
        state_action=data.StateAction(
            state=None,
            action=trajectory.actions.controller_state[1:],
            name=trajectory.name[:-1],
        ),
        is_resetting=trajectory.is_resetting[:-1],
        reward=trajectory.rewards[1:],
    )
    reset_frames = jax_learner.reset_frame_actions(
        policy_frames,
        dummy_action=np.full((3, 2), -1, dtype=np.int32),
    )

    np.testing.assert_array_equal(
        np.asarray(reset_frames.state_action.action),
        np.array([
            [20, 21],
            [30, -1],
            [40, 41],
        ], dtype=np.int32),
    )
    # The original PPO old-policy outputs remain untouched. In particular, the
    # target action/logit at the reset lane is still the sampled actor output;
    # only the recurrent previous-action input was reset to dummy.
    np.testing.assert_array_equal(trajectory.actions.controller_state, controller_state)
    np.testing.assert_array_equal(trajectory.actions.logits, logits)

  def test_reset_target_frames_are_masked_from_policy_metrics(self):
    is_resetting = np.array([
        [False, False],
        [False, False],
        [False, False],
        [False, False],
        [True, False],
        [False, True],
        [False, False],
    ], dtype=np.bool_)
    actor_kl = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
    ], dtype=np.float32)

    mask = jax_learner.valid_policy_step_mask(is_resetting, delay=2)
    masked = jax_learner.mask_policy_array(actor_kl, mask)
    summary = jax_learner.summarize_policy_array_masked(actor_kl, mask)

    np.testing.assert_array_equal(
        np.asarray(mask),
        np.array([
            [True, True],
            [True, True],
            [False, True],
            [False, False],
        ], dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        np.asarray(masked),
        np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [0.0, 30.0],
            [0.0, 0.0],
        ], dtype=np.float32),
    )
    self.assertAlmostEqual(float(summary['mean']), 63.0 / 5.0, places=5)
    self.assertAlmostEqual(float(summary['max']), 30.0)

  def test_valid_policy_step_mask_matches_bruteforce(self):
    rng = np.random.default_rng(17)
    for delay in [0, 1, 2, 21]:
      for length in [delay + 2, delay + 5, delay + 23]:
        resets = rng.random((length, 3)) < 0.2
        mask = np.asarray(jax_learner.valid_policy_step_mask(resets, delay))
        expected = np.ones((length - delay - 1, 3), dtype=np.bool_)
        for t in range(expected.shape[0]):
          expected[t] = ~np.any(resets[t + 1:t + delay + 1], axis=0)
        np.testing.assert_array_equal(mask, expected)

  def test_reset_invalidates_delayed_actions_that_were_never_applied(self):
    delay = 21
    length = 80 + 1
    resets = np.zeros((length, 2), dtype=np.bool_)
    resets[40, 0] = True

    mask = np.asarray(jax_learner.valid_policy_step_mask(resets, delay))
    invalid_indices = np.flatnonzero(~mask[:, 0])

    np.testing.assert_array_equal(invalid_indices, np.arange(19, 40))
    self.assertTrue(np.all(mask[:19, 0]))
    self.assertTrue(np.all(mask[40:, 0]))
    self.assertTrue(np.all(mask[:, 1]))


if __name__ == '__main__':
  unittest.main()
