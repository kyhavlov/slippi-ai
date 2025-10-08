import dataclasses
import unittest

import numpy as np
import tensorflow as tf

from slippi_ai import saving, train_lib, utils, value_function as vf_lib
from slippi_ai.rl import learner as learner_lib
from slippi_ai.rl import run_lib
from slippi_ai.controller_heads import SampleOutputs


def _to_numpy(value):
  if isinstance(value, tf.Tensor):
    return value.numpy()
  return np.asarray(value)


def _assert_nested_allclose(testcase, a, b, rtol=1e-6, atol=1e-6):
  def check(x, y):
    x_np = _to_numpy(x)
    y_np = _to_numpy(y)
    testcase.assertEqual(x_np.shape, y_np.shape)
    np.testing.assert_allclose(x_np, y_np, rtol=rtol, atol=atol)

  tf.nest.map_structure(check, a, b)


class LearnerMaskingTest(unittest.TestCase):

  def setUp(self):
    tf.random.set_seed(12345)
    config = train_lib.Config()
    policy_config = dataclasses.asdict(config)

    self.policy = saving.policy_from_config(policy_config)
    self.policy.initialize_variables()
    self.teacher = saving.policy_from_config(policy_config)
    self.teacher.initialize_variables()

    learner_config = learner_lib.LearnerConfig(compile=False)
    learner_config.ppo = learner_lib.PPOConfig(num_epochs=0)

    value_function = vf_lib.ValueFunction(
        network_config=config.value_function.network,
        embed_state_action=self.policy.embed_state_action,
    )
    self.learner = learner_lib.Learner(
        config=learner_config,
        policy=self.policy,
        teacher=self.teacher,
        value_function=value_function,
    )

    dummy = run_lib.dummy_trajectory(self.policy, unroll_length=4, batch_size=1)
    self.learner.initialize(dummy)

  def _mutate_block(self, block: np.ndarray):
    if block.dtype == np.bool_:
      np.logical_not(block, out=block)
    elif np.issubdtype(block.dtype, np.integer):
      np.add(block, 1, out=block, casting='unsafe')
    elif np.issubdtype(block.dtype, np.floating):
      block += 0.5

  def _modify_time_major(self, struct, mask):
    inactive = np.where(~mask)[0]

    def mutate(arr):
      arr_np = np.asarray(arr)
      if arr_np.ndim == 0:
        return arr_np
      arr_copy = arr_np.copy()
      if arr_copy.ndim == 1:
        subset = arr_copy[inactive, ...]
        self._mutate_block(subset)
        arr_copy[inactive, ...] = subset
      else:
        subset = arr_copy[:, inactive, ...]
        self._mutate_block(subset)
        arr_copy[:, inactive, ...] = subset
      return arr_copy

    return utils.map_single_structure(mutate, struct)

  def _modify_batch_major(self, struct, mask):
    inactive = np.where(~mask)[0]

    def mutate(arr):
      arr_np = np.asarray(arr)
      if arr_np.ndim == 0:
        return arr_np
      arr_copy = arr_np.copy()
      subset = arr_copy[inactive, ...]
      self._mutate_block(subset)
      arr_copy[inactive, ...] = subset
      return arr_copy

    return utils.map_single_structure(mutate, struct)

  def _manual_slice(self, trajectory, mask):
    indices = np.where(mask)[0]
    mask_size = mask.size

    def slice_time_major(struct):
      def slicer(arr):
        arr_np = np.asarray(arr)
        if arr_np.ndim == 0:
          return arr_np
        if arr_np.ndim == 1:
          return np.take(arr_np, indices, axis=0)
        return np.take(arr_np, indices, axis=1)

      return utils.map_single_structure(slicer, struct)

    def slice_batch_major(struct):
      def slicer(arr):
        arr_np = np.asarray(arr)
        if arr_np.ndim == 0:
          return arr_np
        return np.take(arr_np, indices, axis=0)

      return utils.map_single_structure(slicer, struct)

    actions = SampleOutputs(
        controller_state=slice_time_major(trajectory.actions.controller_state),
        logits=slice_time_major(trajectory.actions.logits),
    )
    delayed = [
        SampleOutputs(
            controller_state=slice_batch_major(sample.controller_state),
            logits=slice_batch_major(sample.logits),
        )
        for sample in trajectory.delayed_actions
    ]

    return trajectory._replace(
        states=slice_time_major(trajectory.states),
        name=slice_time_major(trajectory.name),
        actions=actions,
        rewards=slice_time_major(trajectory.rewards),
        is_resetting=slice_time_major(trajectory.is_resetting),
        initial_state=slice_batch_major(trajectory.initial_state),
        delayed_actions=delayed,
        active_mask=np.ones(indices.size, dtype=np.bool_),
    )

  def _build_test_trajectory(self):
    mask = np.array([True, False, True, False, True, True], dtype=np.bool_)
    base = run_lib.dummy_trajectory(self.policy, unroll_length=4, batch_size=mask.size)

    actions = SampleOutputs(
        controller_state=self._modify_time_major(base.actions.controller_state, mask),
        logits=self._modify_time_major(base.actions.logits, mask),
    )
    delayed_actions = [
        SampleOutputs(
            controller_state=self._modify_batch_major(sample.controller_state, mask),
            logits=self._modify_batch_major(sample.logits, mask),
        )
        for sample in base.delayed_actions
    ]

    trajectory = base._replace(
        states=self._modify_time_major(base.states, mask),
        name=self._modify_time_major(base.name, mask),
        actions=actions,
        rewards=self._modify_time_major(base.rewards, mask),
        is_resetting=self._modify_time_major(base.is_resetting, mask),
        initial_state=self._modify_batch_major(base.initial_state, mask),
        delayed_actions=delayed_actions,
        active_mask=mask,
    )

    states = trajectory.states
    is_teams = np.array(states.is_teams)
    singles_indices = np.array([0, 2])
    doubles_indices = np.array([4, 5])
    is_teams[:, singles_indices] = False
    is_teams[:, doubles_indices] = True
    states = states._replace(is_teams=is_teams)
    trajectory = trajectory._replace(states=states)

    return trajectory, mask

  def _clone_state(self, state: learner_lib.LearnerState) -> learner_lib.LearnerState:
    return learner_lib.LearnerState(
        teacher=tf.nest.map_structure(tf.identity, state.teacher),
        value_function=tf.nest.map_structure(tf.identity, state.value_function),
    )

  def test_unroll_and_ppo_grads_ignore_inactive_slots(self):
    trajectory, mask = self._build_test_trajectory()
    manual = self._manual_slice(trajectory, mask)

    active_count = int(mask.sum())
    initial_state = self.learner.initial_state(active_count)

    manual_state = self._clone_state(initial_state)
    masked_state = self._clone_state(initial_state)

    manual_outputs, manual_final = self.learner.unroll(manual, manual_state)
    masked_outputs, masked_final = self.learner.unroll(trajectory, masked_state)

    _assert_nested_allclose(self, manual_outputs, masked_outputs)
    _assert_nested_allclose(self, manual_final, masked_final)

    manual_grads, manual_metrics = self.learner.ppo_grads(manual_outputs, manual)
    masked_grads, masked_metrics = self.learner.ppo_grads(masked_outputs, trajectory)

    self.assertEqual(len(manual_grads), len(masked_grads))
    for grad_a, grad_b in zip(manual_grads, masked_grads):
      np.testing.assert_allclose(_to_numpy(grad_a), _to_numpy(grad_b))

    _assert_nested_allclose(self, manual_metrics, masked_metrics)

  def test_per_mode_metrics_summary(self):
    trajectory, mask = self._build_test_trajectory()
    active_count = int(mask.sum())
    initial_state = self.learner.initial_state(active_count)

    _, metrics = self.learner.ppo([trajectory], initial_state, num_epochs=0)

    per_mode = metrics['per_mode']
    totals = per_mode['totals']
    self.assertEqual(totals['active_columns'], 4)
    self.assertAlmostEqual(totals['singles_ratio'], 0.5, places=3)

    for mode in ['singles', 'doubles']:
      stats = per_mode[mode]
      self.assertGreater(stats['reward']['count'], 0)
      self.assertIn('mean', stats['actor_kl'])
      self.assertIn('mean', stats['teacher_kl'])
      self.assertIn('mean', stats['uev'])


if __name__ == '__main__':
  unittest.main()
