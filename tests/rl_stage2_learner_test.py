import dataclasses
import numbers
import unittest

from slippi_ai import train_lib, saving
from slippi_ai.rl import run_lib, learner as rl_learner
from slippi_ai import value_function as vf_lib


class Stage2LearnerTests(unittest.TestCase):

  def _build_learner_and_trajectory(self, *, num_envs: int, singles_envs: int):
    """Returns (learner, trajectory) with controllable singles/doubles layout."""
    if singles_envs > num_envs:
      raise ValueError('singles_slots must be <= batch_size')

    # Use a tiny MLP policy for speed.
    train_config = train_lib.Config()
    train_config.network['name'] = 'mlp'
    train_config.network['mlp']['width'] = 8
    train_config.network['mlp']['depth'] = 1
    train_config.max_names = 2

    policy_state = dataclasses.asdict(train_config)
    policy = saving.policy_from_config(policy_state)
    policy.initialize_variables()

    teacher = saving.policy_from_config(policy_state)
    teacher.initialize_variables()

    value_function = vf_lib.ValueFunction(
        network_config=train_config.network,
        embed_state_action=policy.embed_state_action,
    )

    rl_config = run_lib.Config()
    rl_config.learner.reward.singles_scale = 1.5
    rl_config.learner.reward.doubles_scale = 0.5
    rl_config.learner.ppo.num_epochs = 1
    rl_config.learner.ppo.num_batches = 1

    learner = rl_learner.Learner(
        config=rl_config.learner,
        policy=policy,
        teacher=teacher,
        value_function=value_function,
    )

    ports_per_env = 4
    full_batch = num_envs * ports_per_env
    trajectory = run_lib.dummy_trajectory(
        policy, unroll_length=4, batch_size=full_batch)

    states = trajectory.states

    states.is_teams[:, :] = True
    num_envs = full_batch // ports_per_env
    for env_idx in range(num_envs):
      is_single = env_idx < singles_envs
      for port_idx in range(ports_per_env):
        col = port_idx * num_envs + env_idx
        states.is_teams[:, col] = not is_single

    # Craft simple percent trajectories so reward recomputation is non-zero.
    time_len = states.p0.percent.shape[0]
    for col in range(full_batch):
      scale = (col + 1) * 5
      for t in range(time_len):
        # Team 1 takes less damage than Team 2 for early slots.
        states.p0.percent[t, col] = scale * (t + 1)
        states.p1.percent[t, col] = (scale // 2) * (t + 1)
        states.p2.percent[t, col] = 2 * (t + 1)
        states.p3.percent[t, col] = 1 * (t + 1)

    return learner, trajectory

  def test_mixed_mode_metrics_expose_per_mode_stats(self):
    learner, trajectory = self._build_learner_and_trajectory(
        num_envs=4, singles_envs=2)

    learner.initialize(trajectory)
    initial_state = learner.initial_state(trajectory.is_resetting.shape[1])
    _, metrics = learner.ppo([trajectory], initial_state, num_epochs=1)

    self.assertIn('mode', metrics)
    mode_metrics = metrics['mode']
    self.assertIn('singles', mode_metrics)
    self.assertIn('doubles', mode_metrics)

    singles = mode_metrics['singles']
    doubles = mode_metrics['doubles']

    self.assertEqual(singles['count'], 2)
    self.assertEqual(singles['ports'], 8)
    self.assertEqual(doubles['count'], 2)
    self.assertEqual(doubles['ports'], 8)

    self.assertIn('reward', singles)
    self.assertIn('reward', doubles)
    self.assertNotAlmostEqual(
        singles['reward']['mean'], doubles['reward']['mean'], places=6)

    for key in ('ppo_objective', 'teacher_kl', 'actor_kl', 'entropy'):
      self.assertIn(key, singles)
      self.assertIn(key, doubles)
      self.assertIsInstance(singles[key]['mean'], numbers.Number)
      self.assertIsInstance(doubles[key]['mean'], numbers.Number)

  def test_doubles_only_metrics_emit_single_entry(self):
    learner, trajectory = self._build_learner_and_trajectory(
        num_envs=4, singles_envs=0)

    learner.initialize(trajectory)
    initial_state = learner.initial_state(trajectory.is_resetting.shape[1])
    _, metrics = learner.ppo([trajectory], initial_state, num_epochs=1)

    self.assertIn('mode', metrics)
    mode_metrics = metrics['mode']
    self.assertNotIn('singles', mode_metrics)
    self.assertIn('doubles', mode_metrics)

    doubles = mode_metrics['doubles']
    self.assertEqual(doubles['count'], 4)
    self.assertEqual(doubles['ports'], 16)
    self.assertIn('reward', doubles)
    self.assertIsInstance(doubles['reward']['mean'], numbers.Number)
    self.assertNotAlmostEqual(doubles['reward']['mean'], 0.0, places=6)


if __name__ == '__main__':
  unittest.main()
