import dataclasses
import types
import unittest

import numpy as np

from slippi_ai.rl import run_lib
from slippi_ai import saving, train_lib
from slippi_ai.rl import learner as rl_learner
from slippi_ai import value_function as vf_lib


class _FakeActor:

  def start(self):
    pass

  def stop(self):
    pass

  def reset_env(self):
    pass

  def update_variables(self, variables):
    self.variables = variables


class Stage2LoggingTests(unittest.TestCase):

  def _build_learner(self):
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

    config = run_lib.Config()
    config.actor.rollout_length = 4
    config.actor.num_envs = 4
    config.actor.enable_singles = True
    config.actor.inner_batch_size = 2
    config.learner.reward.singles_scale = 1.5
    config.learner.reward.doubles_scale = 0.5
    config.learner.ppo.num_epochs = 1
    config.learner.ppo.num_batches = 1

    learner = rl_learner.Learner(
        config=config.learner,
        policy=policy,
        teacher=teacher,
        value_function=value_function,
    )

    init_traj = run_lib.dummy_trajectory(
        policy, config.actor.rollout_length, config.actor.num_envs * 4)
    learner.initialize(init_traj)

    return config, learner, policy

  def _make_trajectory(self, policy, rollout_length, num_envs, singles_envs):
    full_batch = num_envs * 4
    trajectory = run_lib.dummy_trajectory(policy, rollout_length, full_batch)
    states = trajectory.states

    states.is_teams[:, :] = True
    env_count = num_envs
    ports_per_env = 4
    for env_idx in range(env_count):
      is_single = env_idx < singles_envs
      for port_idx in range(ports_per_env):
        col = port_idx * env_count + env_idx
        states.is_teams[:, col] = not is_single

    time_len = states.p0.percent.shape[0]
    for col in range(states.p0.percent.shape[1]):
      scale = (col + 1) * 3
      ramp = np.arange(time_len, dtype=np.uint16)
      states.p0.percent[:, col] = ramp * scale
      states.p1.percent[:, col] = ramp * (scale // 2 + 1)
      states.p2.percent[:, col] = ramp * 2
      states.p3.percent[:, col] = ramp

    return trajectory

  def _run_manager(self, singles_slots: int):
    config, learner, policy = self._build_learner()
    test_traj = self._make_trajectory(
        policy,
        rollout_length=config.actor.rollout_length,
        num_envs=config.actor.num_envs,
        singles_envs=singles_slots,
    )

    manager = run_lib.LearnerManager(
        learner=learner,
        config=config,
        build_actor=lambda: _FakeActor(),
    )

    manager._hidden_state = learner.initial_state(test_traj.is_resetting.shape[1])

    manager._test_trajectory = test_traj
    timing = dict(
        env_pop=0.0,
        env_push=0.0,
        agent_pop={port: 0.0 for port in (1, 2, 3, 4)},
        agent_step={port: 0.0 for port in (1, 2, 3, 4)},
    )
    manager._test_actor_metrics = dict(timing=timing)

    def fake_rollout(self):
      return self._test_trajectory, dict(self._test_actor_metrics)

    manager._rollout = types.MethodType(fake_rollout, manager)

    try:
      _, metrics = manager.step(0)
    finally:
      manager.actor.stop()

    payload = dict(metrics)
    payload['actor'] = dict(metrics['actor'])
    run_lib.attach_mode_metrics(payload)

    return payload, metrics['learner']['mode']

  def test_mode_metrics_present_for_mixed_batch(self):
    payload, mode = self._run_manager(singles_slots=2)

    self.assertIn('singles', mode)
    self.assertIn('doubles', mode)
    self.assertIn('mode', payload)
    self.assertIn('singles', payload['mode'])
    self.assertEqual(payload['mode']['singles']['count'], 2)
    self.assertEqual(payload['mode']['singles']['ports'], 8)
    self.assertEqual(payload['mode'], mode)

  def test_mode_metrics_present_for_doubles_only(self):
    payload, mode = self._run_manager(singles_slots=0)

    self.assertNotIn('singles', mode)
    self.assertIn('doubles', mode)
    self.assertIn('mode', payload)
    self.assertNotIn('singles', payload['mode'])
    self.assertEqual(payload['mode']['doubles']['count'], 4)
    self.assertEqual(payload['mode']['doubles']['ports'], 16)
    self.assertEqual(payload['mode'], mode)


if __name__ == '__main__':
  unittest.main()
