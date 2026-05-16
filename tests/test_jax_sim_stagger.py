import unittest

from slippi_ai.sim_env import multiprocess_env
from slippi_ai.sim_env import jax_rollout


class _Counter:

  def __init__(self, value: int):
    self.value = value


class JaxSimStaggerTest(unittest.TestCase):

  def test_stagger_schedule_activates_one_worker_per_interval(self):
    self.assertEqual(
        jax_rollout.stagger_steps_per_worker(
            workers=4, total_steps=2400),
        600,
    )
    self.assertEqual(
        jax_rollout.initial_stagger_total_steps(
            workers=4, stagger_steps=600),
        2400,
    )

    cases = [
        (0, 1),
        (599, 1),
        (600, 2),
        (1199, 2),
        (1200, 3),
        (1799, 3),
        (1800, 4),
        (2399, 4),
        (2400, 4),
    ]
    for step, expected in cases:
      with self.subTest(step=step):
        self.assertEqual(
            jax_rollout.active_workers_for_stagger_step(
                step=step, workers=4, stagger_steps=600),
            expected,
        )

  def test_zero_stagger_keeps_all_workers_active(self):
    self.assertEqual(
        jax_rollout.initial_stagger_total_steps(
            workers=4, stagger_steps=0),
        0,
    )
    self.assertEqual(
        jax_rollout.active_workers_for_stagger_step(
            step=123, workers=4, stagger_steps=0),
        4,
    )

  def test_worker_active_gate_uses_shared_counter(self):
    counter = _Counter(2)

    self.assertTrue(multiprocess_env.worker_is_active(counter, 0))
    self.assertTrue(multiprocess_env.worker_is_active(counter, 1))
    self.assertFalse(multiprocess_env.worker_is_active(counter, 2))
    self.assertFalse(multiprocess_env.worker_is_active(counter, 3))

    counter.value = 4
    self.assertTrue(multiprocess_env.worker_is_active(counter, 3))
    self.assertTrue(multiprocess_env.worker_is_active(None, 999))

  def test_worker_measurement_gate_defaults_enabled(self):
    counter = _Counter(False)
    self.assertFalse(multiprocess_env.worker_measurement_enabled(counter))
    counter.value = True
    self.assertTrue(multiprocess_env.worker_measurement_enabled(counter))
    self.assertTrue(multiprocess_env.worker_measurement_enabled(None))

  def test_worker_timing_helpers_sum_latest_step(self):
    timings = [0.0] * 6

    multiprocess_env.write_step_timings(
        timings, worker_id=0, step_s=1.0, fill_s=2.0, obs_release_s=3.0)
    multiprocess_env.write_step_timings(
        timings, worker_id=1, step_s=4.0, fill_s=5.0, obs_release_s=6.0)

    self.assertEqual(
        multiprocess_env.sum_step_timings(timings, workers=2),
        (5.0, 7.0, 9.0),
    )

    multiprocess_env.write_step_timings(
        timings, worker_id=1, step_s=0.0, fill_s=0.0, obs_release_s=0.0)
    self.assertEqual(
        multiprocess_env.sum_step_timings(timings, workers=2),
        (1.0, 2.0, 3.0),
    )

  def test_jax_timing_summary_uses_expected_names(self):
    summary = jax_rollout.timing_summary({
        'update_total_s': 10.0,
        'trajectory_collect_total_s': 3.0,
        'learner_ppo_s': 7.0,
        'policy_sample_s': 1.25,
        'env_step_s': 0.75,
        'env_fill_s': 0.5,
        'obs_wait_s': 2.0,
        'action_copy_s': 0.25,
        'action_release_s': 0.125,
    })

    self.assertEqual(summary['total_s'], 10.0)
    self.assertEqual(summary['rollout_s'], 3.0)
    self.assertEqual(summary['learner_s'], 7.0)
    self.assertEqual(summary['agent_step_s'], 1.25)
    self.assertEqual(summary['env_step_s'], 0.75)
    self.assertEqual(summary['obs_wait_s'], 2.0)
    self.assertEqual(summary['action_copy_s'], 0.25)


if __name__ == '__main__':
  unittest.main()
