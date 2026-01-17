import unittest

import numpy as np

from melee import Character

from slippi_ai.envs import FakeBatchedEnvironment
from slippi_ai.rl.character_scheduler import CharacterScheduler, SlotSpec, AssignmentStatus


class MixedSchedulerTest(unittest.TestCase):

  def test_scheduler_supports_two_slots(self):
    allowlist = {
        Character.FOX: {"a"},
        Character.PEACH: {"a"},
    }
    slot_specs = [
        SlotSpec(env_id=0, port_index=0, name="a"),
        SlotSpec(env_id=0, port_index=1, name="a"),
    ]
    scheduler = CharacterScheduler(
        allowlist=allowlist,
        slot_specs=slot_specs,
        slots_per_env=2,
        team_ports=(),
        team_weight=0.0,
        rng_seed=0,
        max_candidates=32,
        max_slot_options=2,
        max_outstanding_per_env=2,
    )

    assignment = scheduler.request_assignment(env_id=0)
    self.assertEqual(assignment.env_id, 0)
    self.assertEqual(len(assignment.characters), 2)
    self.assertEqual(len(assignment.names), 2)

    scheduler.report_outcome(0, assignment.assignment_id, AssignmentStatus.COMPLETE)

  def test_fake_env_sets_is_teams_from_player_count(self):
    doubles = FakeBatchedEnvironment(num_envs=3, players=(1, 2, 3, 4)).current_state()
    singles = FakeBatchedEnvironment(num_envs=3, players=(1, 2)).current_state()

    self.assertTrue(np.all(doubles.gamestates[1].is_teams))
    self.assertTrue(np.all(doubles.gamestates[4].is_teams))
    self.assertTrue(np.all(~singles.gamestates[1].is_teams))
    self.assertTrue(np.all(~singles.gamestates[2].is_teams))


if __name__ == "__main__":
  unittest.main(failfast=True)

