import unittest

import melee
import numpy as np

from slippi_ai import dolphin
from slippi_ai import sim_env


class SimEnvTest(unittest.TestCase):

  def test_current_state_and_step_match_existing_game_shape(self):
    env = sim_env.SimBatchedEnvironment(
        num_envs=3,
        players={
            1: dolphin.AI(melee.Character.FOX),
            2: dolphin.AI(melee.Character.FALCO),
        },
        length=8,
    )
    try:
      initial = env.current_state()
      self.assertEqual(set(initial.gamestates), {1, 2})
      self.assertEqual(initial.needs_reset.shape, (3,))
      self.assertTrue(np.all(initial.gamestates[1].p1.is_dead))
      self.assertTrue(np.all(initial.gamestates[1].p3.is_dead))
      self.assertTrue(np.all(initial.gamestates[1].p0.character == melee.Character.FOX.value))
      self.assertTrue(np.all(initial.gamestates[1].p2.character == melee.Character.FALCO.value))
      self.assertTrue(np.all(initial.gamestates[2].p0.character == melee.Character.FALCO.value))
      self.assertTrue(np.all(initial.gamestates[2].p2.character == melee.Character.FOX.value))

      controllers = {
          1: sim_env.neutral_controllers(3),
          2: sim_env.neutral_controllers(3),
      }
      output = env.step(controllers)
      self.assertEqual(output.gamestates[1].p0.x.shape, (3,))
      self.assertTrue(np.all(output.gamestates[1].stage == melee.Stage.FINAL_DESTINATION.value))
      self.assertTrue(np.all(output.gamestates[1].p0.controller.main_stick.x == 0.5))
      self.assertTrue(np.all(output.gamestates[2].p0.controller.main_stick.x == 0.5))
      self.assertTrue(np.all(output.gamestates[1].p0.action >= 0))
      self.assertGreaterEqual(env.cursor, 1)
    finally:
      env.stop()

  def test_push_pop_queue_and_partial_reset(self):
    env = sim_env.SimBatchedEnvironment(num_envs=2, length=4)
    try:
      first = env.pop()
      self.assertTrue(np.all(first.needs_reset))

      controllers = {
          1: sim_env.neutral_controllers(2),
          2: sim_env.neutral_controllers(2),
      }
      env.push(controllers)
      stepped = env.pop()
      self.assertFalse(np.any(stepped.needs_reset))

      reset = env.reset([1])
      self.assertFalse(reset.needs_reset[0])
      self.assertTrue(reset.needs_reset[1])
      self.assertEqual(reset.gamestates[1].p0.x.shape, (2,))
    finally:
      env.stop()

  def test_stage_assignment_cursor_wrap_and_controller_write(self):
    stages = [
        melee.Stage.FINAL_DESTINATION,
        melee.Stage.BATTLEFIELD,
        melee.Stage.YOSHIS_STORY,
    ]
    env = sim_env.SimBatchedEnvironment(num_envs=3, length=2, stage=stages)
    try:
      current = env.current_state()
      self.assertEqual(current.gamestates[1].stage.tolist(), [stage.value for stage in stages])

      controllers = {
          1: sim_env.neutral_controllers(3),
          2: sim_env.neutral_controllers(3),
      }
      controllers[1].main_stick.x[:] = [0.0, 0.25, 1.0]
      controllers[1].buttons.B[:] = [True, False, True]
      env.step(controllers)
      action = env.buffers.controller_action_view[0]['p'][:, 0]
      self.assertTrue(np.allclose(action['main_stick_x'], [0.0, 0.25, 1.0]))
      self.assertEqual(action['buttons']['B'].tolist(), [1, 0, 1])

      env.step(controllers)
      self.assertEqual(env.cursor, 2)
      env.step(controllers)
      self.assertEqual(env.cursor, 1)
    finally:
      env.stop()

  def test_max_frame_terminal_is_reported_separately(self):
    env = sim_env.SimBatchedEnvironment(num_envs=2, length=128, max_frame_id=0)
    try:
      controllers = {
          1: sim_env.neutral_controllers(2),
          2: sim_env.neutral_controllers(2),
      }
      output = None
      for _ in range(123):
        output = env.step(controllers)
      self.assertIsNotNone(output)
      self.assertTrue(np.all(output.needs_reset))
      terminal = env.last_step_info.terminal
      self.assertTrue(np.all(terminal['done'] == 1))
      self.assertTrue(np.all(terminal['max_frame_reached'] == 1))
      self.assertTrue(np.all(terminal['match_ended'] == 0))
    finally:
      env.stop()

  def test_packed_state_and_encoded_step(self):
    env = sim_env.SimBatchedEnvironment(
        num_envs=2,
        players={
            1: dolphin.AI(melee.Character.FOX),
            2: dolphin.AI(melee.Character.FALCO),
        },
        length=8,
    )
    try:
      state = env.current_packed_state(
          needs_reset=np.ones(2, dtype=np.bool_))
      self.assertEqual(state.needs_reset.shape, (4,))
      self.assertEqual(state.game.p0.x.shape, (4,))
      self.assertTrue(np.all(state.game.p0.character[:2] == melee.Character.FOX.value))
      self.assertTrue(np.all(state.game.p0.character[2:] == melee.Character.FALCO.value))
      self.assertTrue(np.all(state.game.p2.character[:2] == melee.Character.FALCO.value))
      self.assertTrue(np.all(state.game.p2.character[2:] == melee.Character.FOX.value))

      encoded = _neutral_encoded_controller(batch_size=4)
      needs_reset = env.step_encoded(
          encoded,
          axis_spacing=32,
          shoulder_spacing=4,
      )
      self.assertEqual(needs_reset.shape, (2,))
      next_state = env.current_packed_state(needs_reset=needs_reset)
      self.assertEqual(next_state.game.p0.x.shape, (4,))
    finally:
      env.stop()

def _neutral_encoded_controller(batch_size: int):
  shape = (int(batch_size),)
  return sim_env.Controller(
      main_stick=sim_env.Stick(
          x=np.full(shape, 16, dtype=np.uint8),
          y=np.full(shape, 16, dtype=np.uint8),
      ),
      c_stick=sim_env.Stick(
          x=np.full(shape, 16, dtype=np.uint8),
          y=np.full(shape, 16, dtype=np.uint8),
      ),
      shoulder=np.zeros(shape, dtype=np.uint8),
      buttons=sim_env.Buttons(**{
          name: np.zeros(shape, dtype=np.bool_)
          for name in sim_env.Buttons._fields
      }),
  )


if __name__ == '__main__':
  unittest.main()
