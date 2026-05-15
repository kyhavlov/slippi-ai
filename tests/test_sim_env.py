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
      self.assertTrue(np.all(initial.gamestates[1].p1.on_ground))
      self.assertTrue(np.all(initial.gamestates[1].p1.facing))
      self.assertTrue(np.all(initial.gamestates[1].p1.shield_strength == 60.0))
      self.assertTrue(np.all(initial.gamestates[1].p0.character == melee.Character.FOX.value))
      self.assertTrue(np.all(initial.gamestates[1].p2.character == melee.Character.FALCO.value))
      self.assertTrue(np.all(initial.gamestates[1].p0.jumps_left == 1))
      self.assertFalse(np.any(initial.gamestates[1].items.item_0.exists))
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

  def test_partial_reset_clears_observed_previous_controllers(self):
    env = sim_env.SimBatchedEnvironment(num_envs=2, length=8)
    try:
      controllers = {
          1: sim_env.neutral_controllers(2),
          2: sim_env.neutral_controllers(2),
      }
      controllers[1].main_stick.x[:] = [0.0, 1.0]
      controllers[2].main_stick.x[:] = [0.25, 0.75]
      env.step(controllers)

      env.reset([1])
      state = env.current_packed_state(
          needs_reset=np.array([False, True], dtype=np.bool_))

      self.assertTrue(np.allclose(state.game.p0.controller.main_stick.x, [
          0.0,
          0.5,
          0.25,
          0.5,
      ]))
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

  def test_default_supported_stage_pool_excludes_fountain(self):
    self.assertNotIn(melee.Stage.FOUNTAIN_OF_DREAMS, sim_env.supported_stages())

  def test_per_env_character_pairs(self):
    pairs = sim_env.balanced_fox_falco_pairs(4)
    env = sim_env.SimBatchedEnvironment(
        num_envs=4,
        length=8,
        character_pairs=pairs,
    )
    try:
      state = env.current_packed_state(
          needs_reset=np.ones(4, dtype=np.bool_))
      self.assertEqual(
          state.game.p0.character[:4].tolist(),
          [
              melee.Character.FOX.value,
              melee.Character.FALCO.value,
              melee.Character.FOX.value,
              melee.Character.FALCO.value,
          ],
      )
      self.assertEqual(
          state.game.p0.character[4:].tolist(),
          [
              melee.Character.FALCO.value,
              melee.Character.FOX.value,
              melee.Character.FALCO.value,
              melee.Character.FOX.value,
          ],
      )
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

  def test_slot_adapter_matches_libmelee_visible_conventions(self):
    slot = np.zeros(3, dtype=[
        ('present', np.bool_),
        ('stocks', np.uint8),
        ('percent', np.float32),
        ('facing', np.bool_),
        ('pos_x', np.float32),
        ('pos_y', np.float32),
        ('action_id', np.uint16),
        ('invulnerable', np.bool_),
        ('char_id', np.uint8),
        ('jumps_left', np.uint8),
        ('shield_hp', np.float32),
        ('on_ground', np.bool_),
    ])
    slot['present'] = True
    slot['stocks'] = 4
    slot['percent'] = [11.2, 11.8, 12.0]
    slot['char_id'] = melee.Character.FOX.value
    slot['jumps_left'] = [2, 2, 1]
    slot['on_ground'] = [True, False, False]
    slot['shield_hp'] = 60.0

    player = sim_env._player_from_slot(slot, sim_env.neutral_controllers(3))

    self.assertEqual(player.percent.tolist(), [11, 11, 12])
    self.assertEqual(player.jumps_left.tolist(), [2, 1, 1])

  def test_items_are_canonicalized_independent_of_backend_slot_order(self):
    items = np.zeros((1, len(sim_env.Items._fields)), dtype=[
        ('exists', np.bool_),
        ('type', np.uint16),
        ('state', np.uint8),
        ('pos_x', np.float32),
        ('pos_y', np.float32),
    ])
    items[0, :4]['exists'] = [True, True, True, False]
    items[0, :4]['type'] = [74, 54, 74, 0]
    items[0, :4]['state'] = [4, 0, 5, 0]
    items[0, :4]['pos_x'] = [-54.0, -32.0, 60.0, 0.0]
    items[0, :4]['pos_y'] = [20.0, 40.0, 20.0, 0.0]

    out = sim_env._items_from_frame(items)

    self.assertEqual(out.item_0.type.tolist(), [74])
    self.assertEqual(out.item_0.x.tolist(), [-54.0])
    self.assertEqual(out.item_1.type.tolist(), [74])
    self.assertEqual(out.item_1.x.tolist(), [60.0])
    self.assertEqual(out.item_2.type.tolist(), [54])
    self.assertEqual(out.item_2.x.tolist(), [-32.0])
    self.assertFalse(out.item_3.exists[0])

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
