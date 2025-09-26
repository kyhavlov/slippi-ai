import pathlib
import unittest

import numpy as np
import peppi_py  # noqa: F401


if peppi_py is not None:  # pragma: no branch
  from slippi_ai import types
  from slippi_db import parse_peppi
else:  # pragma: no cover
  types = None
  parse_peppi = None


class ParsePeppiIntegrationTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls._replay_dir = pathlib.Path(__file__).parent / 'data' / 'replays'

  def _first_frame(self, filename: str):
    path = self._replay_dir / filename
    game_array = parse_peppi.get_slp(str(path))
    first_frame = game_array.slice(0, 1)
    return types.game_array_to_nt(first_frame)

  def test_parse_singles_replay(self):
    frame = self._first_frame('test_singles_game.slp')

    self.assertEqual(frame.stage[0], 24)
    self.assertEqual(frame.is_teams[0], 0)
    self.assertEqual(frame.p0.character[0], 1)
    self.assertEqual(frame.p1.character[0], 0)
    self.assertEqual(frame.p2.character[0], 22)
    self.assertEqual(frame.p3.character[0], 0)
    self.assertTrue(frame.p1.is_dead[0])
    self.assertFalse(frame.p2.is_dead[0])
    self.assertAlmostEqual(float(frame.p0.x[0]), -38.8, places=1)
    self.assertAlmostEqual(float(frame.p2.x[0]), 38.8, places=1)

  def test_parse_doubles_replay(self):
    frame = self._first_frame('test_doubles_game.slp')

    self.assertEqual(frame.stage[0], 18)
    self.assertEqual(frame.is_teams[0], 1)
    self.assertEqual(int(frame.p0.character[0]), 2)
    self.assertEqual(int(frame.p1.character[0]), 1)
    self.assertEqual(int(frame.p2.character[0]), 1)
    self.assertEqual(int(frame.p3.character[0]), 15)
    for port in (frame.p0, frame.p1, frame.p2, frame.p3):
      self.assertFalse(port.is_dead[0])

  def test_items_replay_contains_projectiles(self):
    path = self._replay_dir / 'test_items_game.slp'
    game_array = parse_peppi.get_slp(str(path))
    game = types.game_array_to_nt(game_array)

    active_slots = []
    for slot_name in types.Items._fields:
      slot = getattr(game.items, slot_name)
      if slot.exists.any():
        active_slots.append(slot)

    self.assertTrue(active_slots, 'Expected at least one item slot to be active')

    # Spot-check the first active slot for reasonable projectile data
    slot = active_slots[0]
    exists_mask = slot.exists.astype(bool)
    self.assertTrue(np.any(slot.type[exists_mask] > 0), msg='Active item has zero type id')

    xy = np.stack([slot.x[exists_mask], slot.y[exists_mask]], axis=0)
    self.assertTrue(np.any(np.abs(xy) > 0.0), msg='Active item has zero coordinates')

    first_idx = np.flatnonzero(exists_mask)[0]
    self.assertEqual(int(slot.type[first_idx]), 75)
    self.assertEqual(int(slot.state[first_idx]), 3)
    self.assertAlmostEqual(float(slot.x[first_idx]), -42.0, places=4)
    self.assertAlmostEqual(float(slot.y[first_idx]), 5.996767, places=4)


if __name__ == '__main__':
  unittest.main(failfast=True)
