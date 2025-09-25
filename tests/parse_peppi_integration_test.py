import pathlib
import unittest

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


if __name__ == '__main__':
  unittest.main(failfast=True)
