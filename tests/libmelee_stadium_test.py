import unittest

from melee import gamestate as gamestate_lib
from melee import console as console_lib


class LibmeleeStadiumTest(unittest.TestCase):

  def test_stadium_event_value_one_is_supported(self):
    self.assertEqual(
        gamestate_lib.StadiumTransformationEvent(1),
        gamestate_lib.StadiumTransformationEvent.UNKNOWN,
    )

  def test_unknown_stadium_event_preserves_previous_value(self):
    warned = set()
    fallback = gamestate_lib.StadiumTransformationEvent.FINISHED

    result = console_lib._coerce_enum_value(
        gamestate_lib.StadiumTransformationEvent,
        99,
        fallback=fallback,
        warning_name='stadium transformation event',
        warned_values=warned,
    )

    self.assertEqual(result, fallback)
    self.assertEqual(warned, {99})


if __name__ == '__main__':
  unittest.main()
