import dataclasses
import unittest
from typing import Optional

from scripts import discordbot


@dataclasses.dataclass
class FakeGameState:
  frame: int
  finalized_frame: Optional[int]
  custom: dict = dataclasses.field(default_factory=dict)


class FinalizedDelayBufferTest(unittest.TestCase):

  def test_waits_for_delay_target_instead_of_latest_finalized(self):
    buffer = discordbot.FinalizedDelayBuffer(delay=3)

    outputs = []
    for frame in range(100, 106):
      outputs.extend(buffer.push(FakeGameState(frame, finalized_frame=105)))

    self.assertEqual([state.frame for state in outputs], [100, 101, 102])
    self.assertNotIn(105, [state.frame for state in outputs])

  def test_does_not_publish_speculative_target(self):
    buffer = discordbot.FinalizedDelayBuffer(delay=3)

    outputs = []
    for frame in range(100, 111):
      outputs.extend(buffer.push(FakeGameState(frame, finalized_frame=106)))

    self.assertEqual([state.frame for state in outputs], list(range(100, 107)))

  def test_publishes_backlog_in_order_when_finalization_jumps(self):
    buffer = discordbot.FinalizedDelayBuffer(delay=2)

    outputs = []
    for frame in range(100, 103):
      outputs.extend(buffer.push(FakeGameState(frame, finalized_frame=99)))
    outputs.extend(buffer.push(FakeGameState(103, finalized_frame=103)))

    self.assertEqual([state.frame for state in outputs], [100, 101])

  def test_marks_published_frames_as_delayed_finalized(self):
    buffer = discordbot.FinalizedDelayBuffer(delay=1)

    outputs = []
    for frame in range(100, 102):
      outputs.extend(buffer.push(FakeGameState(frame, finalized_frame=101)))

    self.assertEqual([state.frame for state in outputs], [100])
    self.assertTrue(outputs[0].custom['discordbot_delayed_finalized'])

  def test_clear_prevents_old_game_frames_from_publishing(self):
    buffer = discordbot.FinalizedDelayBuffer(delay=3)
    buffer.push(FakeGameState(100, finalized_frame=99))
    buffer.push(FakeGameState(101, finalized_frame=99))
    buffer.clear()

    outputs = []
    for frame in range(200, 204):
      outputs.extend(buffer.push(FakeGameState(frame, finalized_frame=203)))

    self.assertEqual([state.frame for state in outputs], [200])


if __name__ == '__main__':
  unittest.main(failfast=True)
