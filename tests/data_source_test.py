import unittest
from unittest import mock

import numpy as np

from slippi_ai import data
from slippi_ai import types


def _dummy_game(length: int = 32) -> types.Game:
  def zeros(dtype):
    return np.zeros(length, dtype=dtype)

  def bools():
    return zeros(np.bool_)

  buttons = types.Buttons(*(bools() for _ in types.Buttons._fields))
  stick = lambda: types.Stick(x=zeros(np.float32), y=zeros(np.float32))
  controller = types.Controller(
      main_stick=stick(),
      c_stick=stick(),
      shoulder=zeros(np.float32),
      buttons=buttons,
  )
  nana = types.Nana(
      exists=bools(),
      percent=zeros(np.uint16),
      facing=bools(),
      x=zeros(np.float32),
      y=zeros(np.float32),
      action=zeros(np.uint16),
      invulnerable=bools(),
      character=zeros(np.uint8),
      jumps_left=zeros(np.uint8),
      shield_strength=zeros(np.float32),
      on_ground=bools(),
  )

  def player():
    return types.Player(
        percent=zeros(np.uint16),
        facing=bools(),
        x=zeros(np.float32),
        y=zeros(np.float32),
        action=zeros(np.uint16),
        invulnerable=bools(),
        character=zeros(np.uint8),
        jumps_left=zeros(np.uint8),
        shield_strength=zeros(np.float32),
        on_ground=bools(),
        is_dead=bools(),
        stocks_left=zeros(np.uint8),
        controller=controller,
        nana=nana,
    )

  randall = types.Randall(x=zeros(np.float32), y=zeros(np.float32))
  item = types.Item(
      exists=bools(),
      type=zeros(np.uint16),
      state=zeros(np.uint8),
      x=zeros(np.float32),
      y=zeros(np.float32),
  )
  items = types.Items(*[item for _ in range(types.MAX_ITEMS)])

  return types.Game(
      p0=player(),
      p1=player(),
      p2=player(),
      p3=player(),
      stage=zeros(np.uint8),
      randall_phase=zeros(np.float32),
      randall=randall,
      items=items,
      is_teams=bools(),
  )


class DataSourceTest(unittest.TestCase):

  def test_parallel_datasource_produces_batches(self):
    with mock.patch('slippi_ai.data.read_table', return_value=_dummy_game(64)):
      source = data.toy_data_source(batch_size=2, unroll_length=8, num_workers=2)
      batch, _ = next(source)
      self.assertEqual(batch.frames.state_action.state.stage.shape[0], 2)
      self.assertEqual(batch.frames.state_action.state.stage.shape[1], 8 + 1)
      source.close()

  def test_prefetch_iterator_matches_serial(self):
    with mock.patch('slippi_ai.data.read_table', return_value=_dummy_game(64)):
      serial_source = data.toy_data_source(batch_size=2, unroll_length=8)
      serial_batch, _ = next(serial_source)

    with mock.patch('slippi_ai.data.read_table', return_value=_dummy_game(64)):
      prefetched = data.PrefetchDataIterator(
          data.toy_data_source(batch_size=2, unroll_length=8), maxsize=2)
      prefetched_batch, _ = next(prefetched)

    np.testing.assert_array_equal(
        serial_batch.frames.state_action.state.stage,
        prefetched_batch.frames.state_action.state.stage)
    np.testing.assert_array_equal(
        serial_batch.frames.reward,
        prefetched_batch.frames.reward)
    np.testing.assert_array_equal(
        serial_batch.frames.is_resetting,
        prefetched_batch.frames.is_resetting)

    serial_source.close()
    prefetched.close()


if __name__ == '__main__':
  unittest.main()
