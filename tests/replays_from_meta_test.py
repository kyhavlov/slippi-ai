import json
import tempfile
import unittest

from slippi_ai import data


def _player(character: int, code: str, team: int) -> dict:
  return {
      'character': character,
      'team': team,
      'netplay': {'code': code, 'name': ''},
      'name_tag': '',
  }


class ReplaysFromMetaTest(unittest.TestCase):

  def test_doubles_swap_includes_both_opponent_orders(self):
    meta_rows = [{
        'raw': 'Players/Test/replay.slp',
        'stage': 2,
        'slp_md5': '0' * 31 + 'a',
        'players': [
            _player(2, 'AAA#111', 0),
            _player(3, 'BBB#222', 0),
            _player(4, 'CCC#333', 1),
            _player(5, 'DDD#444', 1),
        ],
    }]

    with tempfile.TemporaryDirectory() as tmpdir:
      meta_path = f'{tmpdir}/meta.json'
      with open(meta_path, 'w') as f:
        json.dump(meta_rows, f)

      base_cfg = data.DatasetConfig(data_dir='data/Parsed', meta_path=meta_path, swap=False)
      swap_cfg = data.DatasetConfig(data_dir='data/Parsed', meta_path=meta_path, swap=True)

      base_replays = data.replays_from_meta(base_cfg)
      swap_replays = data.replays_from_meta(swap_cfg)

    self.assertEqual(4, len(base_replays))
    self.assertEqual(8, len(swap_replays))

    by_main = {}
    for info in swap_replays:
      by_main.setdefault(info.main_player_index, set()).add(info.opponent_order)

    self.assertEqual({(2, 3), (3, 2)}, by_main[0])
    self.assertEqual({(2, 3), (3, 2)}, by_main[1])
    self.assertEqual({(0, 1), (1, 0)}, by_main[2])
    self.assertEqual({(0, 1), (1, 0)}, by_main[3])

  def test_singles_swap_includes_both_opponent_slots(self):
    meta_rows = [{
        'raw': 'Players/Test/replay.slp',
        'stage': 2,
        'slp_md5': '0' * 31 + 'a',
        'players': [
            _player(2, 'AAA#111', 0),
            _player(3, 'BBB#222', 1),
        ],
    }]

    with tempfile.TemporaryDirectory() as tmpdir:
      meta_path = f'{tmpdir}/meta.json'
      with open(meta_path, 'w') as f:
        json.dump(meta_rows, f)

      base_cfg = data.DatasetConfig(data_dir='data/Parsed', meta_path=meta_path, swap=False)
      swap_cfg = data.DatasetConfig(data_dir='data/Parsed', meta_path=meta_path, swap=True)

      base_replays = data.replays_from_meta(base_cfg)
      swap_replays = data.replays_from_meta(swap_cfg)

    self.assertEqual(2, len(base_replays))
    self.assertEqual(4, len(swap_replays))

    by_main = {}
    for info in swap_replays:
      by_main.setdefault(info.main_player_index, set()).add(info.opponent_order)

    # Singles uses main_index in {0,2}; teammate is always empty at 1. The
    # opponent appears in one of the remaining ports and should be permuted.
    self.assertEqual({(2, 3), (3, 2)}, by_main[0])
    self.assertEqual({(0, 3), (3, 0)}, by_main[2])


if __name__ == '__main__':
  unittest.main(failfast=True)

