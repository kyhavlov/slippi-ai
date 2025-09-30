import types
import unittest
from unittest import mock

from slippi_db import parse_libmelee


class ParseLibmeleeGameTest(unittest.TestCase):

  def test_is_teams_false_for_singles(self):
    TestPlayer = types.SimpleNamespace  # placeholder replaced below
    game_state = types.SimpleNamespace(
        stage=types.SimpleNamespace(value=1),
        frame=0,
        players={1: object(), 2: object()},
    )

    fake_player = mock.Mock()
    fake_player._replace = mock.Mock(return_value='player_with_flag')

    with (
        mock.patch.object(parse_libmelee, 'get_player', return_value=fake_player),
        mock.patch.object(parse_libmelee, 'Game', return_value='game') as mock_game_ctor,
    ):
      result = parse_libmelee.get_game(game_state, ports=(1, 2))

    self.assertEqual(result, 'game')
    self.assertFalse(mock_game_ctor.call_args.kwargs['is_teams'])

  def test_is_teams_true_for_doubles(self):
    fake_player = mock.Mock()
    fake_player._replace = mock.Mock(return_value='player_with_flag')
    game_state = types.SimpleNamespace(
        stage=types.SimpleNamespace(value=1),
        frame=0,
        players={port: object() for port in range(1, 5)},
    )

    with (
        mock.patch.object(parse_libmelee, 'get_player', return_value=fake_player),
        mock.patch.object(parse_libmelee, 'Game', return_value='game') as mock_game_ctor,
    ):
      result = parse_libmelee.get_game(game_state, ports=(1, 2, 3, 4))

    self.assertEqual(result, 'game')
    self.assertTrue(mock_game_ctor.call_args.kwargs['is_teams'])


if __name__ == '__main__':
  unittest.main()
