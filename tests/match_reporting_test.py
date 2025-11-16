import json
import unittest
from unittest import mock

import melee

from slippi_ai import match_reporting


class MatchReportingTest(unittest.TestCase):

  def _make_gamestate(self, stocks):
    gamestate = melee.GameState()
    for port, stock in stocks.items():
      player = melee.PlayerState()
      player.stock = stock
      gamestate.players[port] = player
    return gamestate

  def test_submit_match_maps_ports_correctly(self):
    gamestate = self._make_gamestate({1: 1, 2: 0, 3: 0, 4: 1})
    names = ["A1", "B1", "B2", "A2"]
    characters = [melee.Character.FOX, melee.Character.FALCO,
                  melee.Character.MARTH, melee.Character.SHEIK]

    with mock.patch('requests.post') as post:
      match_reporting.submit_match(gamestate, names, characters)

    post.assert_called_once()
    payload = json.loads(post.call_args.kwargs['data'])
    self.assertEqual(payload['team1_player1']['name'], 'A1')
    self.assertEqual(payload['team1_player1']['character'], 'Fox')
    self.assertEqual(payload['team1_player2']['name'], 'A2')
    self.assertEqual(payload['team2_player1']['name'], 'B1')
    self.assertEqual(payload['winner'], 1)

  def test_submit_match_raises_on_missing_names(self):
    gamestate = self._make_gamestate({1: 1, 2: 0, 3: 0, 4: 1})
    with self.assertRaises(ValueError):
      match_reporting.submit_match(gamestate, ["A1"], [melee.Character.FOX])

  def test_submit_match_skips_on_no_winner(self):
    gamestate = self._make_gamestate({1: 0, 2: 0})
    with mock.patch('requests.post') as post:
      match_reporting.submit_match(gamestate, ["A1", "B1", "", ""],
                                   [melee.Character.FOX] * 4)
    post.assert_not_called()

  def test_submit_match_summary_handles_singles_mode(self):
    names = ["P1", "P2", "P3", "P4"]
    characters = [
        melee.Character.FOX,
        melee.Character.FALCO,
        melee.Character.MARTH,
        melee.Character.SHEIK,
    ]
    with mock.patch('requests.post') as post:
      match_reporting.submit_match_summary(names, characters, winner=1, is_teams=False)

    post.assert_called_once()
    payload = json.loads(post.call_args.kwargs['data'])
    self.assertEqual(payload['mode'], 'singles')
    self.assertEqual(payload['team1_player1']['name'], 'P1')
    self.assertEqual(payload['team2_player1']['name'], 'P2')
    self.assertEqual(payload['team1_player2']['name'], '')


if __name__ == '__main__':
  unittest.main()
