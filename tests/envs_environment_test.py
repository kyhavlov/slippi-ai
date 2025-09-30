import collections
import types
import unittest
from unittest import mock

from melee import Stage

from slippi_ai import envs


class EnvironmentModeTest(unittest.TestCase):

  def setUp(self):
    self.players = {port: mock.Mock(name=f'player{port}') for port in range(1, 5)}
    self.dolphin_kwargs = dict(
        path='fake_path',
        iso='fake_iso',
        players=self.players,
    )

  def _mock_game_factory(self, *, is_teams: bool):
    TestPlayer = collections.namedtuple('TestPlayer', ['is_dead'])
    TestGame = collections.namedtuple('TestGame', ['is_teams', 'p0', 'p1', 'p2', 'p3'])

    def make_game(_state, _ports, singles_opponent_port=2):
      alive = TestPlayer(is_dead=False)
      dead = TestPlayer(is_dead=True)
      if singles_opponent_port == 2:
        p2, p3 = alive, dead
      else:
        p2, p3 = dead, alive
      return TestGame(
          is_teams=is_teams,
          p0=alive,
          p1=dead,
          p2=p2,
          p3=p3,
      )

    return make_game

  def _fake_game_state(self, *, num_players: int):
    return types.SimpleNamespace(
        frame=0,
        stage=Stage.FINAL_DESTINATION,
        players={
            port: types.SimpleNamespace(stock=4, character=port)
            for port in range(1, num_players + 1)
        },
    )

  def test_singles_environment_uses_one_dolphin_and_masks_placeholders(self):
    mock_game = self._mock_game_factory(is_teams=False)

    with (
        mock.patch.object(envs.dolphin, 'Dolphin') as mock_dolphin,
        mock.patch.object(envs, 'get_game', side_effect=mock_game) as mock_get_game,
        mock.patch.object(envs, 'send_controller') as mock_send_controller,
        mock.patch.object(envs.match_reporting, 'match_is_over', return_value=False),
    ):
      dolphin_instance = mock_dolphin.return_value
      dolphin_instance.controllers = {1: mock.Mock(), 2: mock.Mock()}
      dolphin_instance.step.return_value = self._fake_game_state(num_players=2)

      env = envs.Environment(
          self.dolphin_kwargs,
          agent_names=('ally', 'opponent'),
          swap_ports=False,
          is_singles=True,
      )

      self.assertEqual(mock_dolphin.call_count, 1)

      env_output = env.current_state()
      self.assertIn(env._singles_friendly_placeholder, env_output.gamestates)
      placeholder_game = env_output.gamestates[env._singles_friendly_placeholder]
      self.assertTrue(placeholder_game.p0.is_dead)
      self.assertFalse(env_output.gamestates[env._singles_friendly_port].is_teams)
      self.assertFalse(env_output.gamestates[env._singles_enemy_port].is_teams)

      controllers = {port: mock.Mock() for port in range(1, 5)}
      env._step(controllers)
      active_calls = [call.args[0] for call in mock_send_controller.call_args_list]
      self.assertEqual(len(active_calls), 2)
      self.assertCountEqual(active_calls, [dolphin_instance.controllers[1], dolphin_instance.controllers[2]])

      env.stop()
      self.assertGreaterEqual(mock_get_game.call_count, 2)

  def test_doubles_environment_routes_all_ports(self):
    mock_game = self._mock_game_factory(is_teams=True)

    with (
        mock.patch.object(envs.dolphin, 'Dolphin') as mock_dolphin,
        mock.patch.object(envs, 'get_game', side_effect=mock_game) as mock_get_game,
        mock.patch.object(envs, 'send_controller') as mock_send_controller,
        mock.patch.object(envs.match_reporting, 'match_is_over', return_value=False),
    ):
      dolphin_instance = mock_dolphin.return_value
      dolphin_instance.controllers = {port: mock.Mock() for port in range(1, 5)}
      dolphin_instance.step.return_value = self._fake_game_state(num_players=4)

      env = envs.Environment(
          self.dolphin_kwargs,
          agent_names=('ally', 'opponent'),
          swap_ports=False,
          is_singles=False,
      )

      self.assertEqual(mock_dolphin.call_count, 1)

      env_output = env.current_state()
      for port in range(1, 5):
        self.assertIn(port, env_output.gamestates)
        self.assertTrue(env_output.gamestates[port].is_teams)

      controllers = {port: mock.Mock() for port in range(1, 5)}
      env._step(controllers)
      active_calls = [call.args[0] for call in mock_send_controller.call_args_list]
      self.assertEqual(len(active_calls), 4)
      self.assertCountEqual(
          active_calls,
          [dolphin_instance.controllers[port] for port in range(1, 5)],
      )

      env.stop()
      self.assertGreaterEqual(mock_get_game.call_count, 4)


if __name__ == '__main__':
  unittest.main()
