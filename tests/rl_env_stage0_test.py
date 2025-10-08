import collections
import types
import unittest
from unittest import mock

import melee
import numpy as np

from slippi_db import parse_libmelee
from slippi_ai import envs, evaluators, utils
from slippi_ai.controller_heads import SampleOutputs


class Stage0BaselineTests(unittest.TestCase):

  def _make_gamestate(self, port_characters: dict[int, melee.Character]) -> melee.GameState:
    gamestate = melee.GameState()
    gamestate.frame = -123
    gamestate.stage = melee.Stage.BATTLEFIELD
    for port, character in port_characters.items():
      player = melee.PlayerState()
      player.character = character
      player.action = melee.Action.STANDING
      player.position = melee.Position(0, 0)
      player.percent = 0
      player.stock = 4
      gamestate.players[port] = player
    return gamestate

  def _make_player_state(self, character):
    player = melee.PlayerState()
    player.character = character
    player.action = melee.Action.STANDING
    player.position = melee.Position(0, 0)
    player.percent = 0
    player.stock = 4
    return player

  def test_get_game_singles_current_layout(self):
    gamestate = melee.GameState()
    gamestate.frame = 0
    gamestate.stage = melee.Stage.BATTLEFIELD
    gamestate.players[1] = self._make_player_state(melee.Character.FOX)
    gamestate.players[2] = self._make_player_state(melee.Character.FALCO)

    game = parse_libmelee.get_game(gamestate, ports=(1, 2), singles_opponent_port=2)

    self.assertFalse(game.is_teams)
    self.assertTrue(game.p1.is_dead)
    self.assertEqual(game.p2.character, melee.Character.FALCO.value)
    self.assertTrue(game.p3.is_dead)

  def test_environment_singles_spins_up_two_dolphins_today(self):
    players = {
        port: types.SimpleNamespace() for port in (1, 2, 3, 4)
    }

    dolphin_kwargs = dict(
        players=players,
        slippi_port=60000,
        slippi_port2=60001,
    )

    class DummyDolphin:
      instances = []

      def __init__(self, **kwargs):
        self.kwargs = kwargs
        DummyDolphin.instances.append(self)
        self.controllers = {1: mock.Mock(), 2: mock.Mock(), 3: mock.Mock(), 4: mock.Mock()}

      def step(self):
        state = melee.GameState()
        state.frame = -123
        state.stage = melee.Stage.BATTLEFIELD
        return state

      def stop(self):
        pass

    with mock.patch('slippi_ai.envs.dolphin.Dolphin', new=DummyDolphin), \
         mock.patch('slippi_ai.envs.send_controller', side_effect=lambda *args, **kwargs: None):
      env = envs.Environment(dolphin_kwargs=dolphin_kwargs, enable_singles=True)
      self.assertEqual(len(DummyDolphin.instances), 2)
      env.stop()

  def test_batched_environment_singles_currently_raises(self):
    players = {
        port: types.SimpleNamespace() for port in (1, 2, 3, 4)
    }
    dolphin_kwargs = dict(
        players=players,
        slippi_port=61000,
    )

    agent_names = [('', '') for _ in range(2)]

    class DummySafeEnvironment:
      def __init__(self, *args, **kwargs):
        pass
      def stop(self):
        pass

    with mock.patch('slippi_ai.envs.SafeEnvironment', new=DummySafeEnvironment):
      with self.assertRaises(IndexError):
        envs.BatchedEnvironment(
            num_envs=2,
            dolphin_kwargs=dolphin_kwargs,
            slippi_ports=[62000, 62001],
            num_retries=1,
            agent_names=agent_names,
            swap_ports=False,
            enable_singles=True,
        )

  def test_environment_current_state_returns_four_slots(self):
    players = {port: object() for port in (1, 2, 3, 4)}
    doubles_state = self._make_gamestate({
        1: melee.Character.FOX,
        2: melee.Character.FALCO,
        3: melee.Character.MARTH,
        4: melee.Character.SHEIK,
    })

    class DoublesDolphin:
      def __init__(self, **_):
        self.controllers = {port: mock.Mock() for port in (1, 2, 3, 4)}
      def step(self):
        return doubles_state
      def stop(self):
        pass

    with mock.patch('slippi_ai.envs.dolphin.Dolphin', new=DoublesDolphin), \
         mock.patch('slippi_ai.envs.send_controller', side_effect=lambda *args, **kwargs: None):
      env = envs.Environment(
          dolphin_kwargs=dict(players=players, slippi_port=63000),
          enable_singles=False,
      )
      output = env.current_state()
      env.stop()

    self.assertSetEqual(set(output.gamestates.keys()), {1, 2, 3, 4})
    self.assertTrue(output.gamestates[1].is_teams)
    self.assertEqual(output.gamestates[1].p0.character, melee.Character.FOX.value)
    self.assertFalse(output.gamestates[1].p2.is_dead)


    singles_players = {port: object() for port in (1, 2, 3, 4)}
    singles_states = iter([
        self._make_gamestate({1: melee.Character.FOX, 2: melee.Character.FALCO}),
        self._make_gamestate({1: melee.Character.MARTH, 2: melee.Character.SHEIK}),
    ])

    class SinglesDolphin:
      def __init__(self, **kwargs):
        self.controllers = {port: mock.Mock() for port in (1, 2)}
        self._state = next(singles_states)
      def step(self):
        return self._state
      def stop(self):
        pass

    with mock.patch('slippi_ai.envs.dolphin.Dolphin', new=SinglesDolphin), \
         mock.patch('slippi_ai.envs.send_controller', side_effect=lambda *args, **kwargs: None):
      env = envs.Environment(
          dolphin_kwargs=dict(
              players=singles_players,
              slippi_port=64000,
              slippi_port2=64001,
          ),
          enable_singles=True,
      )
      output = env.current_state()
      env.stop()

    self.assertSetEqual(set(output.gamestates.keys()), {1, 2, 3, 4})
    for port in (1, 2, 3, 4):
      game = output.gamestates[port]
      self.assertFalse(game.is_teams)
      self.assertTrue(game.p1.is_dead)
    game_first = output.gamestates[1]
    game_second = output.gamestates[3]
    self.assertFalse(game_first.p2.is_dead)
    self.assertTrue(game_first.p3.is_dead)
    self.assertTrue(game_second.p2.is_dead)
    self.assertFalse(game_second.p3.is_dead)

  def test_async_batched_environment_currently_missing_slippi_port2(self):
    captured = []

    class DummyAsyncEnv:
      def __init__(self, **kwargs):
        captured.append(kwargs)
      def begin_stop(self):
        pass
      def ensure_stopped(self):
        pass
      def stop(self):
        pass

    players = {port: object() for port in (1, 2, 3, 4)}
    dolphin_kwargs = dict(players=players, slippi_port=65000)

    ports = [65010, 65011, 65012]

    with mock.patch('slippi_ai.envs.AsyncEnvMP', new=DummyAsyncEnv), \
         mock.patch('slippi_ai.envs.utils.find_open_udp_ports', return_value=ports):
      env = envs.AsyncBatchedEnvironmentMP(
          num_envs=2,
          dolphin_kwargs=dolphin_kwargs,
          num_steps=0,
          inner_batch_size=1,
          swap_ports=False,
          enable_singles=True,
          agent_names=[('', '') for _ in range(2)],
      )
      env.stop()

    self.assertEqual(len(captured), 2)
    singles_kwargs = captured[0]
    self.assertTrue(singles_kwargs['enable_singles'])
    self.assertEqual(len(singles_kwargs['slippi_ports']), 1)
    self.assertNotIn('slippi_port2', singles_kwargs['dolphin_kwargs'])

  def test_rollout_worker_fake_env_smoke(self):
    def dummy_build_agent(**kwargs):
      class DummyAgent:
        delay = 0
        batch_steps = 1
        name_code = np.uint8(0)
        step_profiler = types.SimpleNamespace(mean_time=lambda: 0.0)

        def __init__(self):
          self.embed_controller = types.SimpleNamespace(
              decode=lambda controller_state: controller_state)
          self.dummy_sample_outputs = SampleOutputs(
              controller_state={'a': 0},
              logits={'a': 0},
          )
          self.hidden_state = 'hidden'

        def start(self):
          pass

        def stop(self):
          pass

        def pop(self):
          return self.dummy_sample_outputs

        def push(self, game, needs_reset):
          pass

        def peek_n(self, n):
          return [self.dummy_sample_outputs] * n

        @property
        def _policy(self):
          return types.SimpleNamespace(
              embed_game=types.SimpleNamespace(from_state=lambda state: state),
          )

      return DummyAgent()

    agent_kwargs = {
        port: dict(state=dict(config={}))
        for port in (1, 2, 3, 4)
    }

    dolphin_kwargs = dict(players={port: object() for port in (1, 2, 3, 4)}, online_delay=0)

    with mock.patch('slippi_ai.eval_lib.build_delayed_agent', side_effect=dummy_build_agent), \
         mock.patch('slippi_ai.eval_lib.update_character', side_effect=lambda *args, **kwargs: None):
      worker = evaluators.RolloutWorker(
          agent_kwargs=agent_kwargs,
          dolphin_kwargs=dolphin_kwargs,
          num_envs=2,
          async_envs=False,
          env_kwargs={},
          use_gpu=False,
          damage_ratio=0,
          use_fake_envs=True,
          use_ray_envs=False,
          agent_names=[('', '') for _ in range(2)],
      )

      trajectories, metrics = worker.rollout(num_steps=1)
      worker.stop()

    self.assertEqual(set(trajectories.keys()), {1, 2, 3, 4})
    for port in (1, 2, 3, 4):
      self.assertEqual(trajectories[port].states.stage.shape[0], 2)
    self.assertIn('timing', metrics)


if __name__ == '__main__':
  unittest.main()
