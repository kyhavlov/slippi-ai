import contextlib
import types
import unittest
from unittest import mock

import numpy as np
from melee import Action, Character, Stage

from slippi_ai import envs
from slippi_db import parse_libmelee


class _FakePlayerState:

  def __init__(self, *, alive: bool, character: Character):
    self.percent = 0.0 if alive else 999.0
    self.facing = True
    self.position = types.SimpleNamespace(x=0.0, y=0.0)
    self.action = types.SimpleNamespace(value=0)
    self.invulnerable = False
    self.character = character
    self.jumps_left = 2
    self.shield_strength = 60.0
    self.on_ground = True
    self.stock = 4 if alive else 0
    self.controller_state = types.SimpleNamespace(
        main_stick=(0.0, 0.0),
        c_stick=(0.0, 0.0),
        l_shoulder=0.0,
        button={button: False for button in parse_libmelee.LIBMELEE_BUTTONS.values()},
    )
    self.nana = None


class FakeDolphin:

  def __init__(self, players, **_kwargs):
    del players
    self.controllers = {port: mock.Mock(name=f'controller{port}') for port in range(1, 5)}

  def step(self):
    players = {
        1: _FakePlayerState(alive=True, character=Character.FOX),
        2: _FakePlayerState(alive=True, character=Character.FALCO),
        3: _FakePlayerState(alive=True, character=Character.MARTH),
        4: _FakePlayerState(alive=True, character=Character.SHEIK),
    }
    return types.SimpleNamespace(
        frame=0,
        stage=Stage.FINAL_DESTINATION,
        players=players,
    )

  def stop(self):
    pass


class BatchedEnvironmentMaskTest(unittest.TestCase):

  def _build_env(self, singles_mask):
    stack = contextlib.ExitStack()
    self.addCleanup(stack.close)
    num_envs = len(singles_mask)
    stack.enter_context(mock.patch.object(envs.dolphin, 'Dolphin', side_effect=FakeDolphin))
    stack.enter_context(mock.patch.object(envs.match_reporting, 'match_is_over', return_value=False))
    stack.enter_context(mock.patch.object(envs.utils, 'find_open_udp_ports', return_value=list(range(5000, 5000 + num_envs))))

    players = {port: mock.Mock(name=f'player{port}') for port in range(1, 5)}
    env = envs.BatchedEnvironment(
        num_envs=num_envs,
        dolphin_kwargs=dict(players=players),
        agent_names=[('', '')] * num_envs,
        singles_mask=singles_mask,
    )
    self.addCleanup(env.stop)
    return env

  def test_batched_environment_respects_singles_mask(self):
    env = self._build_env([True, False])
    output = env.current_state()
    np.testing.assert_array_equal(output.active[1], np.array([True, True]))
    np.testing.assert_array_equal(output.active[2], np.array([False, True]))
    np.testing.assert_array_equal(output.active[3], np.array([True, True]))
    np.testing.assert_array_equal(output.active[4], np.array([False, True]))
    np.testing.assert_array_equal(output.needs_reset, np.array([False, False]))

    games_port1 = output.gamestates[1]
    np.testing.assert_array_equal(games_port1.p2.is_dead, np.array([False, False]))
    np.testing.assert_array_equal(games_port1.p3.is_dead, np.array([True, False]))

    self.assertTrue(env._envs[0]._env._is_singles)
    self.assertFalse(env._envs[1]._env._is_singles)
    self.assertEqual(env._envs[0]._env._singles_opponent_slot, 2)

  def test_batched_environment_balances_singles_opponent_slots(self):
    env = self._build_env([True, True, False, False])
    output = env.current_state()
    np.testing.assert_array_equal(output.active[1], np.array([True, True, True, True]))
    np.testing.assert_array_equal(output.active[2], np.array([False, False, True, True]))
    np.testing.assert_array_equal(output.active[3], np.array([True, True, True, True]))
    np.testing.assert_array_equal(output.active[4], np.array([False, False, True, True]))

    games_port1 = output.gamestates[1]
    np.testing.assert_array_equal(games_port1.p2.is_dead, np.array([False, True, False, False]))
    np.testing.assert_array_equal(games_port1.p3.is_dead, np.array([True, False, False, False]))

    self.assertEqual(env._envs[0]._env._singles_opponent_slot, 2)
    self.assertEqual(env._envs[1]._env._singles_opponent_slot, 3)
    self.assertFalse(env._envs[2]._env._is_singles)
    self.assertFalse(env._envs[3]._env._is_singles)

  def test_async_environment_slices_mask_per_inner_batch(self):
    singles_mask = [True, False, True, False]
    received_masks = []

    class FakeAsyncEnv:

      def __init__(self, *args, singles_mask=None, **_kwargs):
        received_masks.append(list(singles_mask) if singles_mask is not None else None)

      def begin_stop(self):
        pass

      def ensure_stopped(self):
        pass

      def send(self, _controllers):
        pass

      def recv(self):
        raise RuntimeError('recv should not be called in this test')

    with (
        mock.patch.object(envs, 'AsyncEnvMP', side_effect=lambda *args, **kwargs: FakeAsyncEnv(*args, **kwargs)),
        mock.patch.object(envs.utils, 'find_open_udp_ports', return_value=[2000, 2001, 2002, 2003]),
    ):
      envs.AsyncBatchedEnvironmentMP(
          num_envs=4,
          dolphin_kwargs=dict(players={port: mock.Mock() for port in range(1, 5)}),
          num_steps=0,
          inner_batch_size=2,
          num_retries=1,
          swap_ports=False,
          singles_mask=singles_mask,
          agent_names=[('', '')] * 4,
      )

    self.assertEqual(received_masks, [[True, False], [True, False]])


if __name__ == '__main__':
  unittest.main()
