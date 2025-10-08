import types
import unittest

import melee
import numpy as np
from unittest import mock

from slippi_ai import envs, utils, evaluators
from slippi_ai.controller_heads import SampleOutputs


class Stage1SchedulerTests(unittest.TestCase):

  def _make_stub_async_env(self, instantiated):

    class StubAsyncEnv:

      def __init__(self, *, singles_roles=None, enable_singles=False, num_envs=1, **_):
        self._num_envs = num_envs
        self._send_profiler = types.SimpleNamespace(mean_time=lambda: 0.0, num_calls=0)
        self._recv_profiler = types.SimpleNamespace(mean_time=lambda: 0.0, num_calls=0)
        self._needs_reset = np.zeros([num_envs], dtype=bool)
        self.controllers = []

        if enable_singles:
          role = singles_roles[0]
          self.mode = f'singles_{role}'
          if role == 'left':
            self._ports = (1, 2)
            self._values = (101, 102)
          else:
            self._ports = (3, 4)
            self._values = (151, 152)
        else:
          self.mode = 'doubles'
          self._ports = (1, 2, 3, 4)
          self._values = (11, 12, 13, 14)

        instantiated.append(self)

      def begin_stop(self):
        pass

      def ensure_stopped(self):
        pass

      def send(self, controllers):
        self.controllers.append(controllers)

      def recv(self):
        gamestates = {
            port: self._make_game(value)
            for port, value in zip(self._ports, self._values)
        }
        copied = {
            port: utils.map_nt(np.copy, game)
            for port, game in gamestates.items()
        }
        return envs.EnvOutput(gamestates=copied, needs_reset=self._needs_reset.copy())

      def stop(self):
        pass

      def _make_game(self, value):
        def fill(dtype):
          if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            val = value % info.max
          else:
            val = float(value)
          return np.full([self._num_envs], val, dtype=dtype)

        return utils.map_nt(fill, envs.reified_game)

    return StubAsyncEnv

  def _make_singles_state(self, port_characters: dict[int, melee.Character]) -> melee.GameState:
    game_state = melee.GameState()
    game_state.frame = 0
    game_state.stage = melee.Stage.BATTLEFIELD
    for port, character in port_characters.items():
      player = melee.PlayerState()
      player.character = character
      player.action = melee.Action.STANDING
      player.position = melee.Position(0, 0)
      player.percent = 0
      player.stock = 4
      game_state.players[port] = player
    return game_state

  def test_compute_layout_assigns_singles_chunks_first(self):
    layout = envs.compute_mixed_mode_layout(
        total_envs=24,
        singles_ratio=0.5,
        inner_batch_size=4,
    )

    self.assertEqual(layout.total_envs, 24)
    self.assertEqual(layout.inner_batch_size, 4)
    self.assertEqual(layout.num_single_groups, 12)
    self.assertEqual(layout.num_double_groups, 12)
    self.assertEqual(layout.num_single_chunks, 3)
    self.assertEqual(layout.num_double_chunks, 3)
    self.assertEqual(
        layout.chunk_modes,
        ['singles', 'singles', 'singles', 'doubles', 'doubles', 'doubles'],
    )
    self.assertEqual(
        layout.chunk_physical_sizes,
        [8, 8, 8, 4, 4, 4],
    )

  def test_compute_layout_requires_even_inner_batch_size_for_singles(self):
    with self.assertRaises(ValueError):
      envs.compute_mixed_mode_layout(
          total_envs=16,
          singles_ratio=0.25,
          inner_batch_size=3,
      )

  def test_compose_singles_group_populates_expected_port_slots(self):
    left_state = self._make_singles_state({
        1: melee.Character.FOX,
        2: melee.Character.FALCO,
    })
    right_state = self._make_singles_state({
        1: melee.Character.MARTH,
        2: melee.Character.SHEIK,
    })

    combined = envs.compose_singles_group(left_state, right_state)

    self.assertSetEqual(set(combined.keys()), {1, 2, 3, 4})

    game_port_1 = combined[1]
    self.assertFalse(game_port_1.is_teams)
    self.assertEqual(game_port_1.p0.character, melee.Character.FOX.value)
    self.assertTrue(game_port_1.p1.is_dead)
    self.assertEqual(game_port_1.p2.character, melee.Character.FALCO.value)
    self.assertTrue(game_port_1.p3.is_dead)

    game_port_2 = combined[2]
    self.assertEqual(game_port_2.p0.character, melee.Character.FALCO.value)
    self.assertTrue(game_port_1.p1.is_dead)
    self.assertEqual(game_port_2.p2.character, melee.Character.FOX.value)
    self.assertTrue(game_port_2.p3.is_dead)

    game_port_3 = combined[3]
    self.assertEqual(game_port_3.p0.character, melee.Character.MARTH.value)
    self.assertTrue(game_port_1.p1.is_dead)
    self.assertEqual(game_port_3.p3.character, melee.Character.SHEIK.value)
    self.assertTrue(game_port_3.p2.is_dead)

    game_port_4 = combined[4]
    self.assertEqual(game_port_4.p0.character, melee.Character.SHEIK.value)
    self.assertTrue(game_port_1.p1.is_dead)
    self.assertEqual(game_port_4.p3.character, melee.Character.MARTH.value)
    self.assertTrue(game_port_4.p2.is_dead)

  def test_async_batched_environment_merges_singles_and_doubles(self):
    instantiated = []
    StubAsyncEnv = self._make_stub_async_env(instantiated)

    players = {port: object() for port in (1, 2, 3, 4)}
    dolphin_kwargs = dict(players=players, slippi_port=70000)

    controllers = {
        1: np.array([1, 2, 3, 4]),
        2: np.array([5, 6, 7, 8]),
        3: np.array([9, 10, 11, 12]),
        4: np.array([13, 14, 15, 16]),
    }

    with mock.patch('slippi_ai.envs.AsyncEnvMP', new=StubAsyncEnv), \
         mock.patch('slippi_ai.envs.utils.find_open_udp_ports', return_value=list(range(71000, 71008))):
      env = envs.AsyncBatchedEnvironmentMP(
          num_envs=4,
          dolphin_kwargs=dolphin_kwargs,
          num_steps=0,
          inner_batch_size=2,
          swap_ports=False,
          enable_singles=True,
          singles_ratio=0.5,
          agent_names=[('', '') for _ in range(4)],
      )

      initial = env.pop()
      self.assertSetEqual(set(initial.gamestates.keys()), {1, 2, 3, 4})
      np.testing.assert_array_equal(
          initial.gamestates[1].stage,
          np.array([101, 101, 11, 11], dtype=np.uint8),
      )
      np.testing.assert_array_equal(
          initial.gamestates[3].stage,
          np.array([151, 151, 13, 13], dtype=np.uint8),
      )

      env.push(controllers)
      env.pop()
      env.stop()

    singles_envs = [stub for stub in instantiated if stub.mode.startswith('singles')]
    doubles_envs = [stub for stub in instantiated if stub.mode == 'doubles']
    self.assertEqual(len(singles_envs), 2)
    self.assertEqual(len(doubles_envs), 1)

    left_env = next(stub for stub in singles_envs if stub.mode == 'singles_left')
    right_env = next(stub for stub in singles_envs if stub.mode == 'singles_right')
    self.assertEqual(len(left_env.controllers), 1)
    self.assertEqual(len(right_env.controllers), 1)
    self.assertSetEqual(set(left_env.controllers[0]), {1, 2})
    self.assertSetEqual(set(right_env.controllers[0]), {3, 4})
    doubles_env = doubles_envs[0]
    self.assertSetEqual(set(doubles_env.controllers[0]), {1, 2, 3, 4})

  def test_rollout_worker_async_mixed_mode(self):
    instantiated = []
    StubAsyncEnv = self._make_stub_async_env(instantiated)

    def dummy_build_agent(**kwargs):
      class DummyAgent:
        delay = 0
        batch_steps = 1
        name_code = np.uint8(0)
        step_profiler = types.SimpleNamespace(mean_time=lambda: 0.0)

        def __init__(self):
          self.batch_size = kwargs['batch_size']
          self.embed_controller = types.SimpleNamespace(
              decode=lambda controller_state: controller_state)
          self.dummy_sample_outputs = SampleOutputs(
              controller_state=np.arange(self.batch_size),
              logits={'stub': np.arange(self.batch_size)},
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

    with mock.patch('slippi_ai.envs.AsyncEnvMP', new=StubAsyncEnv), \
         mock.patch('slippi_ai.envs.utils.find_open_udp_ports', return_value=list(range(72000, 72006))), \
         mock.patch('slippi_ai.eval_lib.build_delayed_agent', side_effect=dummy_build_agent), \
         mock.patch('slippi_ai.eval_lib.update_character', side_effect=lambda *_, **__: None):
      worker = evaluators.RolloutWorker(
          agent_kwargs=agent_kwargs,
          dolphin_kwargs=dolphin_kwargs,
          num_envs=4,
          async_envs=True,
          env_kwargs=dict(
              num_steps=0,
              inner_batch_size=2,
              enable_singles=True,
              singles_ratio=0.5,
              swap_ports=False,
          ),
          use_gpu=False,
          damage_ratio=0,
          use_fake_envs=False,
          use_ray_envs=False,
          agent_names=[('', '') for _ in range(4)],
      )

      trajectories, _ = worker.rollout(num_steps=1)
      worker.stop()

    singles_envs = [stub for stub in instantiated if stub.mode.startswith('singles')]
    doubles_env = next(stub for stub in instantiated if stub.mode == 'doubles')
    self.assertEqual(len(singles_envs), 2)
    self.assertEqual(len(doubles_env.controllers), 1)
    self.assertSetEqual(set(doubles_env.controllers[0]), {1, 2, 3, 4})

    stage_port1 = trajectories[1].states.stage
    stage_port3 = trajectories[3].states.stage
    np.testing.assert_array_equal(stage_port1[0], np.array([101, 101, 11, 11], dtype=np.uint8))
    np.testing.assert_array_equal(stage_port3[0], np.array([151, 151, 13, 13], dtype=np.uint8))


if __name__ == '__main__':
  unittest.main()
