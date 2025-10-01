import contextlib
import types
import unittest
from unittest import mock

import numpy as np
from melee import Character, Stage

from slippi_ai import evaluators
from slippi_ai import envs
from slippi_ai.controller_heads import SampleOutputs
from slippi_db import parse_libmelee


class _FakePlayerState:

  def __init__(self, *, percent: float, character: Character):
    self.percent = percent
    self.facing = True
    self.position = types.SimpleNamespace(x=0.0, y=0.0)
    self.action = types.SimpleNamespace(value=0)
    self.invulnerable = False
    self.character = character
    self.jumps_left = 2
    self.shield_strength = 60.0
    self.on_ground = True
    self.stock = 4
    self.controller_state = types.SimpleNamespace(
        main_stick=(0.0, 0.0),
        c_stick=(0.0, 0.0),
        l_shoulder=0.0,
        button={button: False for button in parse_libmelee.LIBMELEE_BUTTONS.values()},
    )
    self.nana = None


class FakeDolphin:

  def __init__(self, players, slippi_port=None, **_kwargs):
    del players
    self.controllers = {port: mock.Mock(name=f'controller{port}') for port in range(1, 5)}
    self._env_index = 0 if slippi_port is None else slippi_port - 6000

  def step(self):
    base = float((self._env_index + 1) * 100)
    players = {
        1: _FakePlayerState(percent=base + 1, character=Character.FOX),
        2: _FakePlayerState(percent=base + 2, character=Character.FALCO),
        3: _FakePlayerState(percent=base + 3, character=Character.MARTH),
        4: _FakePlayerState(percent=base + 4, character=Character.SHEIK),
    }
    return types.SimpleNamespace(
        frame=0,
        stage=Stage.FINAL_DESTINATION,
        players=players,
    )

  def stop(self):
    pass


class FakeAgent:

  def __init__(self, *, port: int, batch_size: int):
    self.port = port
    self.batch_size = batch_size
    self.delay = 1
    self.batch_steps = 1
    self._pop_calls = 0
    zeros = np.zeros((batch_size,), dtype=np.float32)
    self.dummy_sample_outputs = SampleOutputs(
        controller_state=zeros.copy(),
        logits=zeros.copy(),
    )
    self.embed_controller = types.SimpleNamespace(
        decode=lambda controller_state: controller_state.copy(),
    )
    self._policy = types.SimpleNamespace(
        embed_game=types.SimpleNamespace(from_state=lambda states: states),
    )
    self.hidden_state = zeros.copy()
    self.step_profiler = types.SimpleNamespace(mean_time=lambda: 0.0)
    self.name_code = np.uint8(port)
    self.push_records: list[np.ndarray] = []

  def push(self, game, needs_reset):
    del needs_reset
    # Record percent values for verification.
    percent = np.array(game.p0.percent, copy=True)
    self.push_records.append(percent)

  def pop(self):
    values = np.full((self.batch_size,), self.port * 10 + self._pop_calls, dtype=np.float32)
    self._pop_calls += 1
    return SampleOutputs(controller_state=values, logits=values)

  def peek_n(self, n: int):
    return [self.dummy_sample_outputs] * n

  def start(self):
    pass

  def stop(self):
    pass

  def warmup(self):
    pass


class FakeAgentFactory:

  def __init__(self):
    self.instances: dict[int, FakeAgent] = {}
    self.batch_sizes: dict[int, int] = {}

  def __call__(self, *, state, batch_size, **_kwargs):
    port = state['port']
    agent = FakeAgent(port=port, batch_size=batch_size)
    self.instances[port] = agent
    self.batch_sizes[port] = batch_size
    return agent


class RolloutMaskingIntegrationTest(unittest.TestCase):

  def setUp(self):
    self.players = {port: mock.Mock(name=f'player{port}') for port in range(1, 5)}
    self.dolphin_kwargs = dict(players=self.players, online_delay=0)

  def _build_agent_kwargs(self, num_envs):
    return {
        port: dict(
            state={'config': {}, 'port': port},
            name=[f'agent{port}_env{i}' for i in range(num_envs)],
        )
        for port in range(1, 5)
    }

  @contextlib.contextmanager
  def _worker_context(self, *, singles_mask, factory, use_fake_envs=False):
    num_envs = len(singles_mask)
    agent_kwargs = self._build_agent_kwargs(num_envs)
    with contextlib.ExitStack() as stack:
      stack.enter_context(mock.patch.object(envs.dolphin, 'Dolphin', side_effect=FakeDolphin))
      stack.enter_context(mock.patch.object(envs.match_reporting, 'match_is_over', return_value=False))
      stack.enter_context(mock.patch.object(envs.utils, 'find_open_udp_ports', return_value=[6000 + i for i in range(num_envs)]))
      stack.enter_context(mock.patch.object(evaluators.eval_lib, 'update_character'))
      stack.enter_context(mock.patch.object(envs, 'send_controller', lambda *_args, **_kwargs: None))

      def _fake_rewards(states, _damage_ratio):
        time_dim = states.p0.percent.shape[0] - 1
        batch_dim = states.p0.percent.shape[1]
        return np.zeros((time_dim, batch_dim), dtype=np.float32)

      stack.enter_context(mock.patch.object(evaluators.reward, 'compute_rewards', side_effect=_fake_rewards))
      stack.enter_context(mock.patch.object(evaluators.eval_lib, 'build_delayed_agent', side_effect=factory))

      worker = evaluators.RolloutWorker(
          agent_kwargs=agent_kwargs,
          dolphin_kwargs=self.dolphin_kwargs,
          env_kwargs=dict(singles_mask=singles_mask, swap_ports=False),
          num_envs=num_envs,
          async_envs=False,
          use_gpu=False,
          use_fake_envs=use_fake_envs,
          agent_names=[('', '')] * num_envs,
      )
      yield worker

  def test_rollout_uses_active_masks_for_agents(self):
    singles_mask = [True, False]
    factory = FakeAgentFactory()
    with self._worker_context(singles_mask=singles_mask, factory=factory) as worker:
      self.assertEqual(factory.batch_sizes, {1: 2, 2: 1, 3: 2, 4: 1})

      trajectories, _ = worker.rollout(num_steps=1)

    # Agent push inputs reflect masking.
    np.testing.assert_array_equal(factory.instances[1].push_records[0], np.array([101., 201.]))
    np.testing.assert_array_equal(factory.instances[2].push_records[0], np.array([202.]))
    np.testing.assert_array_equal(factory.instances[3].push_records[0], np.array([102., 203.]))
    np.testing.assert_array_equal(factory.instances[4].push_records[0], np.array([204.]))

    # Inactive env slots stay neutral.
    port2_actions = trajectories[2].actions.controller_state
    self.assertTrue(np.all(port2_actions[:, 0] == 0.0))
    self.assertTrue(np.all(trajectories[4].actions.controller_state[:, 0] == 0.0))

    # Active env slots receive non-zero controller decisions on the latest frame.
    active_indices_p1 = np.where(trajectories[1].active_mask)[0]
    active_indices_p3 = np.where(trajectories[3].active_mask)[0]
    active_indices_p2 = np.where(trajectories[2].active_mask)[0]
    active_indices_p4 = np.where(trajectories[4].active_mask)[0]

    self.assertTrue(np.all(trajectories[1].actions.controller_state[-1, active_indices_p1] > 0.0))
    self.assertTrue(np.all(trajectories[3].actions.controller_state[-1, active_indices_p3] > 0.0))
    self.assertTrue(np.all(trajectories[2].actions.controller_state[-1, active_indices_p2] > 0.0))
    self.assertTrue(np.all(trajectories[4].actions.controller_state[-1, active_indices_p4] > 0.0))

  def test_all_singles_skips_placeholder_agents(self):
    singles_mask = [True, True]
    factory = FakeAgentFactory()
    with self._worker_context(singles_mask=singles_mask, factory=factory) as worker:
      self.assertEqual(factory.batch_sizes, {1: 2, 3: 2})
      self.assertNotIn(2, factory.instances)
      self.assertNotIn(4, factory.instances)

      trajectories, _ = worker.rollout(num_steps=1)

    # Placeholder ports remain zeroed.
    self.assertTrue(np.all(trajectories[2].actions.controller_state == 0.0))
    self.assertTrue(np.all(trajectories[4].actions.controller_state == 0.0))

    # Active ports still send real data for both environments.
    np.testing.assert_array_equal(factory.instances[1].push_records[0], np.array([101., 201.]))
    np.testing.assert_array_equal(factory.instances[3].push_records[0], np.array([102., 202.]))

    active_indices_p1 = np.where(trajectories[1].active_mask)[0]
    active_indices_p3 = np.where(trajectories[3].active_mask)[0]
    self.assertTrue(np.all(trajectories[1].actions.controller_state[-1, active_indices_p1] > 0.0))
    self.assertTrue(np.all(trajectories[3].actions.controller_state[-1, active_indices_p3] > 0.0))

  def test_mixed_batch_with_mid_batch_gaps(self):
    singles_mask = [True, False, True, False]
    factory = FakeAgentFactory()
    with self._worker_context(singles_mask=singles_mask, factory=factory) as worker:
      self.assertEqual(factory.batch_sizes, {1: 4, 2: 2, 3: 4, 4: 2})
      trajectories, _ = worker.rollout(num_steps=1)

    np.testing.assert_array_equal(factory.instances[1].push_records[0], np.array([101., 201., 301., 401.]))
    np.testing.assert_array_equal(factory.instances[2].push_records[0], np.array([202., 402.]))
    np.testing.assert_array_equal(factory.instances[3].push_records[0], np.array([102., 203., 302., 403.]))
    np.testing.assert_array_equal(factory.instances[4].push_records[0], np.array([204., 404.]))

    expected_port2_mask = np.array([False, True, False, True])
    expected_port4_mask = np.array([False, True, False, True])
    np.testing.assert_array_equal(trajectories[2].active_mask, expected_port2_mask)
    np.testing.assert_array_equal(trajectories[4].active_mask, expected_port4_mask)

    np.testing.assert_array_equal(
        trajectories[2].actions.controller_state[:, expected_port2_mask],
        trajectories[2].actions.controller_state[:, [1, 3]],
    )
    np.testing.assert_array_equal(
        trajectories[4].actions.controller_state[:, expected_port4_mask],
        trajectories[4].actions.controller_state[:, [1, 3]],
    )

    self.assertTrue(np.all(trajectories[2].actions.controller_state[:, ~expected_port2_mask] == 0.0))
    self.assertTrue(np.all(trajectories[4].actions.controller_state[:, ~expected_port4_mask] == 0.0))

    p1_active = np.where(trajectories[1].active_mask)[0]
    p3_active = np.where(trajectories[3].active_mask)[0]
    self.assertTrue(np.all(trajectories[1].actions.controller_state[-1, p1_active] > 0.0))
    self.assertTrue(np.all(trajectories[3].actions.controller_state[-1, p3_active] > 0.0))
    self.assertTrue(np.all(trajectories[2].actions.controller_state[-1, expected_port2_mask] > 0.0))
    self.assertTrue(np.all(trajectories[4].actions.controller_state[-1, expected_port4_mask] > 0.0))


if __name__ == '__main__':
  unittest.main()
