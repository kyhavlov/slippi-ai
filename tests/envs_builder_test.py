import unittest
from unittest import mock

import numpy as np

from slippi_ai import envs


class BatchedEnvironmentMaskTest(unittest.TestCase):

  def test_batched_environment_respects_singles_mask(self):
    singles_mask = [True, False]
    received_modes = []
    controller_logs = []

    class FakeSafeEnvironment:

      def __init__(self, dolphin_kwargs, num_retries, agent_names, *, swap_ports, is_singles):
        del dolphin_kwargs, num_retries, agent_names, swap_ports
        self._is_singles = is_singles
        self._index = len(received_modes)
        received_modes.append(is_singles)
        self.controllers_seen = []

      def current_state(self):
        base = 10 if self._is_singles else 20
        gamestates = {port: base + port for port in range(1, 5)}
        return envs.EnvOutput(gamestates=gamestates, needs_reset=False)

      def step(self, controllers):
        self.controllers_seen.append(dict(controllers))
        controller_logs.append(dict(controllers))
        return self.current_state()

      def multi_step(self, controllers):
        return [self.step(c) for c in controllers]

      def multi_current_state(self):
        return [self.current_state()]

      def stop(self):
        pass

    with (
        mock.patch.object(envs, 'SafeEnvironment', side_effect=lambda *args, **kwargs: FakeSafeEnvironment(*args, **kwargs)),
        mock.patch.object(envs.utils, 'find_open_udp_ports', return_value=[1000, 1001]),
    ):
      batched = envs.BatchedEnvironment(
          num_envs=2,
          dolphin_kwargs=dict(players={port: mock.Mock() for port in range(1, 5)}),
          agent_names=[('', '')] * 2,
          swap_ports=False,
          singles_mask=singles_mask,
      )

      output = batched.current_state()

      self.assertEqual(received_modes, [True, False])
      for port in range(1, 5):
        expected = np.array([10 + port, 20 + port])
        self.assertTrue(np.array_equal(output.gamestates[port], expected), f'port {port}')
      self.assertTrue(np.array_equal(output.needs_reset, np.array([False, False])))

      controllers = {
          port: np.array([port * 100 + 1, port * 100 + 2], dtype=np.int32)
          for port in range(1, 5)
      }
      batched.step(controllers)

      self.assertEqual(len(controller_logs), 2)
      env0_controllers = controller_logs[0]
      env1_controllers = controller_logs[1]
      for port in range(1, 5):
        self.assertEqual(env0_controllers[port], port * 100 + 1, f'env0 port {port}')
        self.assertEqual(env1_controllers[port], port * 100 + 2, f'env1 port {port}')

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
