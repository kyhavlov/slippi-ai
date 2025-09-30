import collections
import types
import unittest
from unittest import mock

import numpy as np

from slippi_ai import evaluators
from slippi_ai import envs
from slippi_ai.controller_heads import SampleOutputs


class RolloutActiveMaskTest(unittest.TestCase):

  def _build_fake_outputs(self, num_envs: int) -> list[envs.EnvOutput]:
    gamestates = {
        port: np.stack([
            np.full((num_envs,), 10 * port + i, dtype=np.int32)
            for i in range(2)
        ])
        for port in range(1, 5)
    }
    # gamestates is [time, batch]; env outputs should expose time slices.
    first = {
        port: gamestates[port][0]
        for port in gamestates
    }
    second = {
        port: gamestates[port][1]
        for port in gamestates
    }
    active = {
        1: np.array([True, True], dtype=np.bool_),
        2: np.array([False, True], dtype=np.bool_),
        3: np.array([True, True], dtype=np.bool_),
        4: np.array([False, True], dtype=np.bool_),
    }
    needs_reset = np.full((num_envs,), False, dtype=np.bool_)
    return [
        envs.EnvOutput(gamestates=first, needs_reset=needs_reset, active=active),
        envs.EnvOutput(gamestates=second, needs_reset=needs_reset, active=active),
    ]

  def test_rollout_records_active_mask(self):
    num_envs = 2

    class FakeAgent:

      def __init__(self, batch_size: int):
        self.delay = 1
        self.batch_steps = 1
        zeros = np.zeros((batch_size,), dtype=np.float32)
        self.dummy_sample_outputs = SampleOutputs(
            controller_state=zeros,
            logits=zeros,
        )
        self.hidden_state = np.zeros((batch_size,), dtype=np.float32)
        self.name_code = 7
        self.step_profiler = types.SimpleNamespace(mean_time=lambda: 0.0)
        self.embed_controller = types.SimpleNamespace(
            decode=lambda controller_state: controller_state)
        self._policy = types.SimpleNamespace(
            embed_game=types.SimpleNamespace(from_state=lambda states: states),
        )

      def start(self):
        pass

      def stop(self):
        pass

      def push(self, _game, _needs_reset):
        pass

      def pop(self):
        zeros = np.zeros((num_envs,), dtype=np.float32)
        return SampleOutputs(controller_state=zeros, logits=zeros)

      def peek_n(self, n: int):
        return [self.dummy_sample_outputs] * n

    class FakeEnv:

      outputs: list[envs.EnvOutput] = []

      def __init__(self, *_args, **_kwargs):
        self._outputs = collections.deque(self.outputs)
        self._push_calls = []
        self.num_steps = 1

      def stop(self):
        pass

      def push(self, controllers):
        self._push_calls.append(controllers)

      def pop(self):
        return self._outputs.popleft()

      def peek(self):
        return self._outputs[0]

    fake_outputs = self._build_fake_outputs(num_envs)
    FakeEnv.outputs = fake_outputs

    agent_kwargs = {
        port: dict(state=dict(config={}), name=['agent'] * num_envs)
        for port in range(1, 5)
    }
    dolphin_kwargs = dict(
        online_delay=0,
        players={port: mock.Mock() for port in range(1, 5)},
    )

    with (
        mock.patch.object(evaluators.eval_lib, 'build_delayed_agent',
                          side_effect=lambda **kwargs: FakeAgent(num_envs)),
        mock.patch.object(evaluators.eval_lib, 'update_character'),
        mock.patch.object(evaluators.reward, 'compute_rewards',
                          return_value=np.zeros((1, num_envs), dtype=np.float32)),
        mock.patch.object(evaluators.env_lib, 'BatchedEnvironment',
                          side_effect=lambda *args, **kwargs: FakeEnv()),
    ):
      worker = evaluators.RolloutWorker(
          agent_kwargs=agent_kwargs,
          dolphin_kwargs=dolphin_kwargs,
          num_envs=num_envs,
          env_kwargs=dict(singles_mask=[True, False], swap_ports=False),
          async_envs=False,
      )

      trajectories, _ = worker.rollout(num_steps=1)

    expected_masks = {
        1: np.array([True, True], dtype=np.bool_),
        2: np.array([False, True], dtype=np.bool_),
        3: np.array([True, True], dtype=np.bool_),
        4: np.array([False, True], dtype=np.bool_),
    }

    for port, trajectory in trajectories.items():
      self.assertTrue(np.array_equal(trajectory.active_mask, expected_masks[port]),
                      f'port {port} mask mismatch')


if __name__ == '__main__':
  unittest.main()
