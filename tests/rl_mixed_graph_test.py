import contextlib
import dataclasses
import types
import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

try:
  tf.config.set_visible_devices([], 'GPU')
except (RuntimeError, ValueError):
  # Either no GPU present or devices already initialized for this process.
  pass

from melee import Character, Stage

from slippi_ai import evaluators, envs, saving, train_lib, utils
from slippi_ai.controller_heads import SampleOutputs
from slippi_ai.rl import learner as learner_lib, run_lib
from slippi_ai import value_function as vf_lib
from slippi_db import parse_libmelee
from slippi_ai.types import Buttons as ControllerButtons
from slippi_ai.types import Controller as ControllerStruct
from slippi_ai.types import Stick as ControllerStick


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
        button={lm_button: False for lm_button in parse_libmelee.LIBMELEE_BUTTONS.values()},
    )
    self.nana = None


class FakeDolphin:

  def __init__(self, players, slippi_port=None, **_kwargs):
    del players
    base_index = 0 if slippi_port is None else slippi_port - 6000
    self._base = float((base_index + 1) * 100)
    self.controllers = {port: mock.Mock(name=f'controller{port}') for port in range(1, 5)}

  def step(self):
    players = {
        1: _FakePlayerState(percent=self._base + 1, character=Character.FOX),
        2: _FakePlayerState(percent=self._base + 2, character=Character.FALCO),
        3: _FakePlayerState(percent=self._base + 3, character=Character.MARTH),
        4: _FakePlayerState(percent=self._base + 4, character=Character.SHEIK),
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
    zeros = np.zeros((batch_size,), dtype=np.float32)
    self.dummy_sample_outputs = self._make_sample(value=0.0)
    self.embed_controller = types.SimpleNamespace(
        decode=lambda controller_state: utils.map_single_structure(
            lambda arr: arr.copy(), controller_state),
    )
    self._policy = types.SimpleNamespace(
        embed_game=types.SimpleNamespace(from_state=lambda states: states),
    )
    self.hidden_state = ()
    self.step_profiler = types.SimpleNamespace(mean_time=lambda: 0.0)
    self.name_code = np.uint8(port)

  def push(self, game, needs_reset):
    del game, needs_reset

  def pop(self):
    return self._make_sample(value=float(self.port))

  def peek_n(self, n: int):
    return [self.dummy_sample_outputs] * n

  def start(self):
    pass

  def stop(self):
    pass

  def _make_sample(self, value: float) -> SampleOutputs:
    discrete = np.full((self.batch_size,), int(value) % 32, dtype=np.uint8)
    bool_array = np.zeros((self.batch_size,), dtype=np.bool_)

    controller_state = ControllerStruct(
        main_stick=ControllerStick(x=discrete.copy(), y=discrete.copy()),
        c_stick=ControllerStick(x=discrete.copy(), y=discrete.copy()),
        shoulder=discrete.copy(),
        buttons=ControllerButtons(
            A=bool_array.copy(),
            B=bool_array.copy(),
            X=bool_array.copy(),
            Y=bool_array.copy(),
            Z=bool_array.copy(),
            L=bool_array.copy(),
            R=bool_array.copy(),
            D_UP=bool_array.copy(),
        ),
    )

    float_val = float(value)
    axis_bins = 17
    shoulder_bins = 5
    logits = ControllerStruct(
        main_stick=ControllerStick(
            x=np.full((self.batch_size, axis_bins), float_val, dtype=np.float32),
            y=np.full((self.batch_size, axis_bins), float_val, dtype=np.float32)),
        c_stick=ControllerStick(
            x=np.full((self.batch_size, axis_bins), float_val, dtype=np.float32),
            y=np.full((self.batch_size, axis_bins), float_val, dtype=np.float32)),
        shoulder=np.full((self.batch_size, shoulder_bins), float_val, dtype=np.float32),
        buttons=ControllerButtons(
            A=np.full((self.batch_size, 1), float_val, dtype=np.float32),
            B=np.full((self.batch_size, 1), float_val, dtype=np.float32),
            X=np.full((self.batch_size, 1), float_val, dtype=np.float32),
            Y=np.full((self.batch_size, 1), float_val, dtype=np.float32),
            Z=np.full((self.batch_size, 1), float_val, dtype=np.float32),
            L=np.full((self.batch_size, 1), float_val, dtype=np.float32),
            R=np.full((self.batch_size, 1), float_val, dtype=np.float32),
            D_UP=np.full((self.batch_size, 1), float_val, dtype=np.float32),
        ),
    )
    return SampleOutputs(controller_state=controller_state, logits=logits)


class FakeAgentFactory:

  def __init__(self):
    self.instances: dict[int, FakeAgent] = {}

  def __call__(self, *, state, batch_size, **_kwargs):
    port = state['port']
    agent = FakeAgent(port=port, batch_size=batch_size)
    self.instances[port] = agent
    return agent


class RLMixedGraphTest(unittest.TestCase):

  def setUp(self):
    config = train_lib.Config()
    config.policy.delay = 1
    policy_config = dataclasses.asdict(config)
    self.policy = saving.policy_from_config(policy_config)
    self.policy.initialize_variables()
    self.teacher = saving.policy_from_config(policy_config)
    self.teacher.initialize_variables()
    learner_config = learner_lib.LearnerConfig()
    value_function = vf_lib.ValueFunction(
        network_config=config.value_function.network,
        embed_state_action=self.policy.embed_state_action,
    )
    self.learner = learner_lib.Learner(
        config=learner_config,
        policy=self.policy,
        teacher=self.teacher,
        value_function=value_function,
    )
    dummy = run_lib.dummy_trajectory(self.policy, 4, 4)
    self.learner.initialize(dummy)

  @contextlib.contextmanager
  def _worker_context(self, singles_mask, factory):
    num_envs = len(singles_mask)
    agent_kwargs = {
        port: dict(
            state={'config': {}, 'port': port},
            name=[f'agent{port}_env{i}' for i in range(num_envs)],
        )
        for port in range(1, 5)
    }
    players = {port: mock.Mock(name=f'player{port}') for port in range(1, 5)}
    dolphin_kwargs = dict(players=players, online_delay=0)

    with contextlib.ExitStack() as stack:
      stack.enter_context(mock.patch.object(envs.dolphin, 'Dolphin', side_effect=FakeDolphin))
      stack.enter_context(mock.patch.object(envs.match_reporting, 'match_is_over', return_value=False))
      stack.enter_context(mock.patch.object(envs.utils, 'find_open_udp_ports', return_value=[6000 + i for i in range(num_envs)]))
      stack.enter_context(mock.patch.object(evaluators.eval_lib, 'update_character'))
      stack.enter_context(mock.patch.object(envs, 'send_controller', lambda *_args, **_kwargs: None))
      stack.enter_context(mock.patch.object(evaluators.eval_lib, 'build_delayed_agent', side_effect=factory))

      worker = evaluators.RolloutWorker(
          agent_kwargs=agent_kwargs,
          dolphin_kwargs=dolphin_kwargs,
          env_kwargs=dict(singles_mask=singles_mask, swap_ports=False),
          num_envs=num_envs,
          async_envs=False,
          use_gpu=False,
          agent_names=[('', '')] * num_envs,
      )
      stack.enter_context(worker.run())
      yield worker

  def test_tf_function_unroll_handles_mixed_masks(self):
    singles_mask = [True, False, True, False]
    factory = FakeAgentFactory()
    with self._worker_context(singles_mask, factory) as worker:
      trajectories, _ = worker.rollout(num_steps=2)

    combined = evaluators.Trajectory.batch([trajectories[p] for p in [1, 2, 3, 4]])

    batch_size = int(combined.is_resetting.shape[1])
    initial_state = self.learner.initial_state(batch_size)

    outputs, final_state = self.learner.compiled_unroll(combined, initial_state)
    self.assertIsInstance(outputs.teacher.log_probs, tf.Tensor)
    self.assertIsNotNone(final_state)

    learner_state = self.learner.initial_state(batch_size)
    _, metrics = self.learner.ppo([combined], learner_state, num_epochs=0)
    self.assertIn('per_mode', metrics)


if __name__ == '__main__':
  unittest.main()
