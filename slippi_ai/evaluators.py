"""Evaluates a policy."""

import collections
import contextlib
import typing as tp
import cProfile

import numpy as np
import ray

from slippi_ai import envs as env_lib
from slippi_ai import (
    embed,
    eval_lib,
    policies,
    reward,
    utils,
)
from slippi_ai.types import Game
from slippi_ai.controller_heads import SampleOutputs

Port = int
Timings = dict
Params = tp.Sequence[np.ndarray]

# Mimics data.Batch
class Trajectory(tp.NamedTuple):
  # The [T+1, ...] arrays overlap in time by 1.
  states: Game  # [T+1, B]
  name: np.ndarray  # [T+1, B]
  actions: SampleOutputs  # [T+1, B]
  rewards: np.ndarray  # [T, B]
  is_resetting: bool  # [T+1, B]
  initial_state: policies.RecurrentState  # [B]
  delayed_actions: list[SampleOutputs]  # [D, B]
  active_mask: np.ndarray  # [B]

  @classmethod
  def batch(cls, trajectories: list['Trajectory']) -> 'Trajectory':
    # TODO: test?
    batch_dims = Trajectory(
        states=1,
        name=1,
        actions=1,
        rewards=1,
        is_resetting=1,
        initial_state=0,
        delayed_actions=0,
        active_mask=0,
    )
    return utils.map_nt(
        lambda axis, *ts: utils.concat_nest_nt(ts, axis),
        batch_dims, *trajectories)


class RolloutWorker:

  def __init__(
      self,
      agent_kwargs: tp.Mapping[Port, dict],
      dolphin_kwargs: dict,
      num_envs: int,
      async_envs: bool = False,
      env_kwargs: dict = {},
      use_gpu: bool = False,
      damage_ratio: float = 0,  # For rewards.
      use_fake_envs: bool = False,
      use_ray_envs: bool = False,
      agent_names: list[tuple[str, str]] = [],
  ):
    print("use_gpu = ", use_gpu)
    self._num_envs = num_envs
    self._use_fake_envs = use_fake_envs
    self._env_kwargs = env_kwargs
    self._async_envs = async_envs
    self._use_ray_envs = use_ray_envs
    self._agent_names = agent_names
    self._agent_ports: list[int] = sorted(agent_kwargs)
    self._agent_kwargs = {port: dict(kwargs) for port, kwargs in agent_kwargs.items()}

    self._dolphin_kwargs = dolphin_kwargs.copy()
    for port, kwargs in agent_kwargs.items():
      eval_lib.update_character(
          self._dolphin_kwargs['players'][port],
          kwargs['state']['config'])

    self._build_env()
    initial_output = self._peek_initial_output()
    self._setup_port_activity(initial_output)

    self._build_agents(agent_kwargs, use_gpu)

    self._prev_agent_outputs = collections.deque()
    self._prev_agent_outputs.append(self._initial_agent_outputs())

    self._damage_ratio = damage_ratio

    self._env_push_profiler = utils.Profiler()
    self._validate_buffer_sizes()
    self.env_runahead = self._compute_env_runahead()
    for _ in range(self.env_runahead):
      self._push_actions()

  def _build_env(self):
    if self._use_ray_envs:
      self._env = env_lib.RayBatchedEnvironment(
          self._num_envs, self._dolphin_kwargs, **self._env_kwargs)
    elif self._use_fake_envs:
      self._env = env_lib.FakeBatchedEnvironment(
          self._num_envs, players=self._agent_ports)
    else:
      if not self._async_envs:
        env_class = env_lib.BatchedEnvironment
      else:
        env_class = env_lib.AsyncBatchedEnvironmentMP
      self._env = env_class(
          self._num_envs, self._dolphin_kwargs, agent_names=self._agent_names, **self._env_kwargs)

  def reset_env(self):
    self._env.stop()
    self._build_env()
    initial_output = self._peek_initial_output()
    self._validate_port_activity(initial_output)

    # Start env runahead
    # TODO: properly reset the agents with dummy actions instead of reusing
    # delayed actions from the previous rollout.
    assert len(self._prev_agent_outputs) == 1 + self.env_runahead
    for agent_outputs in list(self._prev_agent_outputs)[1:]:
      decoded_actions = {}
      for port, output in agent_outputs.items():
        if self._active_counts[port] == 0:
          continue
        decoder = self._controller_decoders[port]
        decoded_actions[port] = decoder.decode(output.controller_state)
      with self._env_push_profiler:
        self._env.push(decoded_actions)


  def _push_actions(self):
    """Pop actions from the agents and push them to the environment."""
    outputs: dict[Port, SampleOutputs] = {}
    for port in self._agent_ports:
      agent = self._agents[port]
      if agent is None:
        outputs[port] = self._copy_sample(self._output_templates[port])
        continue
      profiler = self._agent_profilers[port]
      if profiler is None:
        active_output = agent.pop()
      else:
        with profiler:
          active_output = agent.pop()
      outputs[port] = self._scatter_sample(port, active_output)

    self._prev_agent_outputs.append({
        port: self._copy_sample(sample) for port, sample in outputs.items()
    })

    decoded_actions = {}
    for port, action in outputs.items():
      if self._active_counts[port] == 0:
        continue
      decoder = self._controller_decoders[port]
      decoded_actions[port] = decoder.decode(action.controller_state)
    with self._env_push_profiler:
      self._env.push(decoded_actions)

  def _peek_initial_output(self) -> env_lib.EnvOutput:
    if hasattr(self._env, 'peek'):
      return self._env.peek()
    return self._env.current_state()

  def _setup_port_activity(self, env_output: env_lib.EnvOutput) -> None:
    self._active_masks: dict[Port, np.ndarray] = {}
    self._active_indices: dict[Port, np.ndarray] = {}
    self._inactive_indices: dict[Port, np.ndarray] = {}
    self._active_counts: dict[Port, int] = {}
    self._mask_signature: dict[Port, np.ndarray] = {}
    for port in self._agent_ports:
      mask = self._normalize_mask(env_output.active[port])
      indices = np.nonzero(mask)[0]
      self._active_masks[port] = mask
      self._active_indices[port] = indices
      self._inactive_indices[port] = np.nonzero(~mask)[0]
      self._active_counts[port] = int(indices.size)
      self._mask_signature[port] = mask.copy()

  def get_active_mask(self, port: Port) -> np.ndarray:
    """Returns a copy of the active mask for the requested logical port."""
    return self._mask_signature[port].copy()

  def get_flat_active_mask(self, ports: tp.Sequence[Port]) -> np.ndarray:
    """Returns the concatenated active mask for the provided port order."""
    masks = [self._mask_signature[port] for port in ports]
    if not masks:
      return np.zeros((0,), dtype=np.bool_)
    return np.concatenate([mask.copy() for mask in masks], axis=0)

  def _validate_port_activity(self, env_output: env_lib.EnvOutput) -> None:
    for port in self._agent_ports:
      mask = self._normalize_mask(env_output.active[port])
      if not np.array_equal(mask, self._mask_signature[port]):
        raise ValueError(f'Active mask changed for port {port}.')

  def _normalize_mask(self, mask_value: tp.Union[bool, np.ndarray]) -> np.ndarray:
    mask_array = np.asarray(mask_value, dtype=np.bool_)
    if mask_array.ndim == 0:
      mask_array = np.full((self._num_envs,), bool(mask_array), dtype=np.bool_)
    else:
      if mask_array.ndim != 1:
        raise ValueError(
            f'Active mask must be 1D; received shape {mask_array.shape}.')
      if mask_array.shape[0] != self._num_envs:
        raise ValueError(
            f'Active mask shape {mask_array.shape} does not match '
            f'num_envs {self._num_envs}')
    return mask_array

  def _build_agents(self, agent_kwargs: tp.Mapping[Port, dict], use_gpu: bool) -> None:
    self._agents: dict[Port, tp.Optional[eval_lib.DelayedAgent]] = {}
    self._agent_profilers: dict[Port, tp.Optional[utils.Profiler]] = {}
    self._controller_decoders: dict[Port, tp.Any] = {}

    reference_agent = None
    reference_sample = None
    reference_state = None

    for port in self._agent_ports:
      kwargs = dict(agent_kwargs[port])
      active_count = self._active_counts[port]
      indices = self._active_indices[port]
      names = kwargs.get('name')
      if isinstance(names, list):
        kwargs['name'] = [names[i] for i in indices]

      if active_count == 0:
        self._agents[port] = None
        self._agent_profilers[port] = None
        continue

      agent = eval_lib.build_delayed_agent(
          console_delay=self._dolphin_kwargs['online_delay'],
          batch_size=active_count,
          run_on_cpu=not use_gpu,
          **kwargs,
      )
      self._agents[port] = agent
      self._agent_profilers[port] = utils.Profiler()
      self._controller_decoders[port] = agent.embed_controller

      if reference_agent is None:
        reference_agent = agent
        reference_sample = self._copy_sample(agent.dummy_sample_outputs)
        reference_state = self._copy_structure(agent.hidden_state)

    if reference_agent is None:
      raise ValueError('At least one active port is required to build agents.')

    for port in self._agent_ports:
      self._agents.setdefault(port, None)
      self._agent_profilers.setdefault(port, None)
      self._controller_decoders.setdefault(port, reference_agent.embed_controller)

    self._reference_agent = reference_agent
    self._reference_sample = reference_sample
    self._reference_state = reference_state

    self._init_output_templates()

  def _init_output_templates(self) -> None:
    self._output_templates: dict[Port, SampleOutputs] = {}
    self._zero_states: dict[Port, tp.Any] = {}

    for port in self._agent_ports:
      if self._agents[port] is not None:
        sample_source = self._sample_to_numpy(self._agents[port].dummy_sample_outputs)
        state_source = self._structure_to_numpy(self._agents[port].hidden_state)
      else:
        sample_source = self._reference_sample
        state_source = self._reference_state
      self._output_templates[port] = self._zeros_like_sample(sample_source)
      self._zero_states[port] = self._zeros_like_state(state_source)

  def _initial_agent_outputs(self) -> dict[Port, SampleOutputs]:
    outputs: dict[Port, SampleOutputs] = {}
    for port in self._agent_ports:
      agent = self._agents[port]
      if agent is None or self._active_counts[port] == 0:
        outputs[port] = self._copy_sample(self._output_templates[port])
      else:
        outputs[port] = self._scatter_sample(port, agent.dummy_sample_outputs)
    return outputs

  def _validate_buffer_sizes(self) -> None:
    active_agents = [agent for agent in self._agents.values() if agent is not None]
    if not active_agents:
      return
    for agent in active_agents:
      slack = 1 + agent.delay
      max_agent_buffer = agent.batch_steps - 1
      max_env_buffer = self._env.num_steps - 1
      if max_agent_buffer + max_env_buffer >= slack:
        self._env.stop()
        raise ValueError(
            f'Agent and environment step buffer sizes are too large: '
            f'{max_agent_buffer} + {max_env_buffer} >= {slack}')

  def _compute_env_runahead(self) -> int:
    active_agents = [agent for agent in self._agents.values() if agent is not None]
    if not active_agents:
      return 0
    return min(agent.delay - (agent.batch_steps - 1) for agent in active_agents)

  def _to_numpy(self, value):
    if isinstance(value, np.ndarray):
      return value
    if hasattr(value, 'numpy'):
      return value.numpy()
    return np.array(value)

  def _sample_to_numpy(self, sample: SampleOutputs) -> SampleOutputs:
    controller_state = utils.map_single_structure(self._to_numpy, sample.controller_state)
    logits = utils.map_single_structure(self._to_numpy, sample.logits)
    return SampleOutputs(controller_state=controller_state, logits=logits)

  def _structure_to_numpy(self, struct: tp.Any) -> tp.Any:
    return utils.map_single_structure(self._to_numpy, struct)

  def _zeros_like_sample(self, sample: SampleOutputs) -> SampleOutputs:
    sample_np = self._sample_to_numpy(sample)
    controller_state = utils.map_single_structure(
        lambda arr: np.zeros((self._num_envs,) + arr.shape[1:], dtype=arr.dtype),
        sample_np.controller_state)
    logits = utils.map_single_structure(
        lambda arr: np.zeros((self._num_envs,) + arr.shape[1:], dtype=arr.dtype),
        sample_np.logits)
    return SampleOutputs(controller_state=controller_state, logits=logits)

  def _zeros_like_state(self, state: tp.Any) -> tp.Any:
    state_np = self._structure_to_numpy(state)
    return utils.map_single_structure(
        lambda arr: np.zeros((self._num_envs,) + arr.shape[1:], dtype=arr.dtype),
        state_np)

  def _copy_sample(self, sample: SampleOutputs) -> SampleOutputs:
    sample_np = self._sample_to_numpy(sample)
    controller_state = utils.map_single_structure(lambda arr: arr.copy(), sample_np.controller_state)
    logits = utils.map_single_structure(lambda arr: arr.copy(), sample_np.logits)
    return SampleOutputs(controller_state=controller_state, logits=logits)

  def _copy_structure(self, struct: tp.Any) -> tp.Any:
    struct_np = self._structure_to_numpy(struct)
    return utils.map_single_structure(lambda arr: arr.copy(), struct_np)

  def _scatter_name(self, port: Port, name_code) -> np.ndarray:
    name_array = np.zeros((self._num_envs,), dtype=embed.NAME_DTYPE)
    indices = self._active_indices.get(port)

    arr = np.asarray(name_code)
    if arr.ndim == 0:
      try:
        scalar_value = int(arr)
      except Exception:
        scalar_value = 0
      name_array.fill(np.asarray(scalar_value, dtype=embed.NAME_DTYPE))
      return name_array

    values = arr.reshape(-1).astype(embed.NAME_DTYPE, copy=False)
    if indices is None or indices.size == 0:
      return name_array
    if values.size != indices.size:
      raise ValueError(
          f'Name code length {values.size} does not match active columns {indices.size}')
    name_array[indices] = values
    return name_array

  def _scatter_sample(self, port: Port, sample: SampleOutputs) -> SampleOutputs:
    active_count = self._active_counts[port]
    if active_count == 0:
      return self._copy_sample(self._output_templates[port])
    if active_count == self._num_envs:
      return self._copy_sample(sample)

    sample_np = self._sample_to_numpy(sample)
    indices = self._active_indices[port]
    base_cs = utils.map_single_structure(lambda arr: arr.copy(), self._output_templates[port].controller_state)
    base_logits = utils.map_single_structure(lambda arr: arr.copy(), self._output_templates[port].logits)

    def assign(full_arr, active_arr):
      full_arr[indices] = active_arr
      return full_arr

    controller_state = utils.map_nt(assign, base_cs, sample_np.controller_state)
    logits = utils.map_nt(assign, base_logits, sample_np.logits)
    return SampleOutputs(controller_state=controller_state, logits=logits)

  def _scatter_state(self, port: Port, state: tp.Any) -> tp.Any:
    active_count = self._active_counts[port]
    if active_count == 0:
      return self._copy_structure(self._zero_states[port])
    if active_count == self._num_envs:
      return self._copy_structure(state)

    state_np = self._structure_to_numpy(state)
    indices = self._active_indices[port]
    base_state = self._copy_structure(self._zero_states[port])

    def assign(full_arr, active_arr):
      full_arr[indices] = active_arr
      return full_arr

    return utils.map_nt(assign, base_state, state_np)

  def _slice_game(self, port: Port, game: Game) -> Game:
    if self._active_counts[port] == self._num_envs:
      return game
    indices = self._active_indices[port]
    return utils.map_single_structure(lambda arr: arr[indices], game)

  def _slice_needs_reset(self, port: Port, needs_reset: np.ndarray) -> np.ndarray:
    if self._active_counts[port] == self._num_envs:
      return needs_reset
    return needs_reset[self._active_indices[port]]

  def rollout(self, num_steps: int) -> tuple[tp.Mapping[Port, Trajectory], Timings]:
    # This ensures that the agent can process all of the states it will be fed.
    for agent in self._agents.values():
      if agent is None:
        continue
      if num_steps % agent.batch_steps != 0:
        raise ValueError('Agent batch steps must divide rollout length.')

    # Buffers for per-frame data.
    gamestates: dict[Port, list[Game]] = {
        port: [] for port in self._agent_ports
    }
    sample_outputs: dict[Port, list[SampleOutputs]] = {
        port: [] for port in self._agent_ports
    }
    is_resetting: list[bool] = []
    active_masks: dict[Port, np.ndarray | None] = {
        port: None for port in self._agent_ports
    }

    # Record each agent's initial state at the beginning of the rollout.
    initial_states = {}
    for port in self._agent_ports:
      agent = self._agents[port]
      if agent is None:
        initial_states[port] = self._copy_structure(self._zero_states[port])
      else:
        initial_states[port] = self._scatter_state(port, agent.hidden_state)

    step_profiler = utils.Profiler()

    def record_state(
        env_output: env_lib.EnvOutput,
        prev_agent_outputs: dict[Port, SampleOutputs],
    ):
      for port, game in env_output.gamestates.items():
        gamestates[port].append(game)
        sample_outputs[port].append(prev_agent_outputs[port])
        mask_array = self._normalize_mask(env_output.active[port])
        if not np.array_equal(mask_array, self._mask_signature[port]):
          raise ValueError('Active mask changed during rollout.')
        if active_masks[port] is None:
          active_masks[port] = mask_array
        elif not np.array_equal(active_masks[port], mask_array):
          raise ValueError('Active mask changed during rollout.')
      is_resetting.append(env_output.needs_reset)

    for _ in range(num_steps):
      # Note that there will always be a first gamestate before any actions
      # are fed into the environment; either the initial state or the last
      # state peeked on the previous rollout.
      with step_profiler:
        output = self._env.pop()

      record_state(output, self._prev_agent_outputs.popleft())

      # Asynchronously push the gamestates to the agents.
      for port in self._agent_ports:
        agent = self._agents[port]
        if agent is None:
          continue
        game = output.gamestates[port]
        sliced_game = self._slice_game(port, game)
        sliced_reset = self._slice_needs_reset(port, output.needs_reset)
        # The agent is responsible for calling from_state on the game.
        agent.push(sliced_game, sliced_reset)

      # Feed the actions from the agents into the environment.
      self._push_actions()

    # Record the last gamestate and action, but don't pop them as we will
    # also use them to begin the next rollout.
    record_state(self._env.peek(), self._prev_agent_outputs[0])

    # Record the delayed actions.
    assert len(self._prev_agent_outputs) == 1 + self.env_runahead
    remaining_actions = list(self._prev_agent_outputs)[1:]
    delayed_actions: dict[Port, list[SampleOutputs]] = {}
    for port in self._agent_ports:
      agent = self._agents[port]
      delayed_actions[port] = [self._copy_sample(actions[port]) for actions in remaining_actions]
      if agent is None:
        continue
      num_left = agent.delay - self.env_runahead
      if num_left > 0:
        delayed_actions[port].extend(
            self._scatter_sample(port, peeked)
            for peeked in agent.peek_n(num_left))

      # Note: the above call to peek_n forces the agent to process all
      # of the `num_steps` states that it's been fed. This ensures that the
      # agent's hidden state is the correct one on the next rollout.
      if agent is not None and isinstance(agent, eval_lib.AsyncDelayedAgent):
        # Assert that the agent has in fact processed all of the states.
        assert agent._state_queue.empty()

    # Now batch everything up into time-major Trajectories.
    trajectories = {}
    is_resetting = np.array(is_resetting)
    for port in self._agent_ports:
      agent = self._agents[port]
      if active_masks[port] is None:
        raise ValueError(f'Missing active mask for port {port}.')
      if agent is None:
        name_code = self._reference_agent.name_code
      else:
        name_code = agent.name_code
      full_names = self._scatter_name(port, name_code)
      repeated_names = np.tile(full_names, (num_steps + 1, 1))
      states=utils.batch_nest_nt(gamestates[port])
      policy = self._agents[port]._policy if agent is not None else self._reference_agent._policy
      trajectories[port] = Trajectory(
          # TODO: Let the learner call from_state on game
          states=policy.embed_game.from_state(states),
          name=repeated_names,
          actions=utils.batch_nest_nt(sample_outputs[port]),
          rewards=reward.compute_rewards(states, self._damage_ratio),
          is_resetting=is_resetting,
          initial_state=initial_states[port],
          # Note that delayed actions aren't time-concatenated, mainly to
          # simplify the case where the delay is 0.
          delayed_actions=delayed_actions[port],
          active_mask=active_masks[port].copy(),
      )

    timings = {
        'env_pop': step_profiler.mean_time(),
        'env_push': self._env_push_profiler.mean_time(),
        'agent_pop': {
            port: (profiler.mean_time() if profiler is not None else 0.0)
            for port, profiler in self._agent_profilers.items()},
        'agent_step': {
            port: (self._agents[port].step_profiler.mean_time()
                   if self._agents[port] is not None else 0.0)
            for port in self._agent_ports
        },
    }

    metrics = dict(
        timing=timings,
        unexpected_reset=is_resetting[1:],
    )

    # self._env_push_profiler.dump_stats('env_push.prof')

    return trajectories, metrics

  def update_variables(
      self, updates: tp.Mapping[Port, Params],
  ):
    for port, values in updates.items():
      agent = self._agents.get(port)
      if agent is None:
        continue
      policy = agent._policy
      for var, val in zip(policy.variables, values):
        var.assign(val)

  @contextlib.contextmanager
  def run(self):
    try:
      self.start()
      yield
    finally:
      self.stop()

  def start(self):
    # TODO: don't allow starting more than once, or running without starting.
    for agent in self._agents.values():
      if agent is not None:
        agent.start()

  def stop(self):
    for agent in self._agents.values():
      if agent is not None:
        agent.stop()
    self._env.stop()

class RolloutMetrics(tp.NamedTuple):
  reward: float

  @classmethod
  def from_trajectory(cls, trajectory: Trajectory) -> 'RolloutMetrics':
    return cls(reward=np.sum(trajectory.rewards))


class Evaluator(RolloutWorker):

  def rollout(
      self,
      num_steps: int,
      policy_vars: tp.Optional[tp.Mapping[Port, Params]] = None,
  ) -> tuple[tp.Mapping[Port, RolloutMetrics], Timings]:
    if policy_vars is not None:
      self.update_variables(policy_vars)
    trajectories, timings = super().rollout(num_steps)
    metrics = {
        port: RolloutMetrics.from_trajectory(trajectory)
        for port, trajectory in trajectories.items()
    }
    return metrics, timings

RayRolloutWorker = ray.remote(RolloutWorker)

class RayEvaluator:
  def __init__(
      self,
      agent_kwargs: tp.Mapping[Port, dict],
      dolphin_kwargs: dict,
      num_envs: int,  # per-worker
      num_workers: int = 1,
      async_envs: bool = False,
      env_kwargs: dict = {},
      use_gpu: bool = False,
      resources: tp.Mapping[str, float] = {},
  ):
    # TODO: Allow multiple gpu workers on the same machine. For this,
    # we'll need to tell tensorflow not to reserve all gpu memory.
    build_worker = RayRolloutWorker.options(
        num_gpus=1 if use_gpu else 0,
        resources=resources)

    self._rollout_workers: list[ray.ObjectRef[RolloutWorker]] = []
    for _ in range(num_workers):
      self._rollout_workers.append(build_worker.remote(
          agent_kwargs=agent_kwargs,
          dolphin_kwargs=dolphin_kwargs,
          num_envs=num_envs,
          async_envs=async_envs,
          env_kwargs=env_kwargs,
          use_gpu=use_gpu,
      ))

  def update_variables(
      self, updates: tp.Mapping[Port, tp.Sequence[np.ndarray]],
  ):
    ray.wait([
        worker.update_variables.remote(updates)
        for worker in self._rollout_workers])

  def rollout(
      self,
      num_steps: int,
      policy_vars: tp.Optional[tp.Mapping[Port, Params]] = None,
  ) -> tuple[tp.Mapping[Port, RolloutMetrics], Timings]:
    if policy_vars is not None:
      for worker in self._rollout_workers:
        worker.update_variables.remote(policy_vars)

    rollout_futures = [
        worker.rollout.remote(num_steps)
        for worker in self._rollout_workers
    ]
    rollout_results = ray.get(rollout_futures)
    trajectories, timings = zip(*rollout_results)

    # Merge the results.
    trajectories = utils.concat_nest_nt(trajectories)
    # TODO: handle non-mean timings.
    timings = utils.map_nt(lambda *args: np.mean(args), *timings)

    metrics = {
        port: RolloutMetrics.from_trajectory(trajectory)
        for port, trajectory in trajectories.items()
    }
    return metrics, timings

  @contextlib.contextmanager
  def run(self):
    try:
      ray.wait([worker.start.remote() for worker in self._rollout_workers])
      yield
    except KeyboardInterrupt:
      # Properly shut down the workers. If we don't do this, the workers
      # can get stuck, not sure why.
      raise
    finally:
      ray.wait([worker.stop.remote() for worker in self._rollout_workers])
