"""Evaluates a policy."""

import collections
import contextlib
import dataclasses
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
    )
    return utils.map_nt(
        lambda axis, *ts: utils.concat_nest_nt(ts, axis),
        batch_dims, *trajectories)


@dataclasses.dataclass(frozen=True)
class RolloutGroupSpec:
  label: str
  ports: tuple[Port, ...]
  agent_kwargs: tp.Mapping[Port, dict]
  dolphin_kwargs: dict
  num_envs: int
  async_envs: bool = False
  env_kwargs: dict = dataclasses.field(default_factory=dict)
  use_gpu: bool = False
  damage_ratio: float = 0
  use_fake_envs: bool = False
  use_ray_envs: bool = False
  agent_names: list[tuple[str, ...]] = dataclasses.field(default_factory=list)
  scheduler: tp.Any = None


def _build_env_instance(
    *,
    num_envs: int,
    dolphin_kwargs: dict,
    async_envs: bool,
    env_kwargs: dict,
    use_fake_envs: bool,
    use_ray_envs: bool,
    agent_names: list[tuple[str, ...]],
    scheduler: tp.Any,
    env_ids: list[int],
):
  if use_ray_envs:
    return env_lib.RayBatchedEnvironment(
        num_envs, dolphin_kwargs, **env_kwargs)
  if use_fake_envs:
    return env_lib.FakeBatchedEnvironment(
        num_envs,
        players=dolphin_kwargs['players'],
        agent_names=agent_names,
        env_ids=env_ids,
        scheduler=scheduler,
    )
  env_class = (
      env_lib.AsyncBatchedEnvironmentMP if async_envs
      else env_lib.BatchedEnvironment)
  env_kwargs = dict(env_kwargs)
  env_kwargs.setdefault('env_ids', env_ids)
  return env_class(
      num_envs,
      dolphin_kwargs,
      agent_names=agent_names,
      scheduler=scheduler,
      **env_kwargs,
  )


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
      scheduler = None,
      fuse_ports_inference: bool = False,
  ):
    self._ports = tuple(sorted(agent_kwargs))
    self._num_envs = num_envs
    if fuse_ports_inference and len(self._ports) > 1:
      self._init_fused_agents(
          agent_kwargs=agent_kwargs,
          dolphin_kwargs=dolphin_kwargs,
          num_envs=num_envs,
          use_gpu=use_gpu,
      )
    else:
      self._fused_agent = None
    self._agents = {
        port: eval_lib.build_delayed_agent(
            console_delay=dolphin_kwargs['online_delay'],
            batch_size=num_envs,
            run_on_cpu=not use_gpu,
            **kwargs,
        )
        for port, kwargs in agent_kwargs.items()
    } if self._fused_agent is None else {}
    self._dolphin_kwargs = dolphin_kwargs.copy()
    if self._fused_agent is None:
      for port, kwargs in agent_kwargs.items():
        eval_lib.update_character(
            self._dolphin_kwargs['players'][port],
            kwargs['state']['config'])
    else:
      # In fused mode, all ports share the same policy/config.
      any_kwargs = agent_kwargs[self._ports[0]]
      for port in self._ports:
        eval_lib.update_character(
            self._dolphin_kwargs['players'][port],
            any_kwargs['state']['config'])

    self._prev_agent_outputs = collections.deque()
    if self._fused_agent is None:
      self._prev_agent_outputs.append({
          port: agent.dummy_sample_outputs
          for port, agent in self._agents.items()
      })
    else:
      self._prev_agent_outputs.append(
          self._split_outputs(self._fused_agent.dummy_sample_outputs)
      )

    self._use_fake_envs = use_fake_envs
    self._env_kwargs = env_kwargs
    self._async_envs = async_envs
    self._use_ray_envs = use_ray_envs
    self._agent_names = agent_names
    self._scheduler = scheduler
    self._env_ids = list(range(num_envs))
    self._build_env()

    self._damage_ratio = damage_ratio

    if self._fused_agent is None:
      self._agent_profilers = {
          port: utils.Profiler() for port in self._agents}
    else:
      self._agent_profilers = {port: utils.Profiler() for port in self._ports}
    # self._env_push_profiler = cProfile.Profile()
    self._env_push_profiler = utils.Profiler()

    # Make sure that the buffer sizes aren't too big.
    # TODO: do this check before env/agent creation
    agents_to_check = self._agents.values() if self._fused_agent is None else [self._fused_agent]
    for agent in agents_to_check:
      # We get one environment state (the initial one) for free.
      slack = 1 + agent.delay

      # Maximum number of items that could get stuck.
      max_agent_buffer = agent.batch_steps - 1
      max_env_buffer = self._env.num_steps - 1

      if max_agent_buffer + max_env_buffer >= slack:
        self._env.stop()
        raise ValueError(
            f'Agent and environment step buffer sizes are too large: '
            f'{max_agent_buffer} + {max_env_buffer} >= {slack}')

    # Get the environment to run ahead as much as possible.
    # Because we push env states to the agents once per main loop iteration,
    # we need to leave each agent with at least batch_steps - 1 actions in its
    # buffer. This ensures that the agent will have enough env states pushed to
    # take a multi_step just as its output queue runs out.
    self.env_runahead = min(
        agent.delay - (agent.batch_steps - 1)
        for agent in (self._agents.values() if self._fused_agent is None else [self._fused_agent])
    )
    for _ in range(self.env_runahead):
      self._push_actions()

  def _init_fused_agents(
      self,
      agent_kwargs: tp.Mapping[Port, dict],
      dolphin_kwargs: dict,
      num_envs: int,
      use_gpu: bool,
  ):
    """Build a single DelayedAgent for all ports stacked in batch dimension."""
    ports = tuple(sorted(agent_kwargs))
    first = agent_kwargs[ports[0]]

    for port in ports[1:]:
      other = agent_kwargs[port]
      if other.get('state') is not first.get('state'):
        raise ValueError('fuse_ports_inference requires all ports share the same state object.')
      for key in ('compile', 'jit_compile', 'batch_steps', 'async_inference', 'fake'):
        if other.get(key) != first.get(key):
          raise ValueError(f'fuse_ports_inference requires identical agent kwarg {key} across ports.')

    if first.get('async_inference'):
      raise ValueError('fuse_ports_inference is not supported with async_inference yet.')

    # Build the concatenated name list (port-major order).
    names: list[str] = []
    for port in ports:
      port_names = agent_kwargs[port].get('name')
      if port_names is None:
        raise ValueError('Expected per-port name list for fused inference.')
      if len(port_names) != num_envs:
        raise ValueError(f'Expected name list length {num_envs}, got {len(port_names)} for port {port}.')
      names.extend(port_names)

    fused_kwargs = dict(first)
    fused_kwargs['name'] = names

    self._fused_agent = eval_lib.build_delayed_agent(
        console_delay=dolphin_kwargs['online_delay'],
        batch_size=num_envs * len(ports),
        run_on_cpu=not use_gpu,
        **fused_kwargs,
    )

  def _split_outputs(self, outputs: SampleOutputs) -> dict[Port, SampleOutputs]:
    """Split a [P*B] SampleOutputs into a dict of [B] SampleOutputs per port."""
    if self._fused_agent is None:
      raise RuntimeError('_split_outputs called without fused agent.')
    B = self._num_envs
    per_port: dict[Port, SampleOutputs] = {}
    for idx, port in enumerate(self._ports):
      sl = slice(idx * B, (idx + 1) * B)
      per_port[port] = utils.map_single_structure(lambda x: x[sl], outputs)
    return per_port

  def _pack_states(self, states: dict[Port, Game]) -> Game:
    """Pack per-port [B] states into one [P*B] state."""
    B = self._num_envs
    del B
    return utils.map_nt(
        lambda *xs: np.concatenate(xs, axis=0),
        *[states[port] for port in self._ports],
    )

  def _pack_needs_reset(self, needs_reset: np.ndarray) -> np.ndarray:
    return np.concatenate([needs_reset] * len(self._ports), axis=0)

  def _split_controllers(self, controllers_all) -> dict[Port, tp.Any]:
    """Split a [P*B] controller structure into per-port [B] controllers."""
    B = self._num_envs
    per_port = {}
    for idx, port in enumerate(self._ports):
      sl = slice(idx * B, (idx + 1) * B)
      per_port[port] = utils.map_single_structure(lambda x: x[sl], controllers_all)
    return per_port

  def _build_env(self):
    self._env = _build_env_instance(
        num_envs=self._num_envs,
        dolphin_kwargs=self._dolphin_kwargs,
        async_envs=self._async_envs,
        env_kwargs=self._env_kwargs,
        use_fake_envs=self._use_fake_envs,
        use_ray_envs=self._use_ray_envs,
        agent_names=self._agent_names,
        scheduler=self._scheduler,
        env_ids=self._env_ids,
    )

  def reset_env(self):
    self._env.stop()
    self._build_env()

    # Start env runahead
    # TODO: properly reset the agents with dummy actions instead of reusing
    # delayed actions from the previous rollout.
    assert len(self._prev_agent_outputs) == 1 + self.env_runahead
    for agent_outputs in list(self._prev_agent_outputs)[1:]:
      if self._fused_agent is None:
        decoded_actions = {
            port: self._agents[port].embed_controller.decode(output.controller_state)
            for port, output in agent_outputs.items()
        }
      else:
        decoded_actions = {
            port: self._fused_agent.embed_controller.decode(output.controller_state)
            for port, output in agent_outputs.items()
        }
      with self._env_push_profiler:
        self._env.push(decoded_actions)


  def _push_actions(self):
    """Pop actions from the agents and push them to the environment."""
    if self._fused_agent is None:
      outputs: dict[Port, SampleOutputs] = {}
      for port, agent in self._agents.items():
        with self._agent_profilers[port]:
          outputs[port] = agent.pop()
      self._prev_agent_outputs.append(outputs)

      decoded_actions = {
          port: self._agents[port].embed_controller.decode(action.controller_state)
          for port, action in outputs.items()
      }
    else:
      with self._agent_profilers[self._ports[0]]:
        combined = self._fused_agent.pop()
      outputs = self._split_outputs(combined)
      self._prev_agent_outputs.append(outputs)

      decoded_all = self._fused_agent.embed_controller.decode(combined.controller_state)
      decoded_actions = self._split_controllers(decoded_all)

    with self._env_push_profiler:
      self._env.push(decoded_actions)

  def rollout(self, num_steps: int) -> tuple[tp.Mapping[Port, Trajectory], Timings]:
    # This ensures that the agent can process all of the states it will be fed.
    agents_to_check = self._agents.values() if self._fused_agent is None else [self._fused_agent]
    for agent in agents_to_check:
      if num_steps % agent.batch_steps != 0:
        raise ValueError('Agent batch steps must divide rollout length.')

    # Buffers for per-frame data.
    gamestates: dict[Port, list[Game]] = {
        port: [] for port in self._ports
    }
    sample_outputs: dict[Port, list[SampleOutputs]] = {
        port: [] for port in self._ports
    }
    is_resetting: list[bool] = []

    # Record each agent's initial state at the beginning of the rollout.
    if self._fused_agent is None:
      initial_states = {
          port: agent.hidden_state
          for port, agent in self._agents.items()
      }
    else:
      B = self._num_envs
      fused_initial = self._fused_agent.hidden_state
      initial_states = {
          port: utils.map_single_structure(
              lambda x, sl=slice(i * B, (i + 1) * B): x[sl],
              fused_initial,
          )
          for i, port in enumerate(self._ports)
      }

    step_profiler = utils.Profiler()

    def record_state(
        env_output: env_lib.EnvOutput,
        prev_agent_outputs: dict[Port, SampleOutputs],
    ):
      for port in self._ports:
        game = env_output.gamestates[port]
        gamestates[port].append(game)
        sample_outputs[port].append(prev_agent_outputs[port])
      is_resetting.append(env_output.needs_reset)

    for _ in range(num_steps):
      # Note that there will always be a first gamestate before any actions
      # are fed into the environment; either the initial state or the last
      # state peeked on the previous rollout.
      with step_profiler:
        output = self._env.pop()

      record_state(output, self._prev_agent_outputs.popleft())

      # Asynchronously push the gamestates to the agents.
      if self._fused_agent is None:
        for port, agent in self._agents.items():
          game = output.gamestates[port]
          # The agent is responsible for calling from_state on the game.
          agent.push(game, output.needs_reset)
      else:
        packed_game = self._pack_states(output.gamestates)
        packed_reset = self._pack_needs_reset(output.needs_reset)
        self._fused_agent.push(packed_game, packed_reset)

      # Feed the actions from the agents into the environment.
      self._push_actions()

    # Record the last gamestate and action, but don't pop them as we will
    # also use them to begin the next rollout.
    record_state(self._env.peek(), self._prev_agent_outputs[0])

    # Record the delayed actions.
    assert len(self._prev_agent_outputs) == 1 + self.env_runahead
    remaining_actions = list(self._prev_agent_outputs)[1:]
    delayed_actions: dict[Port, list[SampleOutputs]] = {}
    if self._fused_agent is None:
      for port, agent in self._agents.items():
        delayed_actions[port] = [actions[port] for actions in remaining_actions]
        num_left = agent.delay - self.env_runahead
        delayed_actions[port].extend(agent.peek_n(num_left))

        # Note: the above call to peek_n forces the agent to process all
        # of the `num_steps` states that it's been fed. This ensures that the
        # agent's hidden state is the correct one on the next rollout.
        if isinstance(agent, eval_lib.AsyncDelayedAgent):
          # Assert that the agent has in fact processed all of the states.
          assert agent._state_queue.empty()
    else:
      for port in self._ports:
        delayed_actions[port] = [actions[port] for actions in remaining_actions]
      num_left = self._fused_agent.delay - self.env_runahead
      peeked = self._fused_agent.peek_n(num_left)
      for combined in peeked:
        split = self._split_outputs(combined)
        for port in self._ports:
          delayed_actions[port].append(split[port])

    # Now batch everything up into time-major Trajectories.
    trajectories = {}
    is_resetting = np.array(is_resetting)
    for i, port in enumerate(self._ports):
      agent = self._agents[port] if self._fused_agent is None else self._fused_agent
      name_code = np.asarray(agent.name_code, dtype=embed.NAME_DTYPE)
      if self._fused_agent is not None:
        B = self._num_envs
        if name_code.ndim == 0:
          name_code = np.full([B * len(self._ports)], name_code, dtype=embed.NAME_DTYPE)
        name_code = name_code[i * B:(i + 1) * B]
      elif name_code.ndim == 0:
        name_code = np.full([self._num_envs], name_code, dtype=embed.NAME_DTYPE)
      states=utils.batch_nest_nt(gamestates[port])
      trajectories[port] = Trajectory(
          # TODO: Let the learner call from_state on game
          states=agent._policy.embed_game.from_state(states),
          name=np.broadcast_to(
              np.asarray(name_code, dtype=embed.NAME_DTYPE),
              [num_steps + 1, self._num_envs],
          ).copy(),
          actions=utils.batch_nest_nt(sample_outputs[port]),
          rewards=reward.compute_rewards(states, self._damage_ratio),
          is_resetting=is_resetting,
          initial_state=initial_states[port],
          # Note that delayed actions aren't time-concatenated, mainly to
          # simplify the case where the delay is 0.
          delayed_actions=delayed_actions[port],
      )

    timings = {
        'env_pop': step_profiler.mean_time(),
        'env_push': self._env_push_profiler.mean_time(),
        'agent_pop': {
            port: (profiler.mean_time() if profiler.num_calls else 0.0)
            for port, profiler in self._agent_profilers.items()},
        'agent_step': {
            port: (self._agents[port].step_profiler.mean_time() if self._fused_agent is None else self._fused_agent.step_profiler.mean_time())
            for port in self._ports
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
    if self._fused_agent is not None:
      # In fused mode, all ports share the same policy instance.
      if not updates:
        return
      any_values = next(iter(updates.values()))
      policy = self._fused_agent._policy
      for var, val in zip(policy.variables, any_values):
        var.assign(val)
      return

    for port, values in updates.items():
      policy = self._agents[port]._policy
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
    if self._fused_agent is not None:
      self._fused_agent.start()
    else:
      for agent in self._agents.values():
        agent.start()

  def stop(self):
    if self._fused_agent is not None:
      self._fused_agent.stop()
    else:
      for agent in self._agents.values():
        agent.stop()
    self._env.stop()


@dataclasses.dataclass
class _MixedGroupRuntime:
  spec: RolloutGroupSpec
  env: tp.Any
  env_ids: list[int]
  batch_slices: dict[Port, slice]
  env_pop_profiler: utils.Profiler = dataclasses.field(
      default_factory=lambda: utils.Profiler())
  env_push_profiler: utils.Profiler = dataclasses.field(
      default_factory=lambda: utils.Profiler())


class MixedFusedRolloutWorker:

  def __init__(self, group_specs: tp.Sequence[RolloutGroupSpec]):
    if not group_specs:
      raise ValueError('MixedFusedRolloutWorker requires at least one group.')

    self._group_specs = list(group_specs)
    self._ports = tuple(dict.fromkeys(
        port
        for spec in self._group_specs
        for port in spec.ports
    ))
    self._groups: list[_MixedGroupRuntime] = []
    self._fused_agent = None
    self._prev_agent_outputs = collections.deque()
    self._agent_pop_profiler = utils.Profiler()
    self._total_batch_size = 0
    self._build()

  @property
  def ports(self) -> tuple[int, ...]:
    return self._ports

  def _build(self):
    first_spec = self._group_specs[0]
    first_port = first_spec.ports[0]
    first_kwargs = dict(first_spec.agent_kwargs[first_port])
    first_state = first_kwargs.get('state')
    if first_state is None:
      raise ValueError('Mixed fused worker requires preloaded state in agent kwargs.')
    first_online_delay = first_spec.dolphin_kwargs['online_delay']
    damage_ratio = first_spec.damage_ratio

    fused_kwargs = {
        key: value
        for key, value in first_kwargs.items()
        if key != 'name'
    }
    fused_names: list[str] = []
    offset = 0

    for spec in self._group_specs:
      if spec.use_ray_envs:
        raise ValueError('Mixed fused worker does not support ray_envs=True.')
      if spec.dolphin_kwargs['online_delay'] != first_online_delay:
        raise ValueError('All mixed fused groups must share the same online_delay.')
      if spec.use_gpu != first_spec.use_gpu:
        raise ValueError('All mixed fused groups must share the same use_gpu setting.')
      if spec.damage_ratio != damage_ratio:
        raise ValueError('All mixed fused groups must share the same damage_ratio.')

      batch_slices: dict[Port, slice] = {}
      for port in spec.ports:
        kwargs = dict(spec.agent_kwargs[port])
        if kwargs.get('state') is not first_state:
          raise ValueError('All mixed fused groups must share the same state object.')
        port_names = kwargs.get('name')
        if port_names is None or len(port_names) != spec.num_envs:
          raise ValueError(
              f'Expected name list length {spec.num_envs} for port {port} in group {spec.label}.')
        other_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key != 'name'
        }
        for key in other_kwargs:
          if key == 'state':
            continue
          if other_kwargs[key] != fused_kwargs.get(key):
            raise ValueError(
                f'Mixed fused groups require identical agent kwarg {key}; '
                f'group {spec.label} port {port} differed.')
        fused_names.extend(port_names)
        batch_slices[port] = slice(offset, offset + spec.num_envs)
        offset += spec.num_envs

      dolphin_kwargs = spec.dolphin_kwargs.copy()
      for port in spec.ports:
        eval_lib.update_character(
            dolphin_kwargs['players'][port],
            first_state['config'],
        )
      env_ids = list(range(spec.num_envs))
      env = _build_env_instance(
          num_envs=spec.num_envs,
          dolphin_kwargs=dolphin_kwargs,
          async_envs=spec.async_envs,
          env_kwargs=spec.env_kwargs,
          use_fake_envs=spec.use_fake_envs,
          use_ray_envs=spec.use_ray_envs,
          agent_names=spec.agent_names,
          scheduler=spec.scheduler,
          env_ids=env_ids,
      )
      self._groups.append(_MixedGroupRuntime(
          spec=spec,
          env=env,
          env_ids=env_ids,
          batch_slices=batch_slices,
      ))

    self._total_batch_size = offset
    fused_kwargs['name'] = fused_names
    self._fused_agent = eval_lib.build_delayed_agent(
        console_delay=first_online_delay,
        batch_size=self._total_batch_size,
        run_on_cpu=not first_spec.use_gpu,
        **fused_kwargs,
    )
    self._damage_ratio = damage_ratio

    self._prev_agent_outputs.append(
        self._split_outputs(self._fused_agent.dummy_sample_outputs)
    )

    slack = 1 + self._fused_agent.delay
    max_agent_buffer = self._fused_agent.batch_steps - 1
    max_env_buffer = max(group.env.num_steps - 1 for group in self._groups)
    if max_agent_buffer + max_env_buffer >= slack:
      self.stop()
      raise ValueError(
          f'Agent and environment step buffer sizes are too large: '
          f'{max_agent_buffer} + {max_env_buffer} >= {slack}')

    self.env_runahead = self._fused_agent.delay - (self._fused_agent.batch_steps - 1)
    for _ in range(self.env_runahead):
      self._push_actions()

  def _rebuild_envs(self):
    first_state = self._group_specs[0].agent_kwargs[self._group_specs[0].ports[0]]['state']
    new_groups: list[_MixedGroupRuntime] = []
    for group in self._groups:
      group.env.stop()
      dolphin_kwargs = group.spec.dolphin_kwargs.copy()
      for port in group.spec.ports:
        eval_lib.update_character(
            dolphin_kwargs['players'][port],
            first_state['config'],
        )
      env = _build_env_instance(
          num_envs=group.spec.num_envs,
          dolphin_kwargs=dolphin_kwargs,
          async_envs=group.spec.async_envs,
          env_kwargs=group.spec.env_kwargs,
          use_fake_envs=group.spec.use_fake_envs,
          use_ray_envs=group.spec.use_ray_envs,
          agent_names=group.spec.agent_names,
          scheduler=group.spec.scheduler,
          env_ids=group.env_ids,
      )
      new_groups.append(_MixedGroupRuntime(
          spec=group.spec,
          env=env,
          env_ids=group.env_ids,
          batch_slices=group.batch_slices,
      ))
    self._groups = new_groups

  def _split_outputs(
      self,
      outputs: SampleOutputs,
  ) -> list[dict[Port, SampleOutputs]]:
    split_groups: list[dict[Port, SampleOutputs]] = []
    for group in self._groups:
      per_port: dict[Port, SampleOutputs] = {}
      for port in group.spec.ports:
        sl = group.batch_slices[port]
        per_port[port] = utils.map_single_structure(
            lambda x, sl=sl: x[sl],
            outputs,
        )
      split_groups.append(per_port)
    return split_groups

  def _split_controllers(
      self,
      controllers_all,
  ) -> list[dict[Port, tp.Any]]:
    split_groups: list[dict[Port, tp.Any]] = []
    for group in self._groups:
      per_port = {}
      for port in group.spec.ports:
        sl = group.batch_slices[port]
        per_port[port] = utils.map_single_structure(
            lambda x, sl=sl: x[sl],
            controllers_all,
        )
      split_groups.append(per_port)
    return split_groups

  def _pack_states(self, outputs: list[env_lib.EnvOutput]) -> Game:
    packed = []
    for group, output in zip(self._groups, outputs):
      for port in group.spec.ports:
        packed.append(output.gamestates[port])
    return utils.map_nt(lambda *xs: np.concatenate(xs, axis=0), *packed)

  def _pack_needs_reset(self, outputs: list[env_lib.EnvOutput]) -> np.ndarray:
    pieces = []
    for group, output in zip(self._groups, outputs):
      pieces.extend([output.needs_reset] * len(group.spec.ports))
    return np.concatenate(pieces, axis=0)

  def reset_env(self):
    self._rebuild_envs()
    assert len(self._prev_agent_outputs) == 1 + self.env_runahead
    for outputs_by_group in list(self._prev_agent_outputs)[1:]:
      decoded_actions_by_group = []
      for per_port in outputs_by_group:
        decoded_actions_by_group.append({
            port: self._fused_agent.embed_controller.decode(output.controller_state)
            for port, output in per_port.items()
        })
      for group, decoded_actions in zip(self._groups, decoded_actions_by_group):
        with group.env_push_profiler:
          group.env.push(decoded_actions)

  def _push_actions(self):
    with self._agent_pop_profiler:
      combined = self._fused_agent.pop()
    outputs_by_group = self._split_outputs(combined)
    self._prev_agent_outputs.append(outputs_by_group)

    decoded_all = self._fused_agent.embed_controller.decode(
        combined.controller_state)
    decoded_actions_by_group = self._split_controllers(decoded_all)
    for group, decoded_actions in zip(self._groups, decoded_actions_by_group):
      with group.env_push_profiler:
        group.env.push(decoded_actions)

  def rollout(self, num_steps: int) -> tuple[Trajectory, Timings]:
    if num_steps % self._fused_agent.batch_steps != 0:
      raise ValueError('Agent batch steps must divide rollout length.')

    gamestates = [
        {port: [] for port in group.spec.ports}
        for group in self._groups
    ]
    sample_outputs = [
        {port: [] for port in group.spec.ports}
        for group in self._groups
    ]
    is_resetting = [[] for _ in self._groups]

    fused_initial = self._fused_agent.hidden_state
    initial_states: list[dict[Port, policies.RecurrentState]] = []
    for group in self._groups:
      per_port = {}
      for port in group.spec.ports:
        sl = group.batch_slices[port]
        per_port[port] = utils.map_single_structure(
            lambda x, sl=sl: x[sl],
            fused_initial,
        )
      initial_states.append(per_port)

    def record_state(
        outputs: list[env_lib.EnvOutput],
        prev_outputs: list[dict[Port, SampleOutputs]],
    ):
      for group_index, (group, output, prev) in enumerate(
          zip(self._groups, outputs, prev_outputs)):
        for port in group.spec.ports:
          gamestates[group_index][port].append(output.gamestates[port])
          sample_outputs[group_index][port].append(prev[port])
        is_resetting[group_index].append(output.needs_reset)

    for _ in range(num_steps):
      outputs = []
      for group in self._groups:
        with group.env_pop_profiler:
          outputs.append(group.env.pop())

      record_state(outputs, self._prev_agent_outputs.popleft())
      packed_game = self._pack_states(outputs)
      packed_reset = self._pack_needs_reset(outputs)
      self._fused_agent.push(packed_game, packed_reset)
      self._push_actions()

    final_outputs = [group.env.peek() for group in self._groups]
    record_state(final_outputs, self._prev_agent_outputs[0])

    assert len(self._prev_agent_outputs) == 1 + self.env_runahead
    remaining_actions = list(self._prev_agent_outputs)[1:]
    delayed_actions = [
        {port: [] for port in group.spec.ports}
        for group in self._groups
    ]
    for outputs_by_group in remaining_actions:
      for group_index, group_outputs in enumerate(outputs_by_group):
        for port in self._groups[group_index].spec.ports:
          delayed_actions[group_index][port].append(group_outputs[port])

    num_left = self._fused_agent.delay - self.env_runahead
    peeked = self._fused_agent.peek_n(num_left)
    for combined in peeked:
      outputs_by_group = self._split_outputs(combined)
      for group_index, group_outputs in enumerate(outputs_by_group):
        for port in self._groups[group_index].spec.ports:
          delayed_actions[group_index][port].append(group_outputs[port])

    group_trajectories: list[Trajectory] = []
    unexpected_reset_by_mode: dict[str, np.ndarray] = {}
    timing_by_mode: dict[str, tp.Any] = {}
    name_code = np.asarray(self._fused_agent.name_code, dtype=embed.NAME_DTYPE)
    if name_code.ndim == 0:
      name_code = np.full(
          [self._total_batch_size],
          name_code,
          dtype=embed.NAME_DTYPE,
      )
    for group_index, group in enumerate(self._groups):
      per_port_trajectories = []
      group_is_resetting = np.array(is_resetting[group_index])
      unexpected_reset_by_mode[group.spec.label] = group_is_resetting[1:]
      for port in group.spec.ports:
        sl = group.batch_slices[port]
        states = utils.batch_nest_nt(gamestates[group_index][port])
        per_port_trajectories.append(Trajectory(
            states=self._fused_agent._policy.embed_game.from_state(states),
            name=np.broadcast_to(
                np.asarray(name_code[sl], dtype=embed.NAME_DTYPE),
                [num_steps + 1, group.spec.num_envs],
            ).copy(),
            actions=utils.batch_nest_nt(sample_outputs[group_index][port]),
            rewards=reward.compute_rewards(states, self._damage_ratio),
            is_resetting=group_is_resetting,
            initial_state=initial_states[group_index][port],
            delayed_actions=delayed_actions[group_index][port],
        ))
      group_trajectories.append(Trajectory.batch(per_port_trajectories))

      timing_by_mode[group.spec.label] = {
          'env_pop': group.env_pop_profiler.mean_time(),
          'env_push': group.env_push_profiler.mean_time(),
          'agent_pop': {
              port: self._agent_pop_profiler.mean_time()
              for port in group.spec.ports
          },
          'agent_step': {
              port: self._fused_agent.step_profiler.mean_time()
              for port in group.spec.ports
          },
      }

    return (
        Trajectory.batch(group_trajectories),
        dict(
            timing=timing_by_mode,
            unexpected_reset=unexpected_reset_by_mode,
        ),
    )

  def update_variables(self, updates: tp.Mapping[Port, Params]):
    if not updates:
      return
    any_values = next(iter(updates.values()))
    policy = self._fused_agent._policy
    for var, val in zip(policy.variables, any_values):
      var.assign(val)

  @contextlib.contextmanager
  def run(self):
    try:
      self.start()
      yield
    finally:
      self.stop()

  def start(self):
    self._fused_agent.start()

  def stop(self):
    if self._fused_agent is not None:
      self._fused_agent.stop()
    for group in self._groups:
      group.env.stop()

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
