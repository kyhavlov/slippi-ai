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
    if self._use_ray_envs:
      self._env = env_lib.RayBatchedEnvironment(
          self._num_envs, self._dolphin_kwargs, **self._env_kwargs)
    elif self._use_fake_envs:
      self._env = env_lib.FakeBatchedEnvironment(
          self._num_envs, players=list(self._agents) if self._agents else list(self._ports))
    else:
      if not self._async_envs:
        env_class = env_lib.BatchedEnvironment
      else:
        env_class = env_lib.AsyncBatchedEnvironmentMP
      env_kwargs = dict(self._env_kwargs)
      env_kwargs.setdefault('env_ids', self._env_ids)
      self._env = env_class(
          self._num_envs,
          self._dolphin_kwargs,
          agent_names=self._agent_names,
          scheduler=self._scheduler,
          **env_kwargs,
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
      name_code = agent.name_code
      if self._fused_agent is not None:
        B = self._num_envs
        name_code = name_code[i * B:(i + 1) * B]
      states=utils.batch_nest_nt(gamestates[port])
      trajectories[port] = Trajectory(
          # TODO: Let the learner call from_state on game
          states=agent._policy.embed_game.from_state(states),
          name=np.full(
              [num_steps + 1, self._num_envs],
              name_code,
              dtype=embed.NAME_DTYPE),
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
