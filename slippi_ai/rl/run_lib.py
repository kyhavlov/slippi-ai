import dataclasses
import enum
import itertools
import logging
import os
import pickle
import typing as tp

import numpy as np
import tensorflow as tf
import wandb
from collections import defaultdict

from melee import Character

from slippi_ai import (
    dolphin as dolphin_lib,
    eval_lib,
    evaluators,
    flag_utils,
    nametags,
    policies,
    reward,
    saving,
    tf_utils,
    train_lib,
    utils,
)

from slippi_ai.types import Game
from slippi_ai import value_function as vf_lib
from slippi_ai.rl import learner as learner_lib

field = lambda f: dataclasses.field(default_factory=f)

@dataclasses.dataclass
class RuntimeConfig:
  expt_root: str = 'experiments/rl'
  expt_dir: tp.Optional[str] = None
  tag: tp.Optional[str] = None

  max_step: int = 10  # maximum training step
  max_runtime: tp.Optional[int] = None  # maximum runtime in seconds
  log_interval: int = 10  # seconds between logging
  save_interval: int = 300  # seconds between saving to disk

  # Periodically reset the environments to deal with memory leaks in dolphin.
  reset_every_n_steps: tp.Optional[int] = None
  # Without burnin, we see a spike in teacher_kl after every reset. My guess is
  # that this is because the trajectories and therefore gradients become highly
  # correlated, which is bad. Empirically 10 is a good value to set this to.
  burnin_steps_after_reset: int = 0

@dataclasses.dataclass
class ActorConfig:
  rollout_length: int = 64
  num_envs: int = 1
  async_envs: bool = False
  ray_envs: bool = False
  num_env_steps: int = 0
  inner_batch_size: int = 1
  gpu_inference: bool = True
  use_fake_envs: bool = False
  enable_singles: bool = False

@dataclasses.dataclass
class AgentConfig:
  # TODO: merge with ActorConfig?
  path: tp.Optional[str] = None
  tag: tp.Optional[str] = None
  compile: bool = True
  jit_compile: bool = False
  name: list[str] = field(lambda: [nametags.DEFAULT_NAME])
  batch_steps: int = 0
  async_inference: bool = False

  def get_kwargs(self) -> dict:
    if self.jit_compile:
      logging.warning('jit_compile may lead to instability')
    kwargs = dict(
        compile=self.compile,
        jit_compile=self.jit_compile,
        batch_steps=self.batch_steps,
        async_inference=self.async_inference,
    )
    if self.path or self.tag:
      kwargs['state'] = eval_lib.load_state(path=self.path, tag=self.tag)
    return kwargs

class OpponentType(enum.Enum):
  CPU = 'cpu'
  SELF = 'self'
  OTHER = 'other'

@dataclasses.dataclass
class OpponentConfig:
  type: OpponentType = OpponentType.CPU
  other: AgentConfig = field(AgentConfig)

  # Update self-play parameters every N steps.
  update_interval: tp.Optional[int] = None
  # Train on opponent's data. Implies update_interval=1.
  train: bool = False

  def should_update(self, step: int):
    if self.type is not OpponentType.SELF:
      return False
    if self.train:
      return True
    if self.update_interval is None:
      return False
    return step % self.update_interval == 0

  def should_train(self):
    return self.type is OpponentType.SELF and self.train

@dataclasses.dataclass
class Config:
  runtime: RuntimeConfig = field(RuntimeConfig)

  # num_actors: int = 1
  dolphin: dolphin_lib.DolphinConfig = field(dolphin_lib.DolphinConfig)
  learner: learner_lib.LearnerConfig = field(learner_lib.LearnerConfig)
  actor: ActorConfig = field(ActorConfig)
  agent: AgentConfig = field(AgentConfig)
  opponent: OpponentConfig = field(OpponentConfig)

  # One of these should be set
  teacher: tp.Optional[str] = None
  restore: tp.Optional[str] = None

  override_delay: tp.Optional[int] = None  # for testing

  # Take learner steps without changing the parameters to burn-in the
  # optimizer state for RL.
  optimizer_burnin_steps: int = 0
  # Take some steps to update just the value function before doing RL.
  # Useful if we're training against the level 9 cpu.
  value_burnin_steps: int = 0

DEFAULT_CONFIG = Config()
DEFAULT_CONFIG.dolphin.console_timeout = 30

CHARACTER_WEIGHTINGS = {
      Character.FOX: 2000,
      Character.FALCO: 1000,
      Character.MARTH: 1000,
      Character.SHEIK: 1000,
      Character.PEACH: 1000,
      Character.CPTFALCON: 1000,
      Character.JIGGLYPUFF: 1000,
      Character.PIKACHU: 200,
      Character.YOSHI: 200,
      Character.GANONDORF: 200,
      Character.POPO: 100,
      Character.SAMUS: 100,
      Character.DK: 100,
      Character.LUIGI: 100,
      Character.DOC: 50,
      Character.MARIO: 50,
      Character.YLINK: 50,
      Character.LINK: 50,
      Character.GAMEANDWATCH: 50,
      Character.NESS: 50,
      Character.ROY: 50,
      Character.MEWTWO: 50,
      Character.PICHU: 50,
      Character.BOWSER: 50,
      Character.KIRBY: 50,
}

class LearnerManager:

  def __init__(
      self,
      learner: learner_lib.Learner,
      config: Config,
      build_actor: tp.Callable[[], evaluators.RolloutWorker],
      port: int = 1,
      enemy_port: int = 2,
  ):
    self._config = config
    self._learner = learner
    self._build_actor = build_actor
    self._unroll_length = config.actor.rollout_length
    self._port = port
    self._enemy_port = enemy_port
    self._num_ppo_batches = config.learner.ppo.num_batches
    self._burnin_steps_after_reset = config.runtime.burnin_steps_after_reset

    batch_size = config.actor.num_envs
    if config.opponent.should_train():
      batch_size *= 4
    self._hidden_state = learner.initial_state(batch_size)

    self.update_profiler = utils.Profiler(burnin=0)
    self.learner_profiler = utils.Profiler()
    self.rollout_profiler = utils.Profiler()
    self.reset_profiler = utils.Profiler(burnin=0)

    with self.reset_profiler:
      self.actor = self._build_actor()
      self.actor.start()

      for _ in range(self._burnin_steps_after_reset):
        self.unroll()

  def reset_env(self):
    with self.reset_profiler:
      self.actor.reset_env()

      for _ in range(self._burnin_steps_after_reset):
        self.unroll()

  def _rollout(self) -> tuple[evaluators.Trajectory, dict]:
    trajectories, timings = self.actor.rollout(self._unroll_length)

    ports = [1, 2, 3, 4]
    trajectories = [trajectories[p] for p in ports]
    trajectory = evaluators.Trajectory.batch(trajectories)

    return trajectory, timings

  def unroll(self):
    trajectory, _ = self._rollout()
    _, self._hidden_state = self._learner.compiled_unroll(
        trajectory, self._hidden_state)

  def step(self, step: int, ppo_steps: int = None) -> tuple[list[evaluators.Trajectory], dict]:
    with self.update_profiler:
      variables = {}
      for port in [1, 2, 3, 4]:
        variables[port] = self._learner.policy_variables()
      #variables[self._enemy_port] = self._learner.policy_variables()
      self.actor.update_variables(variables)

    with self.rollout_profiler:
      trajectories = []
      actor_metrics = []
      for _ in range(self._num_ppo_batches):
        trajectory, timings = self._rollout()
        trajectories.append(trajectory)
        actor_metrics.append(timings)

      actor_metrics = tf.nest.map_structure(
          lambda *xs: np.mean(xs), *actor_metrics)
      
      print("collected %d trajectories" % len(trajectories))

    with self.learner_profiler:
      self._hidden_state, metrics = self._learner.ppo(
          trajectories, self._hidden_state, num_epochs=ppo_steps)

    return trajectories, dict(learner=metrics, actor=actor_metrics)

class Logger:

  def __init__(self):
    self.buffer = []

  def record(self, to_log):
    to_log = utils.map_single_structure(train_lib.mean, to_log)
    self.buffer.append(to_log)

  def flush(self, step: int) -> tp.Optional[dict]:
    if not self.buffer:
      return None

    to_log = tf.nest.map_structure(lambda *xs: np.mean(xs), *self.buffer)
    train_lib.log_stats(to_log, step, take_mean=False)
    self.buffer = []
    return to_log

def concise_name(name: str) -> str:
  if name == 'Master Player':
    return 'MP'
  return name

def dummy_trajectory(
    policy: policies.Policy,
    unroll_length: int,
    batch_size: int,
) -> evaluators.Trajectory:
  embedders = dict(policy.embed_state_action.embedding)
  embed_controller = policy.controller_embedding
  shape = [unroll_length + 1, batch_size]
  return evaluators.Trajectory(
      states=embedders['state'].dummy(shape),
      name=embedders['name'].dummy(shape),
      actions=eval_lib.dummy_sample_outputs(embed_controller, shape),
      rewards=np.full([unroll_length, batch_size], 0, dtype=np.float32),
      is_resetting=np.full(shape, False),
      initial_state=policy.initial_state(batch_size),
      delayed_actions=[
          eval_lib.dummy_sample_outputs(embed_controller, [batch_size])
      ] * policy.delay,
  )


def run(config: Config):
  tag = config.runtime.tag or train_lib.get_experiment_tag()
  # Might want to use wandb.run.dir instead, but it doesn't seem
  # to be set properly even when we try to override it.
  expt_dir = config.runtime.expt_dir
  if expt_dir is None:
    expt_dir = os.path.join(config.runtime.expt_root, tag)
    os.makedirs(expt_dir, exist_ok=True)
  logging.info('experiment directory: %s', expt_dir)

  # Restore from existing save file if it exists.
  restore_path = None
  restore_from_checkpoint = False
  pickle_path = os.path.join(expt_dir, 'latest.pkl')
  if os.path.exists(pickle_path):
    logging.info('Restoring from checkpoint %s', pickle_path)
    restore_path = pickle_path
    restore_from_checkpoint = True
  elif config.restore:
    restore_path = config.restore

  if config.teacher and config.restore:
    raise ValueError('Must pass exactly one of "teacher" and "restore".')

  if restore_path:
    rl_state = saving.load_state_from_disk(restore_path)

    previous_config = flag_utils.dataclass_from_dict(
        Config, rl_state['rl_config'])

    if (restore_from_checkpoint and previous_config.restore
        and previous_config.restore != config.restore):
      raise ValueError(
          'Requested restore path does not match checkpoint: '
          f'{config.restore} (requested) != {previous_config.restore} (checkpoint)')

    previous_teacher = previous_config.teacher

    if config.teacher and config.teacher != previous_teacher:
      assert restore_from_checkpoint
      raise ValueError(
          'Requested teacher does not match checkpoint: '
          f'{config.teacher} (requested) != {previous_teacher} (checkpoint)')

    logging.info(f'Using teacher: {previous_teacher}')
    config.teacher = previous_teacher  # For saving
    teacher_state = saving.load_state_from_disk(previous_teacher)

    step = rl_state['step']
  elif config.teacher:
    logging.info(f'Initializing from teacher: {config.teacher}')
    teacher_state = saving.load_state_from_disk(config.teacher)
    rl_state = teacher_state
    step = 0
  else:
    raise ValueError('Must pass exactly one of "teacher" and "restore".')

  if config.override_delay is not None:
    teacher_state['config']['policy']['delay'] = config.override_delay

  # Make sure we don't train the teacher
  with tf_utils.non_trainable_scope():
    teacher = saving.load_policy_from_state(teacher_state)

  policy = saving.load_policy_from_state(rl_state)

  pretraining_config = flag_utils.dataclass_from_dict(
      train_lib.Config, teacher_state['config'])

  # TODO: put this code into saving.py or train_lib.py
  vf_config = pretraining_config.value_function
  value_function = None
  if vf_config.train_separate_network:
    value_net_config = pretraining_config.network
    if vf_config.separate_network_config:
      value_net_config = vf_config.network
    value_function = vf_lib.ValueFunction(
        network_config=value_net_config,
        embed_state_action=policy.embed_state_action,
    )

  learner = learner_lib.Learner(
      config=config.learner,
      teacher=teacher,
      policy=policy,
      value_function=value_function,
  )
  learning_rate = learner.learning_rate

  # Initialize and restore variables
  learner.initialize(dummy_trajectory(policy, 1, 1))
  learner.restore_from_imitation(rl_state['state'])

  PORT = 1

  # set ports 1-4 to AI
  dolphin_kwargs = dict(
      players={port: dolphin_lib.AI(character_weight_table=CHARACTER_WEIGHTINGS) for port in range(1, 5)},
      **config.dolphin.to_kwargs(),
  )

  main_agent_kwargs = config.agent.get_kwargs()
  main_agent_kwargs['state'] = rl_state
  #main_agent_kwargs['fake'] = True
  batch_size = config.actor.num_envs

  if config.opponent.type is not OpponentType.SELF:
    raise NotImplementedError('Only self-play is currently supported.')

  # Generate all possible permutations of 4 players
  all_permutations = list(itertools.product(config.agent.name, repeat=4))
  num_permutations = len(all_permutations)

  # Create the batch by cycling through the permutations in a round-robin manner
  name_configuration_batch = [
      all_permutations[(i * num_permutations // batch_size + i) % num_permutations]
      for i in range(batch_size)
  ]

  agent_kwargs: tp.Mapping[int, dict] = {}
  for i in range(1, 5):
    agent_kwargs[i] = dict(
        name=[name_configuration_batch[j][i-1] for j in range(batch_size)],
        **main_agent_kwargs.copy(),
    )
    print("port names: ", i, agent_kwargs[i]['name'])

  env_kwargs = dict(swap_ports=False)
  if config.actor.async_envs:
    env_kwargs.update(
        num_steps=config.actor.num_env_steps,
        inner_batch_size=config.actor.inner_batch_size,
        enable_singles=config.actor.enable_singles,
    )
    print('num steps', config.actor.num_env_steps)

  build_actor = lambda: evaluators.RolloutWorker(
      agent_kwargs=agent_kwargs,
      dolphin_kwargs=dolphin_kwargs,
      env_kwargs=env_kwargs,
      num_envs=config.actor.num_envs,
      use_ray_envs=config.actor.ray_envs,
      async_envs=config.actor.async_envs,
      use_gpu=config.actor.gpu_inference,
      use_fake_envs=config.actor.use_fake_envs,
      agent_names=name_configuration_batch,
      # Rewards are overridden in the learner.
  )

  learner_manager = LearnerManager(
      config=config,
      learner=learner,
      #port=PORT,
      #enemy_port=ENEMY_PORT,
      build_actor=build_actor,
  )

  step_profiler = utils.Profiler()

  MINUTES_PER_FRAME = 60 * 60

  rev = lambda x: x[::-1]

  if config.opponent.type is OpponentType.SELF:
    '''ordered_name_combinations: list[tuple[str, str]] = []
    for i, n1 in enumerate(config.agent.name):
      for j, n2 in enumerate(config.agent.name):
        if i < j:
          ordered_name_combinations.append((n1, n2))

    ordered_name_combination_indices: dict[tuple[str, str], list[int]] = {
        name_combination: [] for name_combination in ordered_name_combinations
    }
    # TODO: would be easier to disallow unordered name combinations
    reversed_name_combination_indices: dict[tuple[str, str], list[int]] = {
        name_combination: [] for name_combination in ordered_name_combinations
    }

    for i, name_combination in enumerate(name_configuration_batch):
      if name_combination in ordered_name_combination_indices:
        ordered_name_combination_indices[name_combination].append(i)
      elif rev(name_combination) in reversed_name_combination_indices:
        reversed_name_combination_indices[rev(name_combination)].append(i)
      # We don't log mirror matches as the ko_diff will be 0.

    print('ordered_name_combinations:', ordered_name_combinations)
    print('ordered_name_combination_indices:', ordered_name_combination_indices)
    print('reversed_name_combination_indices:', reversed_name_combination_indices)'''

    def get_player_stats(states: Game) -> dict:
      """
      Computes the mean KO differential for each player across a batch of trajectories.
      Expects:
        - states: an array of shape [T, B] where each element is a Game namedtuple with p0, p1, p2, p3 fields.
      Uses the global variable name_configuration_batch (a list of length B where each element is a 4-tuple
      of player names corresponding to positions p0, p1, p2, p3).

      For each environment (rollout) in the batch:
        * Compute the per-player mean reward (pm_kos) over time.
        * Calculate team averages for team1 (positions 0 and 1) and team2 (positions 2 and 3).
        * For each player slot, compute the KO differential:
              diff = (team_avg - opponent_avg) * MINUTES_PER_FRAME
          where team_avg is the average reward for the player's own team and opponent_avg is the other team's.
        * Use the corresponding value from name_configuration_batch as the player's unique identifier.
      Finally, average the differences across all environments where the player appears.
      """
      # Compute rewards over time. tm_kos: [T, P, B]
      tm_kos = reward.compute_rewards(states, damage_ratio=0)
      # Average over time for each player and each environment. pm_kos: [4, B]
      pm_kos = tm_kos.mean(axis=0)
      B = pm_kos.shape[1]

      player_diffs = defaultdict(list)

      # Loop over each environment in the batch.
      for b in range(B):
          # Compute team averages for environment b.
          team1_avg = np.mean([pm_kos[0, b], pm_kos[1, b]])
          team2_avg = np.mean([pm_kos[2, b], pm_kos[3, b]])
          
          # Lookup the 4-tuple of player names for environment b from the global variable.
          names_tuple = name_configuration_batch[b]
          for p_idx in range(4):
              player_name = names_tuple[p_idx]
              # Compute KO differential for the player.
              if p_idx in (0, 1):
                  diff = (team1_avg - team2_avg) * MINUTES_PER_FRAME
              else:
                  diff = (team2_avg - team1_avg) * MINUTES_PER_FRAME
              player_diffs[player_name].append(diff)

      # Average the KO differentials for each player over all environments.
      stats = {player: float(np.mean(diffs)) for player, diffs in player_diffs.items()}
      return stats
    
    def get_character_stats(states: Game) -> dict:
      """
      Computes the mean knockouts for each character across a batch of trajectories.
      Expects `states` to be of type Game, where each field (p0, p1, p2, p3) is a Player with attributes
      (including 'character') stored as np.ndarrays of shape [T, B] (T timesteps, B environments).
      
      For each environment:
        - Computes each player's mean KO reward.
        - Retrieves the player's character from the first timestep using:
              player = getattr(states, f'p{p_idx}')
              player_character = player.character[0, b]
        - Converts the character code to a readable melee.Character name.
      
      Finally, aggregates the KO values per character across all environments.
      """
      tm_kos = reward.compute_rewards(states, damage_ratio=0)
      pm_kos = tm_kos.mean(axis=0)  # shape: [4, B]
      B = pm_kos.shape[1]
      
      char_kos = defaultdict(list)
      
      for b in range(B):
        for p_idx in range(4):
          # Retrieve the Player object from states.
          player = getattr(states, f'p{p_idx}')
          # Access the character code from the player's 'character' ndarray.
          player_character = player.character[0, b]
          char_enum = Character(int(player_character))
          char_name = char_enum.name  # Get the readable name.
          char_kos[char_name].append(pm_kos[p_idx, b])
      
      stats = {char: float(np.mean(values)) for char, values in char_kos.items()}
      return stats

  # TODO: log per-name stats for CPU and OTHER opponents

  def get_log_data(
      trajectories: list[evaluators.Trajectory],
      metrics: dict,
  ) -> dict:
    timings = {}

    # TODO: we shouldn't take the mean over these timings
    step_time = step_profiler.mean_time()
    steps_per_rollout = config.actor.num_envs * config.actor.rollout_length
    fps = len(trajectories) * steps_per_rollout / step_time
    mps = fps / (60 * 60)  # in-game minutes per second

    timings.update(
        rollout=learner_manager.rollout_profiler.mean_time(),
        learner=learner_manager.learner_profiler.mean_time(),
        reset=learner_manager.reset_profiler.mean_time(),
        total=step_time,
        fps=fps,
        mps=mps,
    )
    actor_timing = metrics['actor'].pop('timing')
    for key in ['env_pop', 'env_push']:
      timings[key] = actor_timing[key]
    for key in ['agent_pop', 'agent_step']:
      timings[key] = actor_timing[key][PORT]

    # Stack to shape [T, P, B] where P is the number of trajectories
    states: Game = utils.map_nt(
        lambda *xs: np.stack(xs, axis=1),
        *[t.states for t in trajectories])

    #p0_stats = reward.player_stats(states.p0, states.p1, states.stage)
    p0_stats = reward.team_stats([states.p0, states.p1], [states.p2, states.p3], states.stage)

    '''if config.opponent.type is OpponentType.SELF:
      # The second half of the batch just has the players reversed.
      deduplicated_states = utils.map_single_structure(
          lambda x: x[:, :, :batch_size], states)
      player_stats = get_player_stats(deduplicated_states)
      metrics.update(player_stats=player_stats)
      print("player stats: ", player_stats)

      character_stats = get_character_stats(deduplicated_states)
      metrics.update(character_stats=character_stats)
      print("character stats: ", character_stats)'''

    metrics.update(
        timings=timings,
        p0=p0_stats,
    )

    return metrics

  logger = Logger()

  def flush(step: int):
    metrics = logger.flush(step * steps_per_epoch)
    if metrics is None:
      return

    logging.info('Step: ' + str(step))

    timings: dict = metrics['timings']
    timing_str = ', '.join(
        ['{k}: {v:.3f}'.format(k=k, v=v) for k, v in timings.items()])
    logging.info(timing_str)

    ko_diff = metrics.get('ko_diff')
    if ko_diff is not None:
      logging.info(f'KO_diff_per_minute: {ko_diff:.3f}')

    learner_metrics = metrics['learner']
    pre_update = learner_metrics['ppo_step']['0']
    mean_actor_kl = pre_update['actor_kl']['mean']
    max_actor_kl = pre_update['actor_kl']['max']
    logging.info(f'actor_kl: mean={mean_actor_kl:.3g} max={max_actor_kl:.3g}')
    teacher_kl = pre_update['teacher_kl']
    logging.info(f'teacher_kl: {teacher_kl:.3g}')
    logging.info(f'uev: {learner_metrics["value"]["uev"]:.3f}')

  maybe_flush = utils.Periodically(flush, config.runtime.log_interval)

  # The OpponentType enum sadly needs to be converted.
  rl_config_jsonnable = dataclasses.asdict(config)
  rl_config_jsonnable = tf.nest.map_structure(
      lambda x: x.value if isinstance(x, enum.Enum) else x,
      rl_config_jsonnable
  )

  def save(step: int):
    # Note: this state is valid as an imitation state.
    combined_state = dict(
        state=learner.get_state(),
        config=teacher_state['config'],
        name_map=teacher_state['name_map'],
        step=step,
        rl_config=rl_config_jsonnable,
    )
    pickled_state = pickle.dumps(combined_state)

    save_path = pickle_path 
    if step % 100 == 0:
      save_path = pickle_path + '_' + str(step)
    logging.info('saving state to %s', save_path)
    with open(save_path, 'wb') as f:
      f.write(pickled_state)

    # TODO: save to s3?

  maybe_save = utils.Periodically(save, config.runtime.save_interval)

  try:
    steps_per_epoch = config.learner.ppo.num_batches

    reset_interval = config.runtime.reset_every_n_steps
    if reset_interval:
      reset_interval = reset_interval // steps_per_epoch

    # Optimizer burnin
    if step == 0:
      logging.info('Optimizer burnin')

      learning_rate.assign(0)
      for i in range(config.optimizer_burnin_steps // steps_per_epoch):
        learner_manager.step(0, ppo_steps=1)
        print("optimizer burnin step complete: ", i)
      learning_rate.assign(config.learner.learning_rate)

      logging.info('Value function burnin')

      for i in range(config.value_burnin_steps // steps_per_epoch):
        with step_profiler:
          trajectories, metrics = learner_manager.step(step, ppo_steps=0)

        if i > 0:
          logger.record(get_log_data(trajectories, metrics))
          maybe_flush(step)

        step += 1

    # Need flush here because logging structure changes based on ppo_steps.
    flush(step)

    logging.info('Main training loop')

    '''initial_weight = 0.001
    final_weight = 0.0001
    total_reduction_steps = 1000

    initial_step = 400
    final_step = initial_step + total_reduction_steps'''

    for i in range(config.runtime.max_step):
      # anneal the kl_teacher_weight from initial_weight to final_weight over total_reduction_steps
      '''if step >= initial_step and step < final_step:
        config.learner.kl_teacher_weight = initial_weight - ((initial_weight - final_weight) / total_reduction_steps) * (step - initial_step)
        print("lowered kl_teacher weight: ", config.learner.kl_teacher_weight)
      elif step >= final_step:
        config.learner.kl_teacher_weight = final_weight
        print("using final kl_teacher weight: ", config.learner.kl_teacher_weight)'''

      with step_profiler:
        if i > 0 and reset_interval and i % reset_interval == 0:
          logging.info('Resetting environments')
          learner_manager.reset_env()

        trajectories, metrics = learner_manager.step(step)

      if i > 0:
        logger.record(get_log_data(trajectories, metrics))
        maybe_flush(step)

      step += 1
      if step % 100 == 0:
        logging.info('saving checkpoint at step %d', step)
        save(step)
      else:
        maybe_save(step)

    save(step)

  finally:
    learner_manager.actor.stop()
