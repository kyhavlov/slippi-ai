import copy

import numpy as np
from flax import nnx

from slippi_ai import eval_lib
from slippi_ai.flag_utils import dataclass_from_dict
from slippi_ai.jax import agents as jax_agents
from slippi_ai.jax import jax_utils
from slippi_ai.jax import networks
from slippi_ai.jax import tf_checkpoint
from slippi_ai.jax import train_lib
from slippi_ai.jax.rl import learner as learner_lib


def build_learner_and_actor(
    *,
    state: dict,
    batch_size: int,
    ppo_batches: int,
    ppo_epochs: int,
    learner_minibatch_size: int,
    learner_minibatch_scan_size: int,
    offload_minibatch_outputs: bool,
    learning_rate: float | None,
    policy_gradient_weight: float | None = None,
    kl_teacher_weight: float | None = None,
    value_cost: float | None = None,
    reward_halflife: float | None = None,
    reward_damage_ratio: float | None = None,
    reward_stalling_penalty: float | None = None,
    reward_stalling_threshold: float | None = None,
    reward_approaching_factor: float | None = None,
    reward_ledge_grab_penalty: float | None = None,
    reward_zelda_penalty: float | None = None,
    ppo_beta: float | None = None,
    ppo_epsilon: float | None = None,
    ppo_max_mean_actor_kl: float | None = None,
    post_update_eval_interval: int | None = None,
    revert_on_post_update_actor_kl: bool = False,
    optimizer_burnin_epochs: int | None = None,
    value_burnin_epochs: int | None = None,
    sample_temperature: float,
    learner_param_dtype: str = 'float32',
):
  policy = tf_checkpoint.load_policy_from_tf_state(
      state, param_dtype=learner_param_dtype)
  teacher = tf_checkpoint.load_policy_from_tf_state(
      state, param_dtype=learner_param_dtype)
  config_dict = tf_checkpoint.jax_config_from_tf_config(copy.deepcopy(state['config']))
  _add_missing_network_embed(config_dict['value_function']['network'])
  train_config = dataclass_from_dict(
      train_lib.Config,
      config_dict,
  )
  value_function = train_lib.value_function_from_config(
      train_config, rngs=nnx.Rngs(1))
  value_state = jax_utils.get_module_state(value_function, to_numpy=False)
  value_state = tf_checkpoint.cast_floating_state(
      value_state, learner_param_dtype)
  jax_utils.set_module_state(value_function, value_state)
  tf_checkpoint.set_compute_dtype(value_function, learner_param_dtype)

  learner_config = dataclass_from_dict(
      learner_lib.LearnerConfig,
      copy.deepcopy(state.get('rl_config', {}).get('learner', {})),
  )
  if ppo_batches > 0:
    learner_config.ppo.num_batches = ppo_batches
  if ppo_epochs > 0:
    learner_config.ppo.num_epochs = ppo_epochs
  if learning_rate is not None:
    learner_config.learning_rate = learning_rate
  if policy_gradient_weight is not None:
    learner_config.policy_gradient_weight = policy_gradient_weight
  if kl_teacher_weight is not None:
    learner_config.kl_teacher_weight = kl_teacher_weight
  if value_cost is not None:
    learner_config.value_cost = value_cost
  if reward_halflife is not None:
    learner_config.reward_halflife = reward_halflife
  if reward_damage_ratio is not None:
    learner_config.reward.damage_ratio = reward_damage_ratio
  if reward_stalling_penalty is not None:
    learner_config.reward.stalling_penalty = reward_stalling_penalty
  if reward_stalling_threshold is not None:
    learner_config.reward.stalling_threshold = reward_stalling_threshold
  if reward_approaching_factor is not None:
    learner_config.reward.approaching_factor = reward_approaching_factor
  if reward_ledge_grab_penalty is not None:
    learner_config.reward.ledge_grab_penalty = reward_ledge_grab_penalty
  if reward_zelda_penalty is not None:
    learner_config.reward.zelda_penalty = reward_zelda_penalty
  if ppo_beta is not None:
    learner_config.ppo.beta = ppo_beta
  if ppo_epsilon is not None:
    learner_config.ppo.epsilon = ppo_epsilon
  if ppo_max_mean_actor_kl is not None:
    learner_config.ppo.max_mean_actor_kl = ppo_max_mean_actor_kl
  if post_update_eval_interval is not None:
    learner_config.ppo.post_update_eval_interval = post_update_eval_interval
  if revert_on_post_update_actor_kl:
    learner_config.ppo.revert_on_post_update_actor_kl = True
  if optimizer_burnin_epochs is not None:
    learner_config.optimizer_burnin_epochs = optimizer_burnin_epochs
  if value_burnin_epochs is not None:
    learner_config.value_burnin_epochs = value_burnin_epochs
  learner_config.ppo.minibatch_size = learner_minibatch_size
  learner_config.ppo.minibatch_scan_size = learner_minibatch_scan_size
  learner_config.ppo.offload_minibatch_outputs = offload_minibatch_outputs

  learner = learner_lib.Learner(
      config=learner_config,
      policy=policy,
      teacher=teacher,
      value_function=value_function,
  )
  if 'state' in state:
    learner.restore_from_imitation(
        state['state'], param_dtype=learner_param_dtype)

  name_code = name_code_from_state(state, batch_size)
  actor = jax_agents.BasicAgent(
      policy=learner.policy,
      batch_size=batch_size,
      name_code=name_code,
      sample_kwargs=dict(temperature=sample_temperature),
      compile=True,
      pack_args=True,
  )
  return learner, actor, np.asarray(name_code, dtype=np.int32)


def _add_missing_network_embed(network_config: dict):
  network_config.setdefault('embed', {
      'name': 'simple',
      'simple': {},
      'enhanced': networks.EnhancedEmbedModule.default_config(),
  })


def name_code_from_state(state: dict, batch_size: int):
  names = eval_lib.get_name_from_rl_state(state)
  if names is None:
    return np.zeros(batch_size, dtype=np.int32)
  return np.asarray(
      [eval_lib.get_name_code(state, names[i % len(names)])
       for i in range(batch_size)],
      dtype=np.int32,
  )
