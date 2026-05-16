"""One-way conversion from legacy TF checkpoints to native JAX RL checkpoints."""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

from flax import nnx

from slippi_ai.flag_utils import dataclass_from_dict
from slippi_ai.jax import jax_utils
from slippi_ai.jax import networks
from slippi_ai.jax import saving
from slippi_ai.jax import tf_checkpoint
from slippi_ai.jax import train_lib
from slippi_ai.jax.rl import learner as learner_lib


def convert_state(
    source_state: dict,
    *,
    learner_param_dtype: str = 'float32',
) -> dict:
  config = tf_checkpoint.jax_config_from_tf_config(source_state['config'])
  config['version'] = saving.VERSION
  _add_missing_network_embed(config['value_function']['network'])

  train_config = dataclass_from_dict(train_lib.Config, copy.deepcopy(config))
  policy = tf_checkpoint.load_policy_from_tf_state(
      source_state,
      param_dtype=learner_param_dtype,
  )
  teacher = tf_checkpoint.load_policy_from_tf_state(
      source_state,
      param_dtype=learner_param_dtype,
  )
  value_function = train_lib.value_function_from_config(
      train_config,
      rngs=nnx.Rngs(1),
  )

  value_state = jax_utils.get_module_state(value_function)
  value_state = tf_checkpoint.cast_floating_state(
      value_state,
      learner_param_dtype,
  )
  jax_utils.set_module_state(value_function, value_state)
  tf_checkpoint.set_compute_dtype(value_function, learner_param_dtype)

  learner_config = dataclass_from_dict(
      learner_lib.LearnerConfig,
      copy.deepcopy(source_state.get('rl_config', {}).get('learner', {})),
  )
  learner = learner_lib.Learner(
      config=learner_config,
      teacher=teacher,
      policy=policy,
      value_function=value_function,
  )
  learner.restore_from_imitation(
      source_state['state'],
      param_dtype=learner_param_dtype,
  )

  converted = dict(
      state=learner.get_state(),
      config=config,
      name_map=copy.deepcopy(source_state.get('name_map', {})),
      step=int(source_state.get('step', source_state['state'].get('step', 0))),
  )
  if 'rl_config' in source_state:
    converted['rl_config'] = copy.deepcopy(source_state['rl_config'])
  return converted


def convert_file(
    source_path: str | Path,
    output_path: str | Path,
    *,
    learner_param_dtype: str = 'float32',
) -> None:
  source_path = Path(source_path)
  output_path = Path(output_path)
  with source_path.open('rb') as f:
    source_state = pickle.load(f)
  converted = convert_state(
      source_state,
      learner_param_dtype=learner_param_dtype,
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open('wb') as f:
    pickle.dump(converted, f)


def _add_missing_network_embed(network_config: dict) -> None:
  network_config.setdefault('embed', {
      'name': 'simple',
      'simple': {},
      'enhanced': networks.EnhancedEmbedModule.default_config(),
  })
