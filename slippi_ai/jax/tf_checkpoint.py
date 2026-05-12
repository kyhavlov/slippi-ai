"""Utilities for loading legacy TensorFlow policy checkpoints into JAX."""

from __future__ import annotations

import copy
import typing as tp

import numpy as np
from flax import nnx
import tree

from slippi_ai.flag_utils import dataclass_from_dict
from slippi_ai.jax import controller_heads, embed, jax_utils, networks, policies


def _set_path(params: dict, path: str, value: np.ndarray):
  cursor = params
  parts = path.split('/')
  for part in parts[:-1]:
    key = int(part) if part.isdecimal() and int(part) in cursor else part
    cursor = cursor[key]
  last = parts[-1]
  key = int(last) if last.isdecimal() and int(last) in cursor else last
  cursor[key] = np.asarray(value)


def _split_gates(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  i, f, g, o = np.split(np.asarray(x), 4, axis=-1)
  return i, f, g, o


def jax_config_from_tf_config(config: dict) -> dict:
  """Return a JAX-compatible config dict for the current TF checkpoint format."""
  config = copy.deepcopy(config)

  controller = config['embed']['controller']
  if 'type' not in controller:
    config['embed']['controller'] = {
        'type': 'default',
        'default': controller,
    }

  network = config['network']
  network.setdefault('embed', {
      'name': 'simple',
      'simple': {},
      'enhanced': networks.EnhancedEmbedModule.default_config(),
  })
  network['embed'].setdefault('simple', {})
  network['embed'].setdefault(
      'enhanced', networks.EnhancedEmbedModule.default_config())

  config.setdefault('seed', 0)
  return config


def policy_from_tf_config(
    config: dict,
    *,
    rngs: nnx.Rngs | None = None,
) -> policies.Policy:
  config = jax_config_from_tf_config(config)
  if rngs is None:
    rngs = nnx.Rngs(config.get('seed', 0))

  embed_config = dataclass_from_dict(embed.EmbedConfig, config['embed'])
  policy_config = dataclass_from_dict(policies.PolicyConfig, config['policy'])

  network = networks.build_embed_network(
      rngs=rngs,
      embed_config=embed_config,
      num_names=config['max_names'],
      network_config=config['network'],
      opponent_pooling=policy_config.opponent_pooling,
  )
  controller_head = controller_heads.construct(
      rngs=rngs,
      input_size=network.output_size,
      embed_controller=embed_config.controller.make_embedding(),
      **config['controller_head'],
  )
  return policies.Policy(
      network=network,
      controller_head=controller_head,
      delay=policy_config.delay,
  )


def convert_policy_params(
    policy: policies.Policy,
    tf_policy_state: tp.Any,
) -> dict:
  """Convert TF policy tuple leaves into a named JAX policy state dict."""
  tf_leaves = [np.asarray(x) for x in tree.flatten(tf_policy_state)]
  params = jax_utils.get_module_state(policy)

  if len(tf_leaves) != 145:
    raise ValueError(f'Expected 145 TF policy leaves, got {len(tf_leaves)}')

  # Controller head. TF stores decoder before encoder for each component; the
  # JAX module owns the same arrays under explicit encoder/decoder names.
  for i in range(13):
    tf_base = 4 + i * 8
    jax_base = f'_controller_head/res_blocks/{i}'
    _set_path(params, f'{jax_base}/decoder/bias', tf_leaves[tf_base])
    _set_path(params, f'{jax_base}/decoder/kernel', tf_leaves[tf_base + 1])
    _set_path(params, f'{jax_base}/_encoder/layers/0/bias', tf_leaves[tf_base + 2])
    _set_path(params, f'{jax_base}/_encoder/layers/0/kernel', tf_leaves[tf_base + 3])
    _set_path(params, f'{jax_base}/_encoder/layers/2/bias', tf_leaves[tf_base + 4])
    _set_path(params, f'{jax_base}/_encoder/layers/2/kernel', tf_leaves[tf_base + 5])
    _set_path(params, f'{jax_base}/_encoder/layers/4/bias', tf_leaves[tf_base + 6])
    _set_path(params, f'{jax_base}/_encoder/layers/4/kernel', tf_leaves[tf_base + 7])

  _set_path(params, '_controller_head/to_residual/bias', tf_leaves[108])
  _set_path(params, '_controller_head/to_residual/kernel', tf_leaves[109])

  # Item MLP.
  item_base = 'network/_embed_module/_embedding_modules/0/_mlp'
  _set_path(params, f'{item_base}/layers/0/bias', tf_leaves[110])
  _set_path(params, f'{item_base}/layers/0/kernel', tf_leaves[111])
  _set_path(params, f'{item_base}/layers/2/bias', tf_leaves[112])
  _set_path(params, f'{item_base}/layers/2/kernel', tf_leaves[113])

  # Opponent pooling.
  _set_path(params, 'network/_input_preprocessor/_opp_enc/layers/0/bias', tf_leaves[0])
  _set_path(params, 'network/_input_preprocessor/_opp_enc/layers/0/kernel', tf_leaves[1])
  _set_path(params, 'network/_input_preprocessor/_process_set_context/layers/0/bias', tf_leaves[2])
  _set_path(params, 'network/_input_preprocessor/_process_set_context/layers/0/kernel', tf_leaves[3])

  # Main recurrent stack encoder.
  _set_path(params, 'network/_network/_layers/0/_module/bias', tf_leaves[114])
  _set_path(params, 'network/_network/_layers/0/_module/kernel', tf_leaves[115])

  for layer_index in range(3):
    tf_base = 116 + layer_index * 9
    recurrent_layer = 1 + layer_index * 2
    ffw_layer = recurrent_layer + 1
    recurrent_base = f'network/_network/_layers/{recurrent_layer}/_net/_core'
    w_h_i, w_h_f, w_h_g, w_h_o = _split_gates(tf_leaves[tf_base])
    w_i_i, w_i_f, w_i_g, w_i_o = _split_gates(tf_leaves[tf_base + 1])
    b_i, b_f, b_g, b_o = _split_gates(tf_leaves[tf_base + 2])

    _set_path(params, f'{recurrent_base}/hi/kernel', w_h_i)
    _set_path(params, f'{recurrent_base}/hf/kernel', w_h_f)
    _set_path(params, f'{recurrent_base}/hg/kernel', w_h_g)
    _set_path(params, f'{recurrent_base}/ho/kernel', w_h_o)
    _set_path(params, f'{recurrent_base}/ii/kernel', w_i_i)
    _set_path(params, f'{recurrent_base}/if_/kernel', w_i_f)
    _set_path(params, f'{recurrent_base}/ig/kernel', w_i_g)
    _set_path(params, f'{recurrent_base}/io/kernel', w_i_o)
    _set_path(params, f'{recurrent_base}/hi/bias', b_i)
    _set_path(params, f'{recurrent_base}/hf/bias', b_f)
    _set_path(params, f'{recurrent_base}/hg/bias', b_g)
    _set_path(params, f'{recurrent_base}/ho/bias', b_o)

    ffw_base = f'network/_network/_layers/{ffw_layer}/_module'
    _set_path(params, f'{ffw_base}/layernorm/bias', tf_leaves[tf_base + 3])
    _set_path(params, f'{ffw_base}/layernorm/scale', tf_leaves[tf_base + 4])
    _set_path(params, f'{ffw_base}/linear1/bias', tf_leaves[tf_base + 5])
    _set_path(params, f'{ffw_base}/linear1/kernel', tf_leaves[tf_base + 6])
    _set_path(params, f'{ffw_base}/linear2/bias', tf_leaves[tf_base + 7])
    _set_path(params, f'{ffw_base}/linear2/kernel', tf_leaves[tf_base + 8])

  return params


def load_policy_from_tf_state(state: dict) -> policies.Policy:
  policy = policy_from_tf_config(state['config'])
  params = convert_policy_params(policy, state['state']['policy'])
  jax_utils.set_module_state(policy, params)
  return policy
