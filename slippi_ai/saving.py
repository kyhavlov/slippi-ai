import dataclasses
import pickle

import logging
import numpy as np
import tree
import tensorflow as tf

from slippi_ai import (
    data,
    embed,
    opponent_pooling as opponent_pooling_lib,
    policies,
    networks,
    controller_heads,
    embed,
    s3_lib,
)
from slippi_ai.flag_utils import dataclass_from_dict

VERSION = 5

def upgrade_config(config: dict):
  """Upgrades a config to the latest version."""

  if config.get('version') is None:
    assert 'policy' not in config
    config['policy'] = dict(
      train_value_head=False,
    )
    config['version'] = 1
    logging.warning('Upgraded config to version 1')

  if config['version'] == 1:
    if 'value_function' not in config:
      config['value_function'] = dict(
        train_separate_network=False,
      )

    config['version'] = 2
    logging.warning('Upgraded config version 1 -> 2')

  if config['version'] == 2:
    assert 'embed' not in config
    old_embed_config = embed.EmbedConfig(
        num_players=2,
        player=embed.PlayerConfig(
            xy_scale=0.05,
            shield_scale=0.01,
            speed_scale=0.5,
            with_speeds=False,
            with_controller=False,
        ),
        controller=embed.ControllerConfig(
            axis_spacing=16,
            shoulder_spacing=4,
        )
    )
    config['embed'] = dataclasses.asdict(old_embed_config)
    config['version'] = 3
    logging.warning('Upgraded config version 2 -> 3')

  if config['version'] == 3:
    embed_cfg = config['embed']
    embed_cfg.setdefault('with_randall_phase', True)
    embed_cfg.setdefault('with_randall_xy', False)
    if 'items' not in embed_cfg:
      embed_cfg['items'] = dataclasses.asdict(embed.ItemsConfig())

    player_cfg = embed_cfg.setdefault('player', {})
    player_cfg.setdefault('with_nana', False)
    player_cfg.setdefault('legacy_jumps_left', True)

    config['version'] = 4
    logging.warning('Upgraded config version 3 -> 4')

  if config['version'] == 4:
    policy_cfg = config.setdefault('policy', {})
    policy_cfg.setdefault(
        'opponent_pooling',
        dataclasses.asdict(opponent_pooling_lib.OpponentPoolingConfig()),
    )
    vf_cfg = config.setdefault('value_function', {})
    vf_cfg.setdefault(
        'opponent_pooling',
        dataclasses.asdict(opponent_pooling_lib.OpponentPoolingConfig()),
    )

    config['version'] = 5
    logging.warning('Upgraded config version 4 -> 5')

  embed_cfg = config.get('embed')
  if isinstance(embed_cfg, dict):
    embed_cfg.setdefault('with_randall_phase', True)

  assert config['version'] == VERSION
  return config


def build_policy(
  controller_head_config: dict,
  network_config: dict,
  num_names: int,
  embed_controller: embed.Embedding,
  embed_game: embed.Embedding,
  **policy_kwargs,
) -> policies.Policy:
  controller_head_config = dict(
      controller_head_config,
      embed_controller=embed_controller)

  return policies.Policy(
      networks.construct_network(**network_config),
      controller_heads.construct(**controller_head_config),
      embed_game=embed_game,
      num_names=num_names,
      **policy_kwargs,
  )

def policy_from_config(config: dict) -> policies.Policy:
  # TODO: Take config dataclasses instead of dictionaries
  config = upgrade_config(config)

  return build_policy(
      controller_head_config=config['controller_head'],
      network_config=config['network'],
      num_names=config['max_names'],
      embed_controller=embed.get_controller_embedding(
          **config['embed']['controller']),
      embed_game=embed.make_game_embedding(
          player_config=config['embed']['player'],
          num_players=config['embed'].get('num_players', 4),
          with_randall_phase=config['embed'].get('with_randall_phase', True),
          with_randall_xy=config['embed'].get('with_randall_xy', False),
          items_config=dataclass_from_dict(
              embed.ItemsConfig, config['embed'].get('items', {}))),
      **config['policy'],
  )

def load_policy_from_state(state: dict) -> policies.Policy:
  policy = policy_from_config(state['config'])
  policy.initialize_variables()

  # assign using saved params
  params = state['state']['policy']

  def assign_compatible(var: tf.Variable, val):
    val_arr = val
    if isinstance(val_arr, tf.Tensor):
      val_arr = val_arr.numpy()
    val_arr = np.asarray(val_arr)

    if tuple(var.shape) == tuple(val_arr.shape):
      var.assign(val_arr)
      return

    if var.shape.rank != val_arr.ndim:
      raise ValueError(
          f"Cannot assign {var.name}: rank mismatch {var.shape} vs {val_arr.shape}")

    # Allow old checkpoints that had smaller input embeddings: pad/truncate on the
    # leading dimension when the trailing dimensions match.
    if var.shape.rank == 2 and int(var.shape[1]) == int(val_arr.shape[1]):
      target0 = int(var.shape[0])
      if val_arr.shape[0] < target0:
        pad0 = target0 - int(val_arr.shape[0])
        val_arr = np.pad(val_arr, [(0, pad0), (0, 0)], mode="constant")
      elif val_arr.shape[0] > target0:
        val_arr = val_arr[:target0, :]
      var.assign(val_arr)
      return

    if var.shape.rank == 1:
      target0 = int(var.shape[0])
      if val_arr.shape[0] < target0:
        pad0 = target0 - int(val_arr.shape[0])
        val_arr = np.pad(val_arr, [(0, pad0)], mode="constant")
      elif val_arr.shape[0] > target0:
        val_arr = val_arr[:target0]
      var.assign(val_arr)
      return

    raise ValueError(
        f"Cannot assign {var.name}: shape mismatch {var.shape} vs {val_arr.shape}")

  tree.map_structure(
      assign_compatible,
      policy.variables, params)

  return policy

def load_state_from_s3(tag: str) -> dict:
  key = s3_lib.get_keys(tag).combined
  store = s3_lib.get_store()
  obj = store.get(key)
  return pickle.loads(obj)

def load_policy_from_s3(tag: str) -> policies.Policy:
  state = load_state_from_s3(tag)
  return load_policy_from_state(state)

def load_state_from_disk(path: str) -> dict:
  with open(path, 'rb') as f:
    return pickle.load(f)

def load_policy_from_disk(path: str) -> policies.Policy:
  state = load_state_from_disk(path)
  return load_policy_from_state(state)
