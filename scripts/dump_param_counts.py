#!/usr/bin/env python3
"""Print parameter counts for imitation config variants."""

import argparse
import dataclasses
from typing import Callable, Dict

import tensorflow as tf

from slippi_ai import data, embed, train_lib
from slippi_ai import value_function as vf_lib
from slippi_ai import saving


ConfigBuilder = Callable[[], train_lib.Config]


def _patch_dummy_for_mlpwrapper():
  def dummy(self, shape):
    return self._embed.dummy(shape)

  embed.MLPWrapper.dummy = dummy


def build_baseline_config() -> train_lib.Config:
  """Match scripts/imitation_doubles.sh defaults."""
  config = train_lib.Config()

  config.policy.delay = 18
  config.policy.train_value_head = False

  config.network['name'] = 'tx_like'
  tx = config.network['tx_like']
  tx.update(hidden_size=768, num_layers=2, ffw_multiplier=2)

  config.controller_head['name'] = 'autoregressive'
  config.controller_head['autoregressive'].update(
      residual_size=128, component_depth=2)

  config.embed.controller.axis_spacing = 32

  config.value_function.train_separate_network = True
  config.value_function.separate_network_config = True
  config.value_function.network['name'] = 'tx_like'
  vtx = config.value_function.network['tx_like']
  vtx.update(hidden_size=1024, num_layers=1, ffw_multiplier=2)

  return config


def build_v7_config() -> train_lib.Config:
  """Match scripts/imitation_doubles_v7.sh defaults."""
  config = train_lib.Config()

  config.policy.delay = 21
  config.policy.train_value_head = False

  config.network['name'] = 'tx_like'
  tx = config.network['tx_like']
  tx.update(hidden_size=512, num_layers=3, ffw_multiplier=2)

  config.controller_head['name'] = 'autoregressive'
  config.controller_head['autoregressive'].update(
      residual_size=128, component_depth=2)

  config.embed.controller.axis_spacing = 32
  config.embed.player.with_nana = True
  config.embed.player.legacy_jumps_left = False
  config.embed.with_randall_xy = True
  config.embed.items.type = embed.ItemsType.MLP
  config.embed.items.mlp_sizes = (128, 32)

  config.value_function.train_separate_network = True
  config.value_function.separate_network_config = True
  config.value_function.network['name'] = 'tx_like'
  vtx = config.value_function.network['tx_like']
  vtx.update(hidden_size=512, num_layers=1, ffw_multiplier=2)

  return config


PROFILES: Dict[str, ConfigBuilder] = {
    'baseline': build_baseline_config,
    'v7': build_v7_config,
}


def initialize_policy_and_value(config: train_lib.Config):
  config_dict = dataclasses.asdict(config)

  if config.embed.items.type == embed.ItemsType.MLP:
    _patch_dummy_for_mlpwrapper()

  policy = saving.policy_from_config(config_dict)
  policy.initialize_variables()

  value_fn = vf_lib.ValueFunction(
      network_config=config.value_function.network,
      embed_state_action=policy.embed_state_action,
  )

  batch_size = 1
  time_steps = 2 + policy.delay
  frames = data.Frames(
      state_action=policy.embed_state_action.dummy([time_steps, batch_size]),
      is_resetting=tf.fill([time_steps, batch_size], False),
      reward=tf.zeros([time_steps - 1, batch_size], tf.float32),
  )
  _ = value_fn.loss(frames, value_fn.initial_state(batch_size), discount=0.99)

  return policy, value_fn


def count_variables(variables) -> int:
  return sum(int(tf.size(v)) for v in variables)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--profile',
      choices=PROFILES.keys(),
      default='v7',
      help='Which imitation config to analyse.',
  )
  args = parser.parse_args()

  config = PROFILES[args.profile]()
  policy, value_fn = initialize_policy_and_value(config)

  policy_params = count_variables(policy.trainable_variables)
  value_params = count_variables(value_fn.trainable_variables)
  total_params = policy_params + value_params

  print(f"Profile: {args.profile}")
  print(f"Policy parameters: {policy_params:,}")
  print(f"Value parameters:  {value_params:,}")
  print(f"Total parameters:  {total_params:,}")


if __name__ == '__main__':
  main()
