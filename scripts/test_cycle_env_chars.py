"""Test character/stage combinations in ffw mode as some cause dolphin errors."""

import dataclasses
import itertools
import json
import time
import random
import typing as tp
import ray
from ray.runtime_env import RuntimeEnv

from absl import app
from absl import flags
import fancyflags as ff
import tqdm

import melee
from melee import Controller, Character

from slippi_ai import controller_lib
from slippi_ai import flag_utils, envs, utils
from slippi_ai import dolphin as dolphin_lib
from slippi_ai import evaluators, eval_lib, saving
from slippi_ai.rl import run_lib

PORTS = (1, 2, 3, 4)

OUTPUT = flags.DEFINE_string(
    'output', 'test_ffw_results.json', 'output path')

DEBUG = flags.DEFINE_bool('debug', False, 'Enter ipdb on error.')

DEFAULT_CONFIG = run_lib.Config()

CONFIG = ff.DEFINE_dict(
    'config',
    **flag_utils.get_flags_from_default(DEFAULT_CONFIG))

@dataclasses.dataclass
class EnvConfig:
  rollout_length: int = 200
  num_envs: int = 2
  run_async: bool = False
  inner_batch_size: int = 1

ENV = ff.DEFINE_dict('env', **flag_utils.get_flags_from_dataclass(EnvConfig))

CHARACTER_WEIGHTINGS = {
      Character.FOX: 2000,
      Character.FALCO: 500,
      Character.MARTH: 500,
      Character.SHEIK: 750,
      Character.PEACH: 500,
      Character.CPTFALCON: 500,
      Character.JIGGLYPUFF: 500,
      Character.PIKACHU: 200,
      Character.YOSHI: 200,
      Character.POPO: 50,
      Character.SAMUS: 50,
      Character.DK: 50,
      Character.LUIGI: 50,
      Character.DOC: 25,
      Character.MARIO: 25,
      Character.YLINK: 25,
      Character.LINK: 25,
      Character.GAMEANDWATCH: 25,
      Character.NESS: 5,
      Character.ROY: 5,
      Character.MEWTWO: 5,
      Character.PICHU: 5,
      Character.BOWSER: 5,
}

def main(_):
  eval_lib.disable_gpus()
  config = flag_utils.dataclass_from_dict(run_lib.Config, CONFIG.value)
  env_config = flag_utils.dataclass_from_dict(EnvConfig, ENV.value)
  agent_config = config.agent.get_kwargs()

  dolphin_kwargs = dict(
      players={port: dolphin_lib.AI(character_weight_table=CHARACTER_WEIGHTINGS) for port in range(1, 5)},
      **config.dolphin.to_kwargs(),
  )

  env_kwargs = dict(
      swap_ports=False,
      #num_retries=0,
      num_steps=config.actor.num_env_steps,
      inner_batch_size=config.actor.inner_batch_size,
  )

  for i in range(2):
    characters = random.choices(
        list(CHARACTER_WEIGHTINGS.keys()),
        weights=list(CHARACTER_WEIGHTINGS.values()),
        k=4,
    )

    print(characters)

  names = str.split("Ralph,Tempo,Woopty,xRunRiot,Darkatma,MisterGW", ",")
  
  names = config.agent.name
  batch_size = env_config.num_envs
  print("batch size=", batch_size)

  # Generate all possible permutations of 4 players
  all_permutations = list(itertools.product(names, repeat=4))
  num_permutations = len(all_permutations)

  # Create the batch by cycling through the permutations in a round-robin manner
  name_configuration_batch = [
      all_permutations[(i * num_permutations // batch_size + i) % num_permutations]
      for i in range(batch_size)
  ]

  print(name_configuration_batch)
  print(len(name_configuration_batch))

  name_totals = {name: 0 for name in names}
  for permutation in name_configuration_batch:
    for name in permutation:
      name_totals[name] += 1

  print(name_totals)
  #print("name batches:", p1_names, p2_names)

  agent_kwargs: tp.Mapping[int, dict] = {}
  for i in range(1, 5):
    agent_kwargs[i] = dict(
        name=[name_configuration_batch[j][i-1] for j in range(batch_size)],
        **agent_config,
    )
    print(agent_kwargs[i]['name'])

  ray.init(runtime_env={"working_dir": "."})

  print(ray.cluster_resources())

  worker = evaluators.RolloutWorker(
      agent_kwargs=agent_kwargs,
      dolphin_kwargs=dolphin_kwargs,
      env_kwargs=env_kwargs,
      num_envs=batch_size,
      async_envs=False,
      #use_gpu=True,
      use_fake_envs=False,
      use_ray_envs=True,
      # Rewards are overridden in the learner.
  )

  worker.start()
  start_time = time.perf_counter()
  for i in range(100):
    start = time.perf_counter()
    worker.rollout(env_config.rollout_length)

    elapsed_time = time.perf_counter() - start
    n = i + 1
    time_per_item = elapsed_time / n
    time_left = time_per_item * (100 - n)
    print('Rollout time: ', elapsed_time)
    print(f'Estimated time left: {time_left:.0f}')
    

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None
  app.run(main)
