"""Benchmark imitation data loading throughput."""

import dataclasses
import time

from absl import app, flags, logging
import fancyflags as ff

from slippi_ai import data as data_lib
from slippi_ai import flag_utils
from slippi_ai import train_lib


CONFIG = ff.DEFINE_dict(
    'config', **flag_utils.get_flags_from_dataclass(train_lib.Config))

NUM_BATCHES = flags.DEFINE_integer(
    'num_batches', 64, 'Number of measured batches.')
NUM_WARMUP = flags.DEFINE_integer(
    'num_warmup_batches', 4, 'Number of warmup batches before timing.')


def _prepare_replays(dataset_config: data_lib.DatasetConfig):
  train_replays, test_replays = data_lib.train_test_split(dataset_config)
  logging.info('Train replays: %d  Test replays: %d',
               len(train_replays), len(test_replays))
  return train_replays


def main(_):
  config = flag_utils.dataclass_from_dict(train_lib.Config, CONFIG.value)

  char_filters = {}
  for key in ['allowed_characters', 'allowed_opponents']:
    chars_string = getattr(config.dataset, key)
    char_filters[key] = data_lib.chars_from_string(chars_string)

  train_replays = _prepare_replays(config.dataset)

  data_config = dict(
      dataclasses.asdict(config.data),
      replays=train_replays,
      extra_frames=1 + config.policy.delay,
      name_map=train_lib.create_name_map(train_replays, config.max_names),
      **char_filters,
  )

  source = data_lib.make_source(**data_config)

  for _ in range(NUM_WARMUP.value):
    next(source)

  start = time.perf_counter()
  for _ in range(NUM_BATCHES.value):
    next(source)
  duration = time.perf_counter() - start

  frames = (config.data.batch_size *
            (config.data.unroll_length + 1) * NUM_BATCHES.value)
  logging.info('Batches: %d  Duration: %.3fs  Frames/s: %.1f',
               NUM_BATCHES.value, duration, frames / duration)

  close = getattr(source, 'close', None)
  if callable(close):
    close()


if __name__ == '__main__':
  app.run(main)
