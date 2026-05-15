#!/usr/bin/env python3
"""Report imitation sampling breakdown for a single epoch."""

from collections import Counter

from absl import app
import fancyflags as ff
import melee

from slippi_ai import flag_utils
from slippi_ai import train_lib
from slippi_ai import data as data_lib


CONFIG = ff.DEFINE_dict(
    'config', **flag_utils.get_flags_from_dataclass(train_lib.Config))


def _character_name(character_id: int) -> str:
  try:
    return melee.Character(character_id).name.lower()
  except ValueError:
    return str(character_id)


def _summarize(samples):
  mode_counts: Counter[str] = Counter()
  character_counts = {
      'singles': Counter(),
      'doubles': Counter(),
      'unknown': Counter(),
  }

  for info in samples:
    meta = getattr(info, 'meta', ())
    is_singles = getattr(meta, 'is_singles', None)
    if is_singles is None:
      bucket = 'unknown'
    elif is_singles:
      bucket = 'singles'
    else:
      bucket = 'doubles'

    mode_counts[bucket] += 1

    try:
      character = info.main_player.character
    except Exception:
      character_counts[bucket]['unlabelled'] += 1
      continue
    character_counts[bucket][_character_name(character)] += 1

  return mode_counts, character_counts


def _print_mode_breakdown(dataset_counts, sample_counts):
  dataset_total = sum(dataset_counts.values())
  sample_total = sum(sample_counts.values())
  print('Mode breakdown (dataset -> sampled epoch):')
  for bucket in ('singles', 'doubles', 'unknown'):
    dataset_count = dataset_counts.get(bucket, 0)
    dataset_pct = (dataset_count / dataset_total * 100.0) if dataset_total else 0.0
    sample_count = sample_counts.get(bucket, 0)
    sample_pct = (sample_count / sample_total * 100.0) if sample_total else 0.0
    print(
        f'  {bucket:8s}: '
        f'dataset={dataset_count:5d} ({dataset_pct:5.1f}%)  '
        f'epoch={sample_count:5d} ({sample_pct:5.1f}%)')


def _print_character_breakdown(dataset_counts, sample_counts):
  for bucket in ('singles', 'doubles', 'unknown'):
    dataset_counter = dataset_counts[bucket]
    sample_counter = sample_counts[bucket]
    if not dataset_counter and not sample_counter:
      continue

    dataset_total = sum(dataset_counter.values())
    sample_total = sum(sample_counter.values())
    heading = 'Unknown-mode character counts' if bucket == 'unknown' else f'{bucket.title()} character distribution'
    print(f'\n{heading} (dataset -> sampled epoch):')

    characters = set(dataset_counter.keys()) | set(sample_counter.keys())
    ordered = sorted(
        characters,
        key=lambda name: (
            -dataset_counter.get(name, 0),
            -sample_counter.get(name, 0),
            name,
        ))
    for name in ordered:
      dataset_val = dataset_counter.get(name, 0)
      dataset_pct = (dataset_val / dataset_total * 100.0) if dataset_total else 0.0
      sample_val = sample_counter.get(name, 0)
      sample_pct = (sample_val / sample_total * 100.0) if sample_total else 0.0
      print(
          f'  {name:12s}: '
          f'dataset={dataset_val:5d} ({dataset_pct:5.1f}%)  '
          f'epoch={sample_val:5d} ({sample_pct:5.1f}%)')


def main(_):
  config = flag_utils.dataclass_from_dict(train_lib.Config, CONFIG.value)

  train_replays, _ = data_lib.train_test_split(config.dataset)
  if not train_replays:
    raise ValueError('No training replays available. Check dataset configuration.')

  sample_size = len(train_replays)
  iterator = data_lib.replay_stream(
      train_replays,
      balance_characters=config.data.balance_characters,
      balance_singles_doubles=config.data.balance_singles_doubles,
      character_balance_ratio=config.data.character_balance_ratio,
  )
  samples = [next(iterator) for _ in range(sample_size)]

  dataset_mode_counts, dataset_character_counts = _summarize(train_replays)
  sample_mode_counts, sample_character_counts = _summarize(samples)

  print(f'Epoch sample size: {sample_size}')
  print(f'Character balance ratio: {config.data.character_balance_ratio:.3f}')
  _print_mode_breakdown(dataset_mode_counts, sample_mode_counts)
  _print_character_breakdown(dataset_character_counts, sample_character_counts)


if __name__ == '__main__':
  app.run(main)
