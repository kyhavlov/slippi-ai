"""Run parsing in the local filesystem.

It is assumed that everything is organized under a "root" directory:

Root
  Raw
  raw.json
  Parsed
  parsed.pkl
  meta.json

Raw contains .zip and .7z archives of .slp files, possibly nested under
subdirectories. The raw.json metadata file contains information about each
raw archive, including whether it has been processed. Once a raw archive has
been processed, it may be removed to save space.

The Parsed directory is populated by this script with a parquet file for each
processed .slp file. These files are named by the MD5 hash of the .slp file,
and are used by imitation learning. The parsed.pkl pickle file contains
metadata about each processed .slp in Parsed.

The meta.json file is created by scripts/make_local_dataset.py and is used by
imitation learning to know which files to train on.
TODO: consider merging meta.json and parsed.pkl

Usage: python slippi_db/parse_local.py --root=Root [--threads N] [--dry_run]

This will process all unprocessed .zip and .7z files in the Raw directory,
overwriting any existing files in Parsed, and will update parsed.pkl.
"""

import concurrent.futures
import functools
import json
import logging
import os
import pickle
import shutil
import sys
import tempfile
import time
import typing as tp
from contextlib import contextmanager
from typing import Optional

from absl import app, flags
import tqdm

from slippi_db import file_layout
from slippi_db import parse_peppi
from slippi_db import preprocessing
from slippi_db import utils
from slippi_db import parsing_utils
from slippi_db.parsing_utils import CompressionType


class RawArchive(tp.NamedTuple):
  name: str
  location: str  # 'local' or 'remote'
  source_root: str


def _archive_key(location: str, name: str) -> tuple[str, str]:
  return (location, name)


@contextmanager
def stage_archive(
    archive: RawArchive,
    local_root: str,
    staging_dir: Optional[str],
) -> tp.Iterator[str]:
  """Yield a local path to the archive, copying remote sources on-demand."""
  if archive.location == 'local':
    yield os.path.join(local_root, archive.name)
    return

  if staging_dir is None:
    staging_dir = os.path.join(local_root, '_remote_cache')

  os.makedirs(staging_dir, exist_ok=True)

  remote_path = os.path.join(archive.source_root, archive.name)
  with tempfile.TemporaryDirectory(dir=staging_dir) as tmpdir:
    local_path = os.path.join(tmpdir, os.path.basename(archive.name))
    shutil.copy2(remote_path, local_path)
    yield local_path

def parse_slp(
    file: utils.LocalFile,
    output_dir: str,
    tmpdir: Optional[str],
    compression: CompressionType = CompressionType.NONE,
    compression_level: Optional[int] = None,
) -> dict:
  slp_bytes = file.read()
  slp_size = len(slp_bytes)
  md5 = utils.md5(slp_bytes)

  result = dict(
      name=file.name,
      slp_md5=md5,
      slp_size=slp_size,
  )

  with tempfile.TemporaryDirectory(dir=tmpdir) as tmp_parent:
    path = os.path.join(tmp_parent, 'game.slp')
    with open(path, 'wb') as f:
      f.write(slp_bytes)
    del slp_bytes

    game = parse_peppi.read_slippi(path)
    metadata = preprocessing.get_metadata(game)
    is_training, reason = preprocessing.is_training_replay(metadata)

    result.update(metadata)  # nest?
    result.update(
        valid=True,
        is_training=is_training,
        not_training_reason=reason,
    )

    if is_training:
      game = parse_peppi.from_peppi(game)
      game_bytes = parsing_utils.convert_game(
        game, compression=compression, compression_level=compression_level)
      result.update(
          pq_size=len(game_bytes),
          compression=compression.value,
      )

      output_path = file_layout.ensure_parquet_directory(output_dir, md5)
      with open(output_path, 'wb') as f:
        f.write(game_bytes)

  return result

def parse_slp_safe(file: utils.LocalFile, *args, debug: bool = False, **kwargs):
  if debug:
    return parse_slp(file, *args, **kwargs)

  try:
    return parse_slp(file, *args, **kwargs)
  except KeyboardInterrupt:
    raise
  except BaseException as e:
    return dict(name=file.name, valid=False, reason=repr(e))
  # except:  # should be a catch-all, but sadly prevents KeyboardInterrupt?
  #   result.update(valid=False, reason='uncaught exception')


def parse_slp_with_index(index: int, *args, **kwargs):
  return index, parse_slp_safe(*args, **kwargs)

def _monitor_results(
    results_iter: tp.Iterable[dict],
    total_files: int,
    log_interval: int = 30,
) -> list[dict]:
  """Monitor parsing results and log progress periodically."""
  pbar = tqdm.tqdm(total=total_files, desc="Parsing", unit="slp", smoothing=0)

  last_log_time = 0
  successful_parses = 0
  last_error: Optional[tuple[str, str]] = None

  results: list[dict] = []

  for result in results_iter:
    if result['valid']:
      successful_parses += 1
    else:
      last_error = (result['name'], result['reason'])

    results.append(result)
    pbar.update(1)

    if time.time() - last_log_time > log_interval:
      last_log_time = time.time()
      success_rate = successful_parses / pbar.n
      logging.info(f'Success rate: {success_rate:.2%}')
      if last_error is not None:
        logging.error(f'Last error: {last_error}')
        last_error = None

  pbar.close()

  return results

def parse_files(
    files: list[utils.LocalFile],
    output_dir: str,
    tmpdir: Optional[str],
    num_threads: int = 1,
    compression_options: dict = {},
    log_interval: int = 30,
) -> list[dict]:
  parse_slp_kwargs = dict(
      output_dir=output_dir,
      tmpdir=tmpdir,
      **compression_options,
  )

  if num_threads == 1:
    def results_iter():
      for f in files:
        yield parse_slp(f, **parse_slp_kwargs)

    return _monitor_results(results_iter(), total_files=len(files), log_interval=log_interval)

  worker = functools.partial(parse_slp_safe, **parse_slp_kwargs)

  with concurrent.futures.ProcessPoolExecutor(num_threads) as pool:
    results_iter = pool.map(worker, files, chunksize=8)
    return _monitor_results(results_iter, total_files=len(files), log_interval=log_interval)

def parse_chunk(
    chunk: list[utils.LocalFile],
    output_dir: str,
    tmpdir: str,
    compression_options: dict = {},
    pool: Optional[concurrent.futures.ProcessPoolExecutor] = None,
) -> list[dict]:
  parse_slp_kwargs = dict(
      output_dir=output_dir,
      tmpdir=tmpdir,
      **compression_options,
  )

  if pool is None:
    results = []
    for file in chunk:
      results.append(parse_slp(file, **parse_slp_kwargs))
    return results
  else:
    futures = [
        pool.submit(parse_slp, f, **parse_slp_kwargs)
        for f in chunk]
    return [f.result() for f in futures]

def parse_7z_archive(
    archive: RawArchive,
    local_root: str,
    output_dir: str,
    tmpdir: str,
    num_threads: int = 1,
    compression_options: dict = {},
    chunk_size_gb: float = 0.5,
    in_memory: bool = True,
    staging_dir: Optional[str] = None,
) -> list[dict]:
  print(f"Processing 7z file: {archive.name}")
  results: list[dict] = []

  pool: Optional[concurrent.futures.ProcessPoolExecutor]
  if num_threads == 1:
    pool = None
  else:
    pool = concurrent.futures.ProcessPoolExecutor(num_threads)

  try:
    with stage_archive(archive, local_root, staging_dir) as local_path:
      file_size_gb = os.path.getsize(local_path) / 1024**3
      chunks = utils.traverse_7z_fast(local_path, chunk_size_gb=chunk_size_gb)
      if chunks:
        chunk_sizes = [len(c.files) for c in chunks]
        mean_chunk_size = sum(chunk_sizes) / len(chunks)
      else:
        mean_chunk_size = 0
      print(
          f"{archive.name}: size={file_size_gb:.2f} GB, "
          f"chunks={len(chunks)}, mean chunk files={mean_chunk_size:.1f}")

      for chunk in tqdm.tqdm(chunks, unit='chunk', desc=archive.name):
        with chunk.extract(in_memory) as files:
          chunk_results = parse_chunk(
              files,
              output_dir,
              tmpdir=tmpdir,
              compression_options=compression_options,
              pool=pool,
          )

        for result in chunk_results:
          result['raw'] = archive.name
        results.extend(chunk_results)
  finally:
    if pool is not None:
      pool.shutdown(cancel_futures=True)

  return results


def parse_zip_archive(
    archive: RawArchive,
    local_root: str,
    output_dir: str,
    tmpdir: str,
    num_threads: int = 1,
    compression_options: dict = {},
    log_interval: int = 30,
    staging_dir: Optional[str] = None,
) -> list[dict]:
  print(f"Processing zip file: {archive.name}")
  with stage_archive(archive, local_root, staging_dir) as local_path:
    files = utils.traverse_slp_files_zip(local_path)
    print(f"Found {len(files)} slp files in {archive.name}")
    results = parse_files(
        files,
        output_dir,
        tmpdir,
        num_threads,
        compression_options,
        log_interval,
    )

  for result in results:
    result['raw'] = archive.name

  return results

MD5_KEY = 'slp_md5'

def get_key(row: dict):
  if MD5_KEY in row:
    return row[MD5_KEY]

  return (row['raw'], row['name'])

def run_parsing(
    root: str,
    num_threads: int = 1,
    compression_options: dict = {},
    chunk_size_gb: float = 0.5,
    in_memory: bool = True,
    reprocess: bool = False,
    dry_run: bool = False,
    log_interval: int = 30,
    remote_raw_root: Optional[str] = None,
):
  # Cache tmp dir once
  tmpdir = utils.get_tmp_dir(in_memory=in_memory)

  raw_dir = os.path.join(root, 'Raw')
  os.makedirs(raw_dir, exist_ok=True)

  raw_db_path = os.path.join(root, 'raw.json')
  if os.path.exists(raw_db_path):
    with open(raw_db_path, 'r') as f:
      raw_db = json.load(f)
  else:
    raw_db = []

  for row in raw_db:
    row.setdefault('location', 'local')

  raw_by_key = {
      _archive_key(row['location'], row['name']): row
      for row in raw_db
  }

  archives_to_process: list[RawArchive] = []

  def register_archive(name: str, location: str, source_root: str):
    key = _archive_key(location, name)
    entry = raw_by_key.setdefault(
        key,
        dict(processed=False, name=name, location=location),
    )
    entry.setdefault('location', location)
    if reprocess or not entry['processed']:
      archives_to_process.append(
          RawArchive(name=name, location=location, source_root=source_root))

  for dirpath, dirnames, filenames in os.walk(raw_dir):
    dirnames[:] = [d for d in dirnames if d != '_remote_cache']
    reldirpath = os.path.relpath(dirpath, raw_dir)
    for name in filenames:
      relpath = os.path.join(reldirpath, name).removeprefix('./')
      register_archive(relpath, 'local', raw_dir)

  if remote_raw_root:
    if not os.path.exists(remote_raw_root):
      raise FileNotFoundError(
          f'Remote raw root {remote_raw_root} does not exist')
    for dirpath, _, filenames in os.walk(remote_raw_root):
      reldirpath = os.path.relpath(dirpath, remote_raw_root)
      for name in filenames:
        relpath = os.path.join(reldirpath, name).removeprefix('./')
        register_archive(relpath, 'remote', remote_raw_root)

  print(
      "To process:",
      [f"{archive.location}:{archive.name}" for archive in archives_to_process])

  if dry_run:
    return

  output_dir = os.path.join(root, 'Parsed')
  os.makedirs(output_dir, exist_ok=True)

  if tmpdir and not os.path.exists(tmpdir):
    os.makedirs(tmpdir, exist_ok=True)

  staging_dir = os.path.join(raw_dir, '_remote_cache') if remote_raw_root else None

  # Record slp metadata.
  # TODO: column-major would be more efficient
  slp_db_path = os.path.join(root, 'parsed.pkl')
  if os.path.exists(slp_db_path):
    with open(slp_db_path, 'rb') as f:
      slp_meta = pickle.load(f)
    print(f"Loaded slp metadata with {len(slp_meta)} records.")
  else:
    slp_meta = []

  by_key = {get_key(row): row for row in slp_meta}

  total_processed = 0
  total_valid = 0

  def flush_metadata():
    with open(raw_db_path, 'w') as f:
      raw_entries = [
          raw_by_key[key]
          for key in sorted(raw_by_key.keys())
      ]
      json.dump(raw_entries, f, indent=2)

    with open(slp_db_path, 'wb') as f:
      pickle.dump(list(by_key.values()), f)

  for archive in archives_to_process:
    if archive.name.endswith('.7z'):
      archive_results = parse_7z_archive(
          archive,
          raw_dir,
          output_dir,
          tmpdir,
          num_threads,
          compression_options,
          chunk_size_gb,
          in_memory,
          staging_dir,
      )
    elif archive.name.endswith('.zip'):
      archive_results = parse_zip_archive(
          archive,
          raw_dir,
          output_dir,
          tmpdir,
          num_threads,
          compression_options,
          log_interval,
          staging_dir,
      )
    else:
      logging.warning('Skipping unsupported archive type: %s', archive.name)
      continue

    total_processed += len(archive_results)
    total_valid += sum(r.get('valid', False) for r in archive_results)

    for result in archive_results:
      by_key[get_key(result)] = result

    raw_by_key[_archive_key(archive.location, archive.name)].update(
        processed=True,
    )

    flush_metadata()

    if archive_results:
      num_valid = sum(r['valid'] for r in archive_results)
      print(
          f"Processed {num_valid}/{len(archive_results)} valid files from {archive.name}.")

  if total_processed:
    print(f"Finished. Processed {total_valid}/{total_processed} valid files.")

if __name__ == '__main__':
  ROOT = flags.DEFINE_string('root', None, 'root directory', required=True)
  # MAX_FILES = flags.DEFINE_integer('max_files', None, 'max files to process')
  THREADS = flags.DEFINE_integer('threads', 1, 'number of threads')
  CHUNK_SIZE = flags.DEFINE_float('chunk_size', 0.5, 'max chunk size in GB')
  IN_MEMORY = flags.DEFINE_bool('in_memory', True, 'extract in memory')
  LOG_INTERVAL = flags.DEFINE_integer('log_interval', 30, 'seconds between progress logs')
  COMPRESSION = flags.DEFINE_enum_class(
      name='compression',
      default=parsing_utils.CompressionType.ZLIB,  # best one
      enum_class=parsing_utils.CompressionType,
      help='Type of compression to use.')
  COMPRESSION_LEVEL = flags.DEFINE_integer('compression_level', None, 'Compression level.')
  REPROCESS = flags.DEFINE_bool('reprocess', False, 'Reprocess raw archives.')
  DRY_RUN = flags.DEFINE_bool('dry_run', False, 'dry run')
  REMOTE_RAW_ROOT = flags.DEFINE_string(
      'remote_raw_root',
      None,
      'Optional remote Raw/ directory to stage archives from.')

  def main(_):
    run_parsing(
        ROOT.value,
        num_threads=THREADS.value,
        chunk_size_gb=CHUNK_SIZE.value,
        in_memory=IN_MEMORY.value,
        compression_options=dict(
            compression=COMPRESSION.value,
            compression_level=COMPRESSION_LEVEL.value,
        ),
        reprocess=REPROCESS.value,
        dry_run=DRY_RUN.value,
        log_interval=LOG_INTERVAL.value,
        remote_raw_root=REMOTE_RAW_ROOT.value,
    )

  app.run(main)
