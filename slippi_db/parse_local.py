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

import traceback
import concurrent.futures
import json
import os
import pickle
from typing import Optional
import sys
import tempfile
import subprocess

from absl import app, flags
import tqdm

import peppi_py

from slippi_db import parse_peppi
from slippi_db import preprocessing
from slippi_db import utils
from slippi_db import parsing_utils
from slippi_db.parsing_utils import CompressionType
from slippi_db import file_layout

class ArrowJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "to_pylist"):
            return obj.to_pylist()  # Handle Arrow Arrays
        if hasattr(obj, "as_py"):
            return obj.as_py()  # Handle Arrow Scalars
        return super().default(obj)

def parse_slp(
    file: utils.LocalFile,
    output_dir: str,
    tmpdir: str,
    compression: CompressionType = CompressionType.NONE,
    compression_level: Optional[int] = None,
) -> dict:
  result = dict(name=file.name)

  #print(f"Processing {file.name} ({tmpdir})...")

  try:
    with file.extract(tmpdir) as path:
      #print(f"path: {path}")
      with open(path, 'rb') as f:
        slp_bytes = f.read()
        slp_size = len(slp_bytes)
        md5 = utils.md5(slp_bytes)
        del slp_bytes

      result.update(
          slp_md5=md5,
          slp_size=slp_size,
      )

      game = peppi_py.read_slippi(path)
      metadata = preprocessing.get_metadata(game)
      is_training, reason = preprocessing.is_training_replay(metadata)

      result.update(metadata)  # nest?
      result.update(
          valid=True,
          is_training=is_training,
          not_training_reason=reason,
      )

      # log the game file, is_training and reason
      # print(f"{file.name} {is_training} {reason}")

      if is_training:
        game = parse_peppi.from_peppi(game)
        game_bytes = parsing_utils.convert_game(
          game, compression=compression, compression_level=compression_level)
        result.update(
            pq_size=len(game_bytes),
            compression=compression.value,
        )

        # TODO: consider writing to raw_name/slp_name
        parquet_path = file_layout.ensure_parquet_directory(output_dir, md5)
        with open(parquet_path, 'wb') as f:
          f.write(game_bytes)

  except KeyboardInterrupt as e:
    raise
  except BaseException as e:
    result.update(valid=False, reason=repr(e))
    #print(f"exception: {repr(e)}\n{traceback.format_exc()}")
  # except:  # should be a catch-all, but sadly prevents KeyboardInterrupt?
  #   result.update(valid=False, reason='uncaught exception')

  return result

def parse_files(
    files: list[utils.LocalFile],
    output_dir: str,
    tmpdir: str,
    pool: Optional[concurrent.futures.ProcessPoolExecutor] = None,
    compression_options: dict = {},
) -> list[dict]:
  parse_slp_kwargs = dict(
      output_dir=output_dir,
      tmpdir=tmpdir,
      **compression_options,
  )

  if pool is None:
    return [
        parse_slp(f, **parse_slp_kwargs)
        for f in tqdm.tqdm(files, unit='slp')]

  try:
    futures = [
        pool.submit(parse_slp, f, **parse_slp_kwargs)
        for f in files]
    as_completed = concurrent.futures.as_completed(futures)
    results = [
        f.result() for f in
        tqdm.tqdm(as_completed, total=len(files), smoothing=0, unit='slp')]
    return results
  except KeyboardInterrupt:
    print('KeyboardInterrupt, shutting down')
    raise

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

def parse_7zs(
    raw_dir: str,
    to_process: list[str],
    output_dir: str,
    num_threads: int = 1,
    compression_options: dict = {},
    chunk_size_gb: float = 0.5,
    in_memory: bool = True,
) -> list[dict]:
  print("Processing 7z files.")
  to_process = [f for f in to_process if f.endswith('.7z')]
  if not to_process:
    print("No 7z files to process.")
    return []

  chunks: list[utils.SevenZipChunk] = []
  raw_names = []  # per chunk
  file_sizes = []
  for f in to_process:
    raw_path = os.path.join(raw_dir, f)
    new_chunks = utils.traverse_7z_fast(raw_path, chunk_size_gb=chunk_size_gb)
    chunks.extend(new_chunks)
    raw_names.extend([f] * len(new_chunks))
    file_sizes.append(os.path.getsize(raw_path))

  # print stats on 7z files?
  chunk_sizes = [len(c.files) for c in chunks]
  mean_chunk_size = sum(chunk_sizes) / len(chunks)
  total_size_gb = sum(file_sizes) / 1024**3
  print(f"Found {len(file_sizes)} 7z files totalling {total_size_gb:.2f} GB.")
  print(f"Split into {len(chunks)} chunks, mean size {mean_chunk_size:.1f}")

  # Would be nice to tqdm on files instead of chunks.
  iter_chunks = tqdm.tqdm(chunks, unit='chunk')
  chunks_and_raw_names = zip(iter_chunks, raw_names)

  results = []
  if num_threads == 1:
    pool = None
  else:
    pool = concurrent.futures.ProcessPoolExecutor(num_threads)

  for chunk, raw_name in chunks_and_raw_names:
    with chunk.extract(in_memory) as files:
      try:
        chunk_results = parse_chunk(
            files, output_dir,
            tmpdir=utils.get_tmp_dir(in_memory=in_memory),
            compression_options=compression_options,
            pool=pool)
      except BaseException as e:
        # print(e)
        if pool is not None:
          pool.shutdown()  # shutdown before cleaning up tmpdir
        raise e

    for result in chunk_results:
      result['raw'] = raw_name
    results.extend(chunk_results)

    # TODO: give updates on valid files
    # valid = [r['valid'] for r in chunk_results]
    # num_valid = sum(valid)
    # print(f"Chunk {raw_name} valid: {num_valid}/{len(valid)}")

  if pool is not None:
    pool.shutdown()

  return results

md5_key = 'slp_md5'

def get_key(row: dict):
  if md5_key in row:
    return row[md5_key]

  return (row['raw'], row['name'])

def save_raw_db(raw_by_name: dict, raw_db_path: str):
    with open(raw_db_path, 'w') as f:
        json.dump(list(raw_by_name.values()), f, indent=2)

def save_slp_meta(results: list, slp_db_path: str):
    # Load existing metadata
    if os.path.exists(slp_db_path):
        with open(slp_db_path, 'rb') as f:
            slp_meta = pickle.load(f)
    else:
        slp_meta = []

    # Update with new results
    by_key = {get_key(row): row for row in slp_meta}
    for result in results:
        by_key[get_key(result)] = result

    # Save updated metadata
    with open(slp_db_path, 'wb') as f:
        pickle.dump(list(by_key.values()), f)

def count_replays_in_archive(f: str, raw_dir: str, chunk_size_gb: float, raw_by_name: dict, wipe: bool = False) -> int:
    """Count number of replay files in an archive without processing them."""
    # Skip already processed archives
    if not wipe and raw_by_name[f].get('processed', False):
        return 0
        
    raw_path = os.path.join(raw_dir, f)
    if f.endswith('.7z'):
        chunks = utils.traverse_7z_fast(raw_path, chunk_size_gb=chunk_size_gb)
        return sum(len(chunk.files) for chunk in chunks)
    elif f.endswith('.zip'):
        return len(utils.traverse_slp_files_zip(raw_path))
    return 0

def get_archive_files(archive_name: str, path: str, chunk_size_gb: float) -> list[utils.LocalFile]:
    """Get all files from a single archive."""
    if archive_name.endswith('.7z'):
        chunks = utils.traverse_7z_fast(path, chunk_size_gb=chunk_size_gb)
        files = []
        for chunk in chunks:
            # Create SevenZipFile objects instead of using raw strings
            files.extend(utils.SevenZipFile(path, f) for f in chunk.files)
        return files
    elif archive_name.endswith('.zip'):
        return utils.traverse_slp_files_zip(path)
    return []

def get_all_files(root_dir: str, raw_by_name: dict, wipe: bool, chunk_size_gb: float) -> tuple[list[tuple[str, utils.LocalFile]], dict]:
    """Get all files that need processing across all archives.
    Returns:
        - list of (archive_name, file) tuples
        - dict mapping archive_name to total expected files
    """
    files = []
    archive_totals = {}  # Track total files per archive
    
    for archive_name, meta in raw_by_name.items():
        if wipe or not meta.get('processed', False):
            path = os.path.join(root_dir, archive_name)
            archive_files = get_archive_files(archive_name, path, chunk_size_gb)
            
            if archive_files:
                files.extend((archive_name, f) for f in archive_files)
                archive_totals[archive_name] = len(archive_files)
                
    return files, archive_totals

def process_chunk(files: list[tuple[str, utils.LocalFile]], output_dir: str, 
                 tmpdir: str, compression_options: dict) -> list[tuple[str, dict]]:
    """Process a chunk of files.
    Returns list of (archive_name, result) tuples."""
    results = []
    
    # Group files by archive
    by_archive = {}
    for archive_name, file in files:
        if archive_name not in by_archive:
            by_archive[archive_name] = []
        by_archive[archive_name].append(file)

    # Create one temp dir for the whole chunk
    with tempfile.TemporaryDirectory(dir=tmpdir) as extract_dir:
        # Process each archive's files
        for archive_name, archive_files in by_archive.items():
            if isinstance(archive_files[0], utils.SevenZipFile):
                # Extract all files from this archive at once
                file_paths = [f.path for f in archive_files]
                utils.SevenZipFile.batch_extract(
                    archive_files[0].root, file_paths, extract_dir, 
                    batch_size=1000)  # Larger batch size for fewer 7z calls
                
                # Process extracted files
                for file in archive_files:
                    try:
                        extracted_path = os.path.join(extract_dir, file.path)
                        simple_file = utils.SimplePath(extract_dir, file.path)
                        result = parse_slp(simple_file, output_dir, extract_dir, **compression_options)
                        result['raw'] = archive_name
                        results.append((archive_name, result))
                    except Exception as e:
                        print(f"Failed to process {archive_name}/{file.name}: {e}")
                        result = {'name': file.name, 'raw': archive_name, 'valid': False, 'reason': str(e)}
                        results.append((archive_name, result))
            else:
                # Process zip files
                for file in archive_files:
                    try:
                        result = parse_slp(file, output_dir, extract_dir, **compression_options)
                        result['raw'] = archive_name
                        results.append((archive_name, result))
                    except Exception as e:
                        print(f"Failed to process {archive_name}/{file.name}: {e}")
                        result = {'name': file.name, 'raw': archive_name, 'valid': False, 'reason': str(e)}
                        results.append((archive_name, result))
    
    return results

def run_parsing(
    root: str,
    num_threads: int = 1,
    compression_options: dict = {},
    chunk_size: int = 1000,  # Increased default chunk size
    in_memory: bool = True,
    wipe: bool = False,
    dry_run: bool = False,
):
    # Cache tmp dir once
    tmpdir = utils.get_tmp_dir(in_memory=in_memory)

    raw_dir = os.path.join(root, 'Raw')

    # Load existing raw.json
    raw_db_path = os.path.join(root, 'raw.json')
    if os.path.exists(raw_db_path):
        with open(raw_db_path) as f:
            raw_db = json.load(f)
    else:
        raw_db = []

    raw_by_name = {row['name']: row for row in raw_db}

    # Scan Raw directory for archives
    to_process = []
    for dirpath, _, filenames in os.walk(raw_dir):
        for f in filenames:
            if f.endswith('.7z') or f.endswith('.zip'):
                rel_path = os.path.relpath(os.path.join(dirpath, f), raw_dir)
                if rel_path not in raw_by_name:
                    # Add new archive to raw_by_name
                    raw_by_name[rel_path] = {'name': rel_path, 'processed': False}
                    to_process.append(rel_path)
                elif wipe or not raw_by_name[rel_path].get('processed', False):
                    # Add unprocessed or wiped archives
                    to_process.append(rel_path)

    if not to_process:
        print("No new archives to process")
        return

    print(f"To process: {to_process}")

    # Get all files that need processing
    all_files, archive_totals = get_all_files(raw_dir, raw_by_name, wipe, 
                                            compression_options.get('chunk_size_gb', 0.5))
    if not all_files:
        print("No files to process")
        return
    
    print(f"Found {len(all_files)} files across {len(archive_totals)} archives to process")
    
    # Track processed files per archive
    archive_processed = {name: 0 for name in archive_totals}
    
    # Split into roughly equal chunks
    chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]
    print(f"Split into {len(chunks)} chunks of ~{chunk_size} files each")

    # Use more threads for processing
    max_concurrent = num_threads  # Use all available threads
    print(f"Using {max_concurrent} concurrent workers")

    # Batch metadata updates
    metadata_batch_size = 5000
    pending_results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_concurrent) as pool:
        futures = [
            pool.submit(process_chunk, chunk, os.path.join(root, 'Parsed'), tmpdir, compression_options)
            for chunk in chunks
        ]

        pbar = tqdm.tqdm(total=len(all_files), desc="Processing files", unit="file")
        
        for future in concurrent.futures.as_completed(futures):
            try:
                results = future.result()
                pending_results.extend(results)
                
                # Update archive processed counts
                for archive_name, result in results:
                    archive_processed[archive_name] += 1
                
                # Batch update metadata
                if len(pending_results) >= metadata_batch_size:
                    # Group by archive
                    by_archive = {}
                    for archive_name, result in pending_results:
                        if archive_name not in by_archive:
                            by_archive[archive_name] = []
                        by_archive[archive_name].append(result)
                    
                    # Update metadata for completed archives
                    for archive_name, archive_results in by_archive.items():
                        save_slp_meta(archive_results, os.path.join(root, 'parsed.pkl'))
                        if archive_processed[archive_name] >= archive_totals[archive_name]:
                            raw_by_name[archive_name].update(processed=True)
                            save_raw_db(raw_by_name, raw_db_path)
                            pbar.write(f"Completed archive {archive_name}")
                    
                    pending_results = []
                
                # Update progress
                num_valid = sum(r[1]['valid'] for r in results)
                pbar.update(len(results))
                completed_archives = sum(
                    1 for name, count in archive_processed.items() 
                    if count >= archive_totals[name]
                )
                pbar.set_postfix({
                    'valid': f"{num_valid}/{len(results)}", 
                    'archives': f"{completed_archives}/{len(archive_totals)}"
                }, refresh=True)
                
            except Exception as e:
                pbar.write(f"Chunk failed: {e}")
                traceback.print_exc()
        
        # Process any remaining results
        if pending_results:
            by_archive = {}
            for archive_name, result in pending_results:
                if archive_name not in by_archive:
                    by_archive[archive_name] = []
                by_archive[archive_name].append(result)
            
            for archive_name, archive_results in by_archive.items():
                save_slp_meta(archive_results, os.path.join(root, 'parsed.pkl'))
                if archive_processed[archive_name] >= archive_totals[archive_name]:
                    raw_by_name[archive_name].update(processed=True)
                    save_raw_db(raw_by_name, raw_db_path)
        
        pbar.close()

def main(_):
  run_parsing(
      ROOT.value,
      num_threads=THREADS.value,
      chunk_size=CHUNK_SIZE.value,
      in_memory=IN_MEMORY.value,
      compression_options=dict(
          compression=COMPRESSION.value,
          compression_level=COMPRESSION_LEVEL.value,
      ),
      wipe=WIPE.value,
      dry_run=DRY_RUN.value,
  )

if __name__ == '__main__':
  ROOT = flags.DEFINE_string('root', None, 'root directory', required=True)
  # MAX_FILES = flags.DEFINE_integer('max_files', None, 'max files to process')
  THREADS = flags.DEFINE_integer('threads', 1, 'number of threads')
  CHUNK_SIZE = flags.DEFINE_integer('chunk_size', 1000, 'max chunk size in files')
  IN_MEMORY = flags.DEFINE_bool('in_memory', True, 'extract in memory')
  # LOG_INTERVAL = flags.DEFINE_integer('log_interval', 20, 'log interval')
  COMPRESSION = flags.DEFINE_enum_class(
      name='compression',
      default=parsing_utils.CompressionType.ZLIB,  # best one
      enum_class=parsing_utils.CompressionType,
      help='Type of compression to use.')
  COMPRESSION_LEVEL = flags.DEFINE_integer('compression_level', None, 'Compression level.')
  WIPE = flags.DEFINE_bool('wipe', False, 'Wipe existing metadata')
  DRY_RUN = flags.DEFINE_bool('dry_run', False, 'dry run')

  app.run(main)
