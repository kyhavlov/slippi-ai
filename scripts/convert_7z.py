"""Convert 7z files (recursively) to zip for faster processing."""

import itertools
import os
import shutil
import subprocess
import tempfile
import typing as tp
import zipfile

import py7zr
import tqdm

from absl import app, flags

from slippi_db import utils

T = tp.TypeVar('T')

def chunked(iterable: tp.Iterable[T], n: int) -> tp.Iterator[tp.Iterator[T]]:
  """Maximally lazy chunking."""
  it = iter(iterable)
  while True:
    chunk = itertools.islice(it, n)

    try:
      first = next(chunk)
    except StopIteration:
      break

    yield itertools.chain([first], chunk)

class ExtractFileError(Exception):

  def __init__(self, failed_file: str, log: str):
    self.failed_file = failed_file
    self.log = log
    super().__init__(f'Failed to extract {failed_file}')


def extract_file_list(
    path: str,
    files: tp.Sequence[str],
    output_dir: str,
) -> None:
  """Extract files from a 7z archive."""
  path = os.path.abspath(path)

  with tempfile.NamedTemporaryFile() as input_list:
    for file in files:
      input_list.write(f'{file}\n'.encode('utf-8'))
    input_list.seek(0)

    # TODO: is there a way to do this multithreaded?
    result = subprocess.run(
        ['7z', 'x', path, f'@{input_list.name}'],
        cwd=output_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

  if result.returncode == 0:
    return

  failed_file = _parse_failed_path(result.stdout, files)
  if failed_file is None:
    print(result.stdout)
    raise subprocess.CalledProcessError(result.returncode, result.args)

  raise ExtractFileError(failed_file, result.stdout)


def _parse_failed_path(log: str, requested_files: tp.Sequence[str]) -> tp.Optional[str]:
  lines = log.splitlines()

  # try to find a line immediately following an ERROR marker
  for idx, line in enumerate(lines):
    if 'ERROR' not in line.upper():
      continue
    candidate_from_line = _extract_path_from_error_line(line)
    if candidate_from_line and _looks_like_requested_file(candidate_from_line, requested_files):
      return candidate_from_line
    for follow in lines[idx + 1: idx + 4]:
      candidate = follow.strip()
      if _looks_like_requested_file(candidate, requested_files):
        return candidate

  # fallback: scan for any requested filename mentioned in the log
  for candidate in reversed(lines):
    candidate = candidate.strip()
    if _looks_like_requested_file(candidate, requested_files):
      return candidate

  return None


def _looks_like_requested_file(candidate: str, requested_files: tp.Sequence[str]) -> bool:
  if not candidate:
    return False
  norm = _normalize_path(candidate)
  for name in requested_files:
    if norm.endswith(_normalize_path(name)):
      return True
  return False


def _extract_path_from_error_line(line: str) -> tp.Optional[str]:
  if not line:
    return None
  if ':' not in line:
    return line.strip()
  tail = line.rsplit(':', 1)[-1].strip()
  return tail or None


def _normalize_path(path: str) -> str:
  return path.replace('\\', '/').lstrip('./')


def _dir_has_files(path: str) -> bool:
  for _dirpath, _dirnames, filenames in os.walk(path):
    if filenames:
      return True
  return False


def _list_extracted_files(root: str) -> dict[str, str]:
  files: dict[str, str] = {}
  for dirpath, _dirnames, filenames in os.walk(root):
    for name in filenames:
      rel_path = os.path.relpath(os.path.join(dirpath, name), root)
      norm = _normalize_path(rel_path)
      files[norm] = rel_path
  return files


def _remove_extracted_file(root: str, rel_path: str) -> None:
  abs_path = os.path.join(root, rel_path)
  if os.path.exists(abs_path):
    os.remove(abs_path)
    _cleanup_empty_dirs(os.path.dirname(abs_path), root)


def _cleanup_empty_dirs(path: str, stop_dir: str) -> None:
  stop_dir = os.path.abspath(stop_dir)
  while path and os.path.abspath(path).startswith(stop_dir):
    try:
      os.rmdir(path)
    except OSError:
      break
    if os.path.abspath(path) == stop_dir:
      break
    path = os.path.dirname(path)


def _create_chunk_zip(
    zip_path: str,
    source_dir: str,
    skipped_files: tp.Set[str],
) -> bool:
  cmd = ['7z', '-tzip', 'a', zip_path, '*']
  while True:
    files_map = _list_extracted_files(source_dir)
    if not files_map:
      print('No files left to zip in', source_dir)
      if os.path.exists(zip_path):
        os.remove(zip_path)
      return False

    if os.path.exists(zip_path):
      os.remove(zip_path)

    result = subprocess.run(
        cmd,
        cwd=source_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if result.returncode == 0:
      return True

    failed = _parse_failed_path(result.stdout, list(files_map.keys()))
    if failed is None:
      print(result.stdout)
      raise subprocess.CalledProcessError(result.returncode, cmd)

    norm_failed = _normalize_path(failed)
    rel_path = files_map.get(norm_failed)
    skipped_files.add(norm_failed)

    if rel_path is None:
      print('7z reported failure on', failed,
            'but file was not found in chunk; retrying without changes')
      continue

    print('Skipping file during zip due to error:', rel_path)
    _remove_extracted_file(source_dir, rel_path)


def convert(
    input_path: str,
    output_path: str,
    max_chunk_size_gb: float = 16,  # uncompressed
    in_memory: bool = False,
) -> None:
  cwd = os.getcwd()
  input_path = os.path.abspath(input_path)
  output_path = os.path.abspath(output_path)
  archive = py7zr.SevenZipFile(input_path, 'r')
  skipped_files: set[str] = set()

  # calculate optimal chunks
  folders = archive.header.main_streams.unpackinfo.folders

  max_chunk_size = max_chunk_size_gb * 1024**3
  chunks: list[list[str]] = []
  chunk_size = 0
  chunk: list[str] = []

  for folder in folders:
    for file in folder.files:
      if chunk_size + file.uncompressed > max_chunk_size:
        chunks.append(chunk)
        chunk = []
        chunk_size = 0

      chunk_size += file.uncompressed
      chunk.append(file.filename)

  # commit last chunk
  if chunk:
    chunks.append(chunk)

  print('Chunks:', len(chunks))

  # relpaths = [p for p in archive.getnames() if p.endswith('.slp')]
  # relpaths = reversed(relpaths)
  # chunks = chunked(tqdm.tqdm(relpaths), chunk_size)

  archive_dir = os.path.dirname(input_path)
  with tempfile.TemporaryDirectory(dir=archive_dir) as zipdir:
    zip_paths = []
    if in_memory:
      tmp_root = utils.get_tmp_dir(in_memory=True)
    else:
      tmp_root = archive_dir
    if tmp_root is not None:
      os.makedirs(tmp_root, exist_ok=True)

    for i, chunk in enumerate(tqdm.tqdm(chunks, smoothing=0)):

      # with tempfile.TemporaryDirectory() as tmpdir:
      with tempfile.TemporaryDirectory(dir=tmp_root) as tmpdir:
        remaining = list(chunk)

        while remaining:
          try:
            extract_file_list(input_path, remaining, tmpdir)
            break
          except ExtractFileError as err:
            failed_file = err.failed_file
            norm_failed = _normalize_path(failed_file)
            match = next((r for r in remaining if _normalize_path(r) == norm_failed), None)
            if match is None:
              raise
            skipped_files.add(norm_failed)
            remaining.remove(match)
            print('Skipping file due to extraction error:', match)
            continue
        else:
          # nothing left to extract
          pass

        if not _dir_has_files(tmpdir):
          print('No files extracted for chunk', i, '- skipping zip creation')
          continue

        zip_path = os.path.join(zipdir, f'{i}.zip')
        if _create_chunk_zip(zip_path, tmpdir, skipped_files):
          zip_paths.append(zip_path)

    # combine all zip files, preferring zipmerge when available
    if not zip_paths:
      print('No zip chunks created for', input_path, '- producing empty archive')
      with zipfile.ZipFile(output_path, 'w'):
        pass
    else:
      zipmerge_path = shutil.which('zipmerge')
      if zipmerge_path is not None:
        try:
          subprocess.check_call([zipmerge_path, output_path, *zip_paths])
          return
        except subprocess.CalledProcessError as exc:
          print('zipmerge failed with', exc, '- falling back to Python merge')

      _merge_zip_archives(zip_paths, output_path)


def _merge_zip_archives(zip_paths: list[str], output_path: str) -> None:
  # combine all zip files using Python fallback
  with zipfile.ZipFile(output_path, 'w') as final_zip:
    for zip_path in zip_paths:
      with zipfile.ZipFile(zip_path, 'r') as chunk_zip:
        for member in chunk_zip.infolist():
          if member.is_dir():
            final_zip.writestr(member, b'')
            continue

          member_copy = zipfile.ZipInfo(member.filename)
          member_copy.date_time = member.date_time
          member_copy.compress_type = member.compress_type
          member_copy.comment = member.comment
          member_copy.extra = member.extra
          member_copy.internal_attr = member.internal_attr
          member_copy.external_attr = member.external_attr
          with chunk_zip.open(member, 'r') as src, final_zip.open(member_copy, 'w') as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)

  with zipfile.ZipFile(output_path) as zip_archive:
    for sf in archive.files:
      if sf.is_directory or _normalize_path(sf.filename) in skipped_files:
        continue
      info = zip_archive.getinfo(sf.filename)
      assert info.file_size == sf.uncompressed

  if skipped_files:
    print('Skipped', len(skipped_files), 'files due to extraction errors')

  os.chdir(cwd)  # for line_profiler

INPUT = flags.DEFINE_string('input', None, 'Input path.', required=True)
OUTPUT_DIR = flags.DEFINE_string('output_dir', None, 'Output directory.')
CHUNK_SIZE = flags.DEFINE_float('chunk_size', 1, 'Max chunk size in GB.')


def main(_):
  input_path = os.path.abspath(INPUT.value)
  chunk_size = CHUNK_SIZE.value

  if os.path.isdir(input_path):
    base_output = OUTPUT_DIR.value
    if base_output is not None:
      base_output = os.path.abspath(base_output)
      os.makedirs(base_output, exist_ok=True)

    targets: list[tuple[str, str]] = []
    for root, _dirs, files in os.walk(input_path):
      for name in files:
        if not name.lower().endswith('.7z'):
          continue

        archive_path = os.path.join(root, name)
        if base_output is None:
          target_dir = root
        else:
          rel_dir = os.path.relpath(root, input_path)
          target_dir = os.path.join(base_output, rel_dir)

        os.makedirs(target_dir, exist_ok=True)
        output_name = name.removesuffix('.7z') + '.zip'
        output_path = os.path.join(target_dir, output_name)
        if os.path.exists(output_path):
          print('Skipping', archive_path, '- output already exists at', output_path)
          continue
        targets.append((archive_path, output_path))

    if not targets:
      print('No .7z archives found under', input_path)
      return

    for src, dst in sorted(targets):
      print('Converting', src, 'to', dst)
      convert(src, dst, max_chunk_size_gb=chunk_size)
  else:
    output_dir = OUTPUT_DIR.value or os.path.dirname(INPUT.value)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_name = os.path.basename(INPUT.value).removesuffix('.7z') + '.zip'
    output_path = os.path.join(output_dir, output_name)
    if os.path.exists(output_path):
      print('Skipping', INPUT.value, '- output already exists at', output_path)
      return
    print('Converting', INPUT.value, 'to', output_path)
    convert(INPUT.value, output_path, max_chunk_size_gb=chunk_size)

if __name__ == '__main__':
  app.run(main)
