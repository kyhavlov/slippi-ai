#!/usr/bin/env python3
"""Flatten hashed parquet layout under data/Parsed.

Moves files from two-character hash-prefix directories (e.g. data/Parsed/ab/abcdef...)
into the top-level data/Parsed directory. After moving, removes empty prefix
folders. Existing files with the same name are left untouched.
"""
import argparse
from pathlib import Path
import string

HEX = set(string.hexdigits.lower())


def is_prefix_dir(path: Path) -> bool:
  return (
      path.is_dir()
      and len(path.name) == 2
      and all(ch in HEX for ch in path.name.lower())
  )


def flatten(root: Path, dry_run: bool) -> tuple[int, int]:
  moved = 0
  skipped = 0

  for prefix_dir in sorted(root.iterdir()):
    if not is_prefix_dir(prefix_dir):
      continue

    for child in sorted(prefix_dir.iterdir()):
      if not child.is_file():
        continue
      dest = root / child.name
      if dest.exists():
        skipped += 1
        continue
      if dry_run:
        print(f"DRY-RUN: would move {child} -> {dest}")
      else:
        child.replace(dest)
      moved += 1

    if not dry_run:
      try:
        prefix_dir.rmdir()
      except OSError:
        # Directory not empty (e.g. skipped files) — leave it.
        pass

  return moved, skipped


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--root',
      default='data/Parsed',
      type=Path,
      help='Root directory containing hashed parquet files.')
  parser.add_argument(
      '--dry-run',
      action='store_true',
      help='Print planned moves without modifying the filesystem.')
  args = parser.parse_args()

  root = args.root
  if not root.exists():
    parser.error(f"Root directory {root} does not exist.")

  moved, skipped = flatten(root, args.dry_run)
  print(f"Moved {moved} files. Skipped {skipped} existing files.")


if __name__ == '__main__':
  main()
