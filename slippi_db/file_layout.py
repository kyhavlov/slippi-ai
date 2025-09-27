import os
from typing import Optional

PARQUET_PREFIX_LEN = 2


def parquet_relative_path(md5: str, prefix_len: int = PARQUET_PREFIX_LEN) -> str:
  prefix = md5[:prefix_len]
  return os.path.join(prefix, md5)


def parquet_full_path(root: str, md5: str, prefix_len: int = PARQUET_PREFIX_LEN) -> str:
  return os.path.join(root, parquet_relative_path(md5, prefix_len=prefix_len))


def resolve_parquet_path(root: str, md5: str, prefix_len: int = PARQUET_PREFIX_LEN) -> str:
  hashed_path = parquet_full_path(root, md5, prefix_len=prefix_len)
  if os.path.exists(hashed_path):
    return hashed_path
  return os.path.join(root, md5)


def ensure_parquet_directory(root: str, md5: str, prefix_len: int = PARQUET_PREFIX_LEN) -> str:
  path = parquet_full_path(root, md5, prefix_len=prefix_len)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  return path
