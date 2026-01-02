import dataclasses
import math
import typing as tp

import numpy as np
from multiprocessing import shared_memory


@dataclasses.dataclass(frozen=True)
class ShmArraySpec:
  key: str
  dtype: np.dtype
  shape: tuple[int, ...]

  @property
  def nbytes(self) -> int:
    return int(np.dtype(self.dtype).itemsize) * int(math.prod(self.shape))


def _align_up(x: int, alignment: int) -> int:
  return (x + alignment - 1) // alignment * alignment


class ShmNumpyBuffer:
  """A single SharedMemory segment partitioned into multiple numpy arrays.

  This avoids creating a SharedMemory object per array (fd explosion).
  """

  def __init__(
      self,
      shm: shared_memory.SharedMemory,
      specs: tp.Sequence[ShmArraySpec],
      offsets: tp.Mapping[str, int],
      *,
      alignment: int = 64,
  ):
    self._shm = shm
    self._specs = list(specs)
    self._offsets = dict(offsets)
    self._alignment = int(alignment)
    self._arrays: dict[str, np.ndarray] = {}

  @property
  def name(self) -> str:
    return self._shm.name

  @property
  def size(self) -> int:
    return self._shm.size

  def array(self, key: str) -> np.ndarray:
    arr = self._arrays.get(key)
    if arr is not None:
      return arr
    spec = next(s for s in self._specs if s.key == key)
    offset = self._offsets[key]
    buf = self._shm.buf[offset:offset + spec.nbytes]
    arr = np.ndarray(shape=spec.shape, dtype=spec.dtype, buffer=buf)
    self._arrays[key] = arr
    return arr

  def close(self):
    self._arrays.clear()
    self._shm.close()

  def unlink(self):
    self._shm.unlink()

  @classmethod
  def _compute_offsets(
      cls,
      specs: tp.Sequence[ShmArraySpec],
      *,
      alignment: int,
  ) -> tuple[dict[str, int], int]:
    offsets: dict[str, int] = {}
    cursor = 0
    for spec in specs:
      cursor = _align_up(cursor, alignment)
      offsets[spec.key] = cursor
      cursor += spec.nbytes
    total = _align_up(cursor, alignment)
    return offsets, total

  @classmethod
  def create(
      cls,
      specs: tp.Sequence[ShmArraySpec],
      *,
      name: tp.Optional[str] = None,
      alignment: int = 64,
  ) -> "ShmNumpyBuffer":
    offsets, total = cls._compute_offsets(specs, alignment=alignment)
    shm = shared_memory.SharedMemory(create=True, size=total, name=name)
    return cls(shm, specs, offsets, alignment=alignment)

  @classmethod
  def attach(
      cls,
      name: str,
      specs: tp.Sequence[ShmArraySpec],
      *,
      alignment: int = 64,
  ) -> "ShmNumpyBuffer":
    offsets, total = cls._compute_offsets(specs, alignment=alignment)
    shm = shared_memory.SharedMemory(create=False, name=name, size=total)
    return cls(shm, specs, offsets, alignment=alignment)

