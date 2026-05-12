import functools
from types import GenericAlias
from typing import Generic, Mapping, NamedTuple, TypeVar, Union
import typing as tp
import numpy as np

import pyarrow as pa
from melee.enums import Button

T = TypeVar('T')
S = TypeVar('S', bound=tuple[int, ...])
Nest = Union[Mapping[str, 'Nest'], T]
Rank1 = tuple[int]
Rank2 = tuple[int, int]

BoolDType = np.dtype[np.bool_]
FloatDType = np.dtype[np.float32]
Int32DType = np.dtype[np.int32]
BoolArray: tp.TypeAlias = np.ndarray[S, BoolDType]
FloatArray: tp.TypeAlias = np.ndarray[S, FloatDType]
Int32Array: tp.TypeAlias = np.ndarray[S, Int32DType]
UInt8Array: tp.TypeAlias = np.ndarray[S, np.dtype[np.uint8]]
UInt16Array: tp.TypeAlias = np.ndarray[S, np.dtype[np.uint16]]

# we define NamedTuples for python typechecking and IDE integration

class Buttons(NamedTuple, Generic[S]):
  A: BoolArray[S]
  B: BoolArray[S]
  X: BoolArray[S]
  Y: BoolArray[S]
  Z: BoolArray[S]
  L: BoolArray[S]
  R: BoolArray[S]
  D_UP: BoolArray[S]

LIBMELEE_BUTTONS = {name: Button(name) for name in Buttons._fields}

class Stick(NamedTuple, Generic[S]):
  x: FloatArray[S]
  y: FloatArray[S]

class Controller(NamedTuple, Generic[S]):
  main_stick: Stick[S]
  c_stick: Stick[S]
  shoulder: FloatArray[S]
  buttons: Buttons[S]

class Nana(NamedTuple, Generic[S]):
  exists: BoolArray[S]
  percent: UInt16Array[S]
  facing: BoolArray[S]
  x: FloatArray[S]
  y: FloatArray[S]
  action: UInt16Array[S]
  invulnerable: BoolArray[S]
  character: UInt8Array[S]
  jumps_left: UInt8Array[S]
  shield_strength: FloatArray[S]
  on_ground: BoolArray[S]


class Player(NamedTuple, Generic[S]):
  percent: UInt16Array[S]
  facing: BoolArray[S]
  x: FloatArray[S]
  y: FloatArray[S]
  action: UInt16Array[S]
  invulnerable: BoolArray[S]
  character: UInt8Array[S]
  jumps_left: UInt8Array[S]
  shield_strength: FloatArray[S]
  on_ground: BoolArray[S]
  is_dead: BoolArray[S]
  stocks_left: UInt8Array[S]
  controller: Controller[S]
  nana: Nana[S]


class Randall(NamedTuple, Generic[S]):
  x: FloatArray[S]
  y: FloatArray[S]


class FoDPlatforms(NamedTuple, Generic[S]):
  left: FloatArray[S]
  right: FloatArray[S]


MAX_ITEMS = 15


class Item(NamedTuple, Generic[S]):
  exists: BoolArray[S]
  type: UInt16Array[S]
  state: UInt8Array[S]
  x: FloatArray[S]
  y: FloatArray[S]


Items = NamedTuple('Items', [
    (f'item_{i}', Item) for i in range(MAX_ITEMS)
])


class Game(NamedTuple, Generic[S]):
  p0: Player[S]
  p1: Player[S]
  p2: Player[S]
  p3: Player[S]
  stage: UInt8Array[S]
  randall_phase: FloatArray[S]
  randall: Randall[S]
  items: Items
  is_teams: BoolArray[S]
  fod_platforms: FoDPlatforms = ()

# maps pyarrow types back to NamedTuples
PA_TO_NT = {}

Leaf = type[np.generic]
Node = list[tuple[str, type]]


@functools.cache
def get_node_or_leaf(t: type | GenericAlias | tp._GenericAlias) -> Node | Leaf:
  if isinstance(t, GenericAlias):
    assert t.__origin__ is np.ndarray
    generic_dtype = t.__args__[1]
    if (
        not isinstance(generic_dtype, GenericAlias)
        or generic_dtype.__origin__ is not np.dtype
    ):
      raise ValueError(
          f'Expected numpy dtype GenericAlias, got {generic_dtype}')
    dtype = generic_dtype.__args__[0]
    assert issubclass(dtype, np.generic)
    return dtype

  if isinstance(t, tp._GenericAlias):
    t = t.__origin__

  if isinstance(t, type) and issubclass(t, tuple):
    return [(name, t.__annotations__[name]) for name in t._fields]

  return t


def reify_tuple_type(t: type[T] | GenericAlias | tp._GenericAlias) -> T:
  node_or_leaf = get_node_or_leaf(t)
  if isinstance(node_or_leaf, list):
    tuple_type = t.__origin__ if hasattr(t, '__origin__') else t
    return tuple_type(*(
        reify_tuple_type(field_type) for _, field_type in node_or_leaf))
  return node_or_leaf


def _zeros_for_type(t: type, length: int):
  """Return a numpy nest matching the given type filled with zeros/False."""
  node_or_leaf = get_node_or_leaf(t)
  if isinstance(node_or_leaf, list):
    tuple_type = t.__origin__ if hasattr(t, '__origin__') else t
    values = {
        name: _zeros_for_type(field_type, length)
        for name, field_type in node_or_leaf
    }
    return tuple_type(**values)
  else:
    return np.zeros(length, dtype=node_or_leaf)

@functools.lru_cache
def nt_to_pa(nt: type | GenericAlias) -> pa.StructType:
  """Convert and register a NamedTuple (or numpy) type."""

  node_or_leaf = get_node_or_leaf(nt)
  if isinstance(node_or_leaf, list):
    struct_type = pa.struct([
        (name, nt_to_pa(field_type))
        for name, field_type in node_or_leaf
    ])
    PA_TO_NT[struct_type] = nt
    return struct_type
  return pa.from_numpy_dtype(node_or_leaf)

BUTTONS_TYPE = nt_to_pa(Buttons)
STICK_TYPE = nt_to_pa(Stick)
CONTROLLER_TYPE = nt_to_pa(Controller)
NANA_TYPE = nt_to_pa(Nana)
PLAYER_TYPE = nt_to_pa(Player)
RANDALL_TYPE = nt_to_pa(Randall)
FOD_PLATFORMS_TYPE = nt_to_pa(FoDPlatforms)
ITEM_TYPE = nt_to_pa(Item)
ITEMS_TYPE = nt_to_pa(Items)
GAME_TYPE = nt_to_pa(Game)

def array_from_nest(val: Nest[np.ndarray]) -> pa.StructArray:
  if isinstance(val, Mapping):
    values = [array_from_nest(v) for v in val.values()]
    return pa.StructArray.from_arrays(values, names=val.keys())
  else:
    return val

def array_from_nt(val: Union[tuple, np.ndarray]) -> pa.StructArray:
  if isinstance(val, tuple):
    values = [array_from_nt(v) for v in val]
    return pa.StructArray.from_arrays(values, names=val._fields)
  else:
    return val

def nt_to_nest(val: Union[tuple, T]) -> Nest[T]:
  """ Converts a NamedTuple to a Nest."""
  if isinstance(val, tuple) and hasattr(val, '_fields'):
    return {k: nt_to_nest(v) for k, v in zip(val._fields, val)}
  return val

def array_to_nest(val: pa.Array) -> Nest[np.ndarray]:
  if isinstance(val.type, pa.StructType):
    result = {}
    for field in val.type:
      result[field.name] = array_to_nest(val.field(field.name))
    return result
  else:
    assert val.type.num_fields == 0
    return val.to_numpy(zero_copy_only=False)

def array_to_nt(nt: type, val: pa.Array) -> Union[tuple, np.ndarray]:
  node_or_leaf = get_node_or_leaf(nt)
  if isinstance(node_or_leaf, list):
    tuple_type = nt.__origin__ if hasattr(nt, '__origin__') else nt
    assert isinstance(val.type, pa.StructType)
    result = {}
    field_names = {field.name for field in val.type}
    for name, field_type in node_or_leaf:
      if name in field_names:
        result[name] = array_to_nt(field_type, val.field(name))
      else:
        result[name] = _zeros_for_type(field_type, len(val))
    return tuple_type(**result)

  assert val.type.num_fields == 0
  return val.to_numpy(zero_copy_only=False).astype(node_or_leaf)

def game_array_to_nt(game: pa.StructArray) -> Game:
  result = array_to_nt(Game, game)
  assert isinstance(result, Game)
  return result

class InvalidGameError(Exception):
  """Base class for invalid game exceptions."""


Action = TypeVar('Action')
NAME_DTYPE = np.int32


class StateAction(NamedTuple, Generic[S, Action]):
  state: Game[S]
  action: Action
  name: np.ndarray[S, np.dtype[NAME_DTYPE]]


class Frames(NamedTuple, Generic[S, Action]):
  state_action: StateAction[S, Action]
  is_resetting: BoolArray[S]
  reward: FloatArray[S]
