"""
Converts SSBM types to Tensorflow types.
"""

import abc
import dataclasses
import enum
import math
from typing import (
    Any, Callable, Dict, Generic, Iterator, Mapping, NamedTuple, Optional, Sequence,
    Tuple, Type, TypeVar, Union
)

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt

from slippi_ai import utils
from slippi_ai.types import (
    Buttons,
    Controller,
    Game,
    Item,
    Items,
    Nana,
    Nest,
    Player,
    Randall,
    Stick,
)
from slippi_ai.controller_lib import LEGAL_BUTTONS
from slippi_ai.data import Action, StateAction

float_type = tf.float32
In = TypeVar('In')
Out = TypeVar('Out')

class Embedding(Generic[In, Out], abc.ABC, snt.Module):
  """Embeds game type (In) into tf-ready type Out."""

  def __init__(self, name: Optional[str] = None):
    super().__init__(name=name)

  def from_state(self, state: In) -> Out:
    """Encodes a parsed state."""
    return self.dtype(state)

  @abc.abstractmethod
  def __call__(self, x: Out) -> tf.Tensor:
    """Embed the input state as a flat tensor."""

  def map(self, f, *args: Out) -> Out:
    return f(self, *args)

  def flatten(self, struct: Out) -> Iterator[Any]:
    yield struct

  def unflatten(self, seq: Iterator[Any]) -> Out:
    return next(seq)

  def decode(self, out: Out) -> In:
    """Inverse of `from_state`."""
    return out

  # def preprocess(self, x: In):
  #   """Used by discretization."""
  #   return x

  def dummy(self, shape: Sequence[int] = ()) -> Out:
    """A dummy value."""
    return np.zeros(shape, self.dtype)

  def dummy_embedding(self, shape: Sequence[int] = ()):
    return np.zeros(shape + [self.size], np.float32)

  def sample(self, embedded: tf.Tensor, **kwargs) -> Out:
    raise NotImplementedError

  def distance(self, embedded: tf.Tensor, target: Out) -> Out:
    """Negative log-prob of the target sample."""
    raise NotImplementedError

  def distribution(self, embedded: tf.Tensor):
    raise NotImplementedError


class BoolEmbedding(Embedding[bool, np.bool_]):
  size = 1
  dtype = np.bool_

  def __init__(self, name='bool', on=1., off=0.):
    super().__init__(name=name)
    self.on = on
    self.off = off

  def __call__(self, t):
    return tf.expand_dims(tf.where(t, self.on, self.off), -1)

  def distance(self, predicted, target):
    logits = tf.squeeze(predicted, [-1])
    labels = tf.cast(target, float_type)

    common_shape = tf.broadcast_static_shape(logits.shape, labels.shape)
    logits = tf.broadcast_to(logits, common_shape)
    labels = tf.broadcast_to(labels, common_shape)

    return tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits, labels=labels)

  def sample(self, t, temperature=None):
    t = tf.squeeze(t, -1)
    if temperature is not None:
      t = t / temperature
    dist = tfp.distributions.Bernoulli(logits=t, dtype=tf.bool)
    return dist.sample()

  def distribution(self, embedded: tf.Tensor):
    logits = tf.squeeze(embedded, -1)
    return tfp.distributions.Bernoulli(logits=logits, dtype=tf.bool)

embed_bool = BoolEmbedding()

class FloatEmbedding(Embedding[float, np.float32]):
  dtype = np.float32
  size = 1

  def __init__(self, name, scale=None, bias=None, lower=-10., upper=10.):
    super().__init__(name=name)
    self.scale = scale
    self.bias = bias
    self.lower = lower
    self.upper = upper

  def encode(self, t):
    if t.dtype is not float_type:
      t = tf.cast(t, float_type)
    if self.bias is not None:
      t += self.bias
    if self.scale is not None:
      t *= self.scale
    if self.lower:
      t = tf.maximum(t, self.lower)
    if self.upper:
      t = tf.minimum(t, self.upper)
    return t

  def __call__(self, t, **_):
    return tf.expand_dims(self.encode(t), -1)

  def extract(self, t):
    if self.scale:
      t /= self.scale
    if self.bias:
      t -= self.bias
    return tf.squeeze(t, [-1])

  def to_input(self, t):
    return t

  def distance(self, predicted, target):
    target = self.encode(target)
    predicted = tf.squeeze(predicted, [-1])
    return tf.square(predicted - target)

  def sample(self, t, **_):
    raise NotImplementedError("Can't sample floats yet.")

embed_float = FloatEmbedding("float")

class OneHotEmbedding(Embedding[int, np.int32]):

  def __init__(self, name, size, dtype=np.int32):
    super().__init__(name=name)
    self.size = size
    self.input_size = size
    self.dtype = dtype
    self.tf_dtype = tf.dtypes.as_dtype(dtype)

  def __call__(self, t: tf.Tensor, residual=False, **_):
    if t.dtype != self.tf_dtype:
      raise ValueError(f"Expected {self.tf_dtype}, got {t.dtype}.")

    one_hot = tf.one_hot(t, self.size)

    if residual:
      logits = math.log(self.size * 10) * one_hot
      return logits
    else:
      return one_hot

  def to_input(self, logits):
    return tf.nn.softmax(logits)

  def extract(self, embedded):
    # TODO: pick a random sample?
    return tf.argmax(embedded, -1, output_type=self.dtype)

  def distance(self, embedded, target):
    logprobs = tf.nn.log_softmax(embedded)
    target = self(target)
    return -tf.reduce_sum(logprobs * target, -1)

  def sample(self, embedded, temperature=None):
    logits = embedded
    if temperature is not None:
      logits = logits / temperature
    dist = tfp.distributions.Categorical(logits=logits, dtype=self.tf_dtype)
    return dist.sample()

  def distribution(self, embedded: tf.Tensor):
    return tfp.distributions.Categorical(logits=embedded, dtype=self.tf_dtype)

class NullEmbedding(Embedding[In, None]):
  size = 0

  def from_state(self, state: In) -> None:
    return None

  def __call__(self, t):
    assert t is None
    return None

  def map(self, f, *args: None) -> None:
    for x in args:
      assert x is None
    return None

  def flatten(self, none: None):
    pass

  def unflatten(self, seq):
    return None

  def decode(self, out: None) -> None:
    assert out is None
    return None

  def dummy(self) -> None:
    """A dummy value."""
    return None

  def sample(self, embedded, **_):
    return None

NT = TypeVar("NT")

class StructEmbedding(Embedding[NT, NT]):
  """Embeds structures: dictionaries or NamedTuples or dataclasses.

  Sub-embeddings are a subset of the keys/fields in the input type.
  The order of sub-embeddings determines the order of traversal, which
  is important for autoregressive sampling.
  """
  def __init__(
      self,
      name: str,
      embedding: Sequence[Tuple[str, Embedding]],
      builder: Callable[[Mapping[str, Any]], NT],
      getter: Callable[[NT, str], Any],
  ):
    super().__init__(name=name)
    self.embedding = embedding
    self.builder = builder
    self.getter = getter

    self.size = 0
    for _, op in embedding:
      self.size += op.size

  def map(self, f, *args: NT) -> NT:
    result = {
        k: e.map(f, *(self.getter(x, k) for x in args))
        for k, e in self.embedding}
    return self.builder(result)

  def flatten(self, struct: NT):
    for k, e in self.embedding:
      yield from e.flatten(self.getter(struct, k))

  def unflatten(self, seq: Iterator[Any]) -> NT:
    return self.builder({k: e.unflatten(seq) for k, e in self.embedding})

  def from_state(self, state: NT) -> NT:
    struct = {k: e.from_state(self.getter(state, k)) for k, e in self.embedding}
    return self.builder(struct)

  def __call__(self, struct: NT, **kwargs) -> tf.Tensor:
    embed = []

    for field, op in self.embedding:
      t = op(self.getter(struct, field), **kwargs)
      embed.append(t)

    return tf.concat(axis=-1, values=embed)

  # def split(self, embedded: tf.Tensor) -> Mapping[str, tf.Tensor]:
  #   fields, ops = zip(*self.embedding)
  #   sizes = [op.size for op in ops]
  #   splits = tf.split(embedded, sizes, -1)
  #   return dict(zip(fields, splits))

  # def distance(self, embedded: tf.Tensor, target: NT) -> NT:
  #   distances = {}
  #   split = self.split(embedded)
  #   for field, op in self.embedding:
  #     distances[field] = op.distance(split[field], self.getter(target, field))
  #   return self.builder(distances)

  # def sample(self, embedded: tf.Tensor, **kwargs):
  #   """Samples sub-components independently."""
  #   samples = {}
  #   split = self.split(embedded)
  #   for field, op in self.embedding:
  #     samples[field] = op.sample(split[field], **kwargs)
  #   return self.builder(samples)

  def dummy(self, shape: Sequence[int] = ()):
    return self.map(lambda e: e.dummy(shape))

  def dummy_embedding(self, shape):
    return self.map(lambda e: e.dummy_embedding(shape))

  def decode(self, struct: NT) -> NT:
    return self.map(lambda e, x: e.decode(x), struct)

T = TypeVar("T")

# use this because lambdas can't be properly pickled :(
class SplatKwargs(Generic[T]):
  """Wraps a function that takes kwargs."""

  def __init__(self, f: Callable[..., T], fixed_kwargs: Mapping[str, Any] = {}):
      self._func = f
      self._fixed_kwargs = fixed_kwargs

  def __call__(self, kwargs: Mapping[str, Any]) -> T:
      return self._func(**kwargs, **self._fixed_kwargs)

def struct_embedding_from_nt(name: str, nt: NT) -> StructEmbedding[NT]:
  return StructEmbedding(
      name=name,
      embedding=list(zip(nt._fields, nt)),
      builder=SplatKwargs(type(nt)),
      getter=getattr,
  )

# annoyingly, type inference doesn't work here
def ordered_struct_embedding(
    name: str,
    embedding: Sequence[Tuple[str, Embedding]],
    nt_type: Type[NT],
) -> StructEmbedding[NT]:
  """Supports missing fields, which will appear as ()."""
  existing_fields = set(k for k, _ in embedding)
  missing_fields = set(nt_type._fields) - existing_fields
  missing_kwargs = {k: () for k in missing_fields}

  return StructEmbedding(
      name=name,
      embedding=embedding,
      builder=SplatKwargs(nt_type, missing_kwargs),
      getter=getattr,
  )

K = TypeVar("K")
V = TypeVar("V")

def get_dict(d: Mapping[K, V], k: K) -> V:
  return d[k]

id_fn = lambda x: x

def dict_embedding(
    name: str,
    embedding: Sequence[Tuple[str, Embedding]],
) -> StructEmbedding[Dict[str, Any]]:
  return StructEmbedding(
      name=name,
      embedding=embedding,
      builder=id_fn,
      getter=get_dict,
  )


class MLPWrapper(Embedding[In, Out]):

  def __init__(
      self,
      output_sizes: Sequence[int],
      embed: Embedding[In, Out],
  ):
    super().__init__(name=f'MLP_{embed.name}')
    self._output_sizes = output_sizes
    self._embed = embed
    self.size = output_sizes[-1]
    self._mlp = snt.nets.MLP(
        output_sizes, activate_final=True,
        activation=tf.nn.relu,
    )

  def from_state(self, state: In) -> Out:
    return self._embed.from_state(state)

  def __call__(self, inputs: Out) -> tf.Tensor:
    embedded = self._embed(inputs)
    return self._mlp(embedded)

  def dummy(self, shape: Sequence[int] = ()):
    return self._embed.dummy(shape)

  def dummy_embedding(self, shape):
    return self._embed.dummy_embedding(shape)

# one larger than KIRBY_STONE_UNFORMING
# embed_action = EnumEmbedding(enums.Action, size=0x18F, dtype=np.int16)
embed_action = OneHotEmbedding('Action', size=0x18F, dtype=np.int32)

# one larger than SANDBAG
# embed_char = EnumEmbedding(enums.Character, size=0x21, dtype=np.uint8)
embed_char = OneHotEmbedding('Character', size=0x21, dtype=np.uint8)

# puff and kirby have 6 jumps in the legacy encoding
legacy_embed_jumps_left = OneHotEmbedding(
    "legacy_jumps_left", 6, dtype=np.uint8)
# Allow an updated encoding with 7 bins when explicitly enabled
embed_jumps_left = OneHotEmbedding("jumps_left", 7, dtype=np.uint8)


def _base_player_embedding(
    xy_scale: float,
    shield_scale: float,
    speed_scale: float,
    with_speeds: bool,
    legacy_jumps_left: bool,
) -> list[tuple[str, Embedding]]:
  embed_xy = FloatEmbedding("xy", scale=xy_scale)

  embedding: list[tuple[str, Embedding]] = [
      ("percent", FloatEmbedding("percent", scale=0.01)),
      ("facing", BoolEmbedding("facing", off=-1.)),
      ("x", embed_xy),
      ("y", embed_xy),
      ("action", embed_action),
      ("character", embed_char),
      ("invulnerable", embed_bool),
      ("jumps_left", legacy_embed_jumps_left if legacy_jumps_left else embed_jumps_left),
      ("shield_strength", FloatEmbedding("shield_size", scale=shield_scale)),
      ("on_ground", embed_bool),
      ("is_dead", embed_bool),
      ("stocks_left", FloatEmbedding("stocks_left", scale=0.25)),
  ]

  if with_speeds:
    embed_speed = FloatEmbedding("speed", scale=speed_scale)
    embedding.extend([
        ('speed_air_x_self', embed_speed),
        ('speed_ground_x_self', embed_speed),
        ('speed_y_self', embed_speed),
        ('speed_x_attack', embed_speed),
        ('speed_y_attack', embed_speed),
    ])

  return embedding


def _make_nana_embedding(
    base_embedding: list[tuple[str, Embedding]],
    shield_scale: float,
) -> StructEmbedding[Nana]:
  nana_embedding = [
      (name, embed)
      for name, embed in base_embedding
      if name not in ('is_dead', 'stocks_left')
  ]
  nana_embedding.append(('exists', embed_bool))
  return ordered_struct_embedding("nana", nana_embedding, Nana)


def make_player_embedding(
    xy_scale: float = 0.05,
    shield_scale: float = 0.01,
    speed_scale: float = 0.5,
    with_speeds: bool = False,
    with_controller: bool = False,
    with_nana: bool = False,
    legacy_jumps_left: bool = True,
) -> StructEmbedding[Player]:
  base_embedding = _base_player_embedding(
      xy_scale=xy_scale,
      shield_scale=shield_scale,
      speed_scale=speed_scale,
      with_speeds=with_speeds,
      legacy_jumps_left=legacy_jumps_left,
  )

  embedding = list(base_embedding)

  if with_controller:
    embed_controller_default = get_controller_embedding()
    embedding.append(('controller', embed_controller_default))

  if with_nana:
    embed_nana = _make_nana_embedding(base_embedding, shield_scale)
    embedding.append(('nana', embed_nana))

  return ordered_struct_embedding("player", embedding, Player)

@dataclasses.dataclass
class PlayerConfig:
  xy_scale: float = 0.05
  shield_scale: float = 0.01
  speed_scale: float = 0.5
  with_speeds: bool = False
  # don't use opponent's controller
  # our own will be embedded separately
  with_controller: bool = False
  with_nana: bool = False
  legacy_jumps_left: bool = True

# future proof in case we want to play on wacky stages
# embed_stage = EnumEmbedding(enums.Stage, size=64, dtype=np.uint8)
embed_stage = OneHotEmbedding('Stage', size=64, dtype=np.uint8)

embed_randall_phase = FloatEmbedding("randall_phase", scale=1/1200.)


class ItemsType(enum.Enum):
  SKIP = 'skip'
  FLAT = 'flat'
  MLP = 'mlp'


@dataclasses.dataclass
class ItemsConfig:
  type: ItemsType = ItemsType.SKIP
  mlp_sizes: tuple[int, ...] = (128, 32)


MAX_ITEM_TYPE = 0xEC
MAX_ITEM_STATE = 11


def make_item_embedding(xy_scale: float) -> StructEmbedding[Item]:
  embed_xy = FloatEmbedding("item_xy", scale=xy_scale)
  return struct_embedding_from_nt("item", Item(
      exists=embed_bool,
      type=OneHotEmbedding('ItemType', size=MAX_ITEM_TYPE + 1, dtype=np.int32),
      state=OneHotEmbedding('ItemState', size=MAX_ITEM_STATE + 1, dtype=np.uint8),
      x=embed_xy,
      y=embed_xy,
  ))


def make_items_embedding(
    items_config: ItemsConfig,
    xy_scale: float,
) -> Embedding[Items, Any]:
  if items_config.type is ItemsType.SKIP:
    return ordered_struct_embedding("items", [], Items)

  embed_item_flat = make_item_embedding(xy_scale)

  if items_config.type is ItemsType.FLAT:
    embed_item = embed_item_flat
  elif items_config.type is ItemsType.MLP:
    embed_item = MLPWrapper(
        output_sizes=items_config.mlp_sizes,
        embed=embed_item_flat,
    )
  else:
    raise ValueError(f"Unsupported items config type: {items_config.type}")

  return ordered_struct_embedding(
      "items",
      [(field, embed_item) for field in Items._fields],
      Items)

_PORTS = (0, 1)
# _PLAYERS = tuple(f'p{p}' for p in _PORTS)
# _SWAP_MAP = dict(zip(_PLAYERS, reversed(_PLAYERS)))

def make_game_embedding(
    player_config: dict | PlayerConfig | None = None,
    num_players: int = 4,
    with_randall_phase: bool = True,
    with_randall_xy: bool = False,
    items_config: ItemsConfig = ItemsConfig(),
):
  if num_players not in (2, 4):
    raise ValueError(f"num_players must be 2 or 4, got {num_players}")

  if player_config is None:
    player_config_dict: dict[str, Any] = {}
  elif dataclasses.is_dataclass(player_config):
    player_config_dict = dataclasses.asdict(player_config)
  else:
    player_config_dict = dict(player_config)

  embed_player = make_player_embedding(**player_config_dict)

  xy_scale = player_config_dict.get('xy_scale', 0.05)

  embedding_fields: list[tuple[str, Embedding]] = [
      ('p0', embed_player),
      ('p1', embed_player),
  ]
  if num_players == 4:
    embedding_fields.extend([
        ('p2', embed_player),
        ('p3', embed_player),
    ])
  embedding_fields.append(('stage', embed_stage))
  if with_randall_phase:
    embedding_fields.append(('randall_phase', embed_randall_phase))
  embedding_fields.append(('is_teams', embed_bool))

  if with_randall_xy:
    embed_xy = FloatEmbedding("randall_xy", scale=xy_scale)
    embedding_fields.append((
        'randall',
        struct_embedding_from_nt(
            "randall", Randall(x=embed_xy, y=embed_xy)),
    ))

  if items_config.type is not ItemsType.SKIP:
    embed_items = make_items_embedding(
        items_config=items_config,
        xy_scale=xy_scale,
    )
    embedding_fields.append(('items', embed_items))

  return ordered_struct_embedding("game", embedding_fields, Game)

# don't use opponent's controller
# our own will be exposed in the input
default_embed_game = make_game_embedding(
    player_config=dict(with_controller=False))

# Embeddings for controllers
embed_buttons = ordered_struct_embedding(
    'buttons',
    [(b.value, BoolEmbedding(name=b.value)) for b in LEGAL_BUTTONS],
    Buttons,
)

class DiscreteEmbedding(OneHotEmbedding):
  """Buckets float inputs in [0, 1]."""

  def __init__(self, n=16):
    super().__init__('DiscreteEmbedding', n+1, dtype=np.uint8)
    self.n = n

  # def sample(self, embedded, **kwargs):
  #   discrete = super().sample(embedded, **kwargs)
  #   return tf.cast(discrete, tf.float32) / self.n

  def from_state(self, a: Union[np.float32, np.ndarray]):
    assert a.dtype == np.float32
    return (a * self.n + 0.5).astype(self.dtype)

  def decode(self, a: Union[np.uint8, np.ndarray]) -> Union[np.float32, np.ndarray]:
    assert a.dtype == self.dtype
    return (a / self.n).astype(np.float32)

NATIVE_AXIS_SPACING = 160
NATIVE_SHOULDER_SPACING = 140

def get_controller_embedding(
    axis_spacing: int = 0,
    shoulder_spacing: int = 4,
) -> StructEmbedding[Controller]:
  """Controller embedding. Used for autoregressive sampling, so order matters."""
  if axis_spacing:
    if NATIVE_AXIS_SPACING % axis_spacing != 0:
      raise ValueError(
        f'Axis spacing must divide {NATIVE_AXIS_SPACING}, got {axis_spacing}.')
    embed_axis = DiscreteEmbedding(axis_spacing)
  else:
    embed_axis = embed_float

  embed_stick = struct_embedding_from_nt(
      "stick", Stick(x=embed_axis, y=embed_axis))

  if NATIVE_SHOULDER_SPACING % shoulder_spacing != 0:
    raise ValueError(
      f'Shoulder spacing must divide {NATIVE_SHOULDER_SPACING}, got {shoulder_spacing}.')
  embed_shoulder = DiscreteEmbedding(shoulder_spacing)

  return ordered_struct_embedding(
      "controller", [
          ("buttons", embed_buttons),
          ("main_stick", embed_stick),
          ("c_stick", embed_stick),
          ("shoulder", embed_shoulder),
      ], Controller)

@dataclasses.dataclass
class ControllerConfig:
  axis_spacing: int = 16
  shoulder_spacing: int = 4

@dataclasses.dataclass
class EmbedConfig:
  num_players: int = 4
  player: PlayerConfig = utils.field(PlayerConfig)
  controller: ControllerConfig = utils.field(ControllerConfig)
  with_randall_phase: bool = True
  with_randall_xy: bool = False
  items: ItemsConfig = utils.field(ItemsConfig)

NAME_DTYPE = np.int32

def get_state_action_embedding(
  embed_game: Embedding[Game, Any],
  embed_action: Embedding[Action, Any],
  num_names: int,
) -> StructEmbedding[StateAction]:
  embedding = StateAction(
      state=embed_game,
      action=embed_action,
      name=OneHotEmbedding('name', num_names, dtype=NAME_DTYPE),
  )
  return struct_embedding_from_nt("state_action", embedding)

def _stick_to_str(stick):
  return f'({stick[0].item():.2f}, {stick[1].item():.2f})'

def controller_to_str(controller):
  """Pretty-prints a sampled controller."""
  buttons = [b.value for b in LEGAL_BUTTONS if controller['button'][b.value].item()]

  components = [
      f'Main={_stick_to_str(controller["main_stick"])}',
      f'C={_stick_to_str(controller["c_stick"])}',
      ' '.join(buttons),
      f'LS={controller["l_shoulder"].item():.2f}',
      f'RS={controller["r_shoulder"].item():.2f}',
  ]

  return ' '.join(components)
