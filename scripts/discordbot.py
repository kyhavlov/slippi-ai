"""Bot that runs on Discord and lets people play against phillip 2 in doubles matches."""

import copy
import dataclasses
import datetime
import json
import logging
import os
import queue
import threading
import time
import uuid
from typing import Optional, Dict, List, Tuple, Any

from absl import app, flags
import fancyflags as ff
import discord
from discord import app_commands
from discord.ext import commands, tasks
import numpy as np
import ray
import tensorflow as tf

from slippi_ai import data as data_lib
from slippi_ai import flag_utils, eval_lib, nametags, utils, envs as env_lib
from slippi_ai import dolphin as dolphin_lib
from slippi_ai.controller_lib import send_controller
from slippi_db.parse_libmelee import get_game
from slippi_ai.types import Game
from melee.enums import Character
import melee

# Discord settings
default_bot_token = os.environ.get('DISCORD_BOT_TOKEN')
BOT_TOKEN = flags.DEFINE_string(
    'token', default_bot_token, 'Discord bot token',
    required=default_bot_token is None)
COMMAND_PREFIX = flags.DEFINE_string('prefix', '!', 'Command prefix for the bot')
ADMIN_ROLE = flags.DEFINE_string('admin_role', 'AI Admin', 'Role name for admin commands')

# Bot settings
_DOLPHIN_CONFIG = dolphin_lib.DolphinConfig(
    online_delay=15,
    infinite_time=False,
    save_replays=False,
    replay_dir='discordbot/replays',
    disable_audio=True,
    log_types=[],
    render=True,
)
DOLPHIN = ff.DEFINE_dict(
    'dolphin', **flag_utils.get_flags_from_default(_DOLPHIN_CONFIG))

MODELS_PATH = flags.DEFINE_string('models', 'discordbot/models', 'Path to models')

# Serves as the default agent people play against
agent_flags = eval_lib.AGENT_FLAGS.copy()
agent_flags.update(
    async_inference=ff.Boolean(False),
    jit_compile=ff.Boolean(False),
)
AGENT = ff.DEFINE_dict('agent', **agent_flags)
GPU_MODEL = flags.DEFINE_string(
    'gpu_model', None,
    'Basename or path of a single checkpoint to serve on a shared GPU inference actor.')
GPU_MICROBATCH_MS = flags.DEFINE_float(
    'gpu_microbatch_ms', 1.0,
    'Maximum time in milliseconds to wait for additional GPU-model session requests before running a batch.')
GPU_MEMORY_LIMIT_MB = flags.DEFINE_integer(
    'gpu_memory_limit_mb', None,
    'Optional TensorFlow VRAM cap in MB for the shared Discord GPU inference process.')
DEFAULT_PLAYSTYLE_BY_CHARACTER = flags.DEFINE_string(
    'default_playstyle_by_character', '',
    'Semicolon-separated model-scoped character defaults, e.g. '
    'rl_doubles_d21_v4_latest:MARTH=Dragunov,FOX=Ralph,SHEIK=Darkatma')

# Session management settings
MENU_TIMEOUT = flags.DEFINE_float(
    'menu_timeout', 3, 'Minutes before timing out a session in menu')
MAX_SESSIONS = flags.DEFINE_integer(
    'max_sessions', 4, 'Maximum number of concurrent sessions')
STALL_TIMEOUT_SECONDS = 120.0
DEFAULT_PLAYSTYLE_SENTINEL = "__DEFAULT__"

class SessionLaunchError(Exception):
    """User-facing session launch failure."""


PLAYABLE_CHARACTER_CHOICES = [
    (Character.FOX, "Fox"),
    (Character.FALCO, "Falco"),
    (Character.SHEIK, "Sheik"),
    (Character.MARTH, "Marth"),
    (Character.PEACH, "Peach"),
    (Character.CPTFALCON, "Captain Falcon"),
    (Character.JIGGLYPUFF, "Jigglypuff"),
    (Character.PIKACHU, "Pikachu"),
    (Character.SAMUS, "Samus"),
    (Character.YOSHI, "Yoshi"),
    (Character.POPO, "Ice Climbers"),
    (Character.LUIGI, "Luigi"),
    (Character.DK, "Donkey Kong"),
    (Character.GAMEANDWATCH, "Mr. Game & Watch"),
    (Character.GANONDORF, "Ganondorf"),
    (Character.BOWSER, "Bowser"),
    (Character.LINK, "Link"),
    (Character.DOC, "Dr. Mario"),
    (Character.MARIO, "Mario"),
    (Character.NESS, "Ness"),
    (Character.MEWTWO, "Mewtwo"),
    (Character.ROY, "Roy"),
    (Character.ZELDA, "Zelda"),
    (Character.YLINK, "Young Link"),
    (Character.PICHU, "Pichu"),
]


def _filter_autocomplete_choices(
    choices: list[tuple[str, str]],
    current: str,
) -> list[app_commands.Choice[str]]:
    normalized = current.strip().lower()
    if normalized:
        filtered = [
            (name, value) for name, value in choices
            if normalized in name.lower() or normalized in value.lower()
        ]
    else:
        filtered = choices
    return [
        app_commands.Choice(name=name, value=value)
        for name, value in filtered[:25]
    ]


def _supported_state_names(state: dict) -> list[str]:
    rl_names = eval_lib.get_name_from_rl_state(state)
    if rl_names is not None:
        return rl_names

    name_map = state.get('name_map', {})
    if not name_map:
        return []

    by_code = {}
    for name, code in name_map.items():
        by_code.setdefault(code, name)
    return [by_code[code] for code in sorted(by_code)]


def get_supported_playstyle_name(state: dict, playstyle: Optional[str]) -> Optional[str]:
    if not playstyle:
        return None
    normalized = nametags.normalize_name(playstyle)
    for supported_name in _supported_state_names(state):
        supported_normalized = nametags.normalize_name(supported_name)
        if (
            supported_normalized == normalized or
            supported_normalized.lower() == normalized.lower()
        ):
            return supported_name
    return None


def resolve_playstyle_for_state(
    requested_playstyle: Optional[str],
    default_name: Optional[str],
    state: dict,
) -> str:
    """Return a safe nametag for this checkpoint.

    RL checkpoints with `rl_config.agent.name` reject name batches outside that
    list, so resolve user-facing playstyles before calling build_delayed_agent.
    """
    supported_names = _supported_state_names(state)
    if not supported_names:
        return nametags.DEFAULT_NAME

    if requested_playstyle == DEFAULT_PLAYSTYLE_SENTINEL:
        requested_playstyle = None

    for candidate in (requested_playstyle, default_name, nametags.DEFAULT_NAME):
        supported_candidate = get_supported_playstyle_name(state, candidate)
        if supported_candidate is not None:
            return supported_candidate

    fallback = supported_names[0]
    if requested_playstyle:
        logging.warning(
            'Requested playstyle %s is not supported by this model; using %s.',
            requested_playstyle, fallback)
    return fallback


def format_playstyle(requested_playstyle: Optional[str], resolved_playstyle: str) -> str:
    if requested_playstyle == DEFAULT_PLAYSTYLE_SENTINEL:
        return resolved_playstyle
    if not requested_playstyle:
        return resolved_playstyle
    if nametags.normalize_name(requested_playstyle) == nametags.normalize_name(resolved_playstyle):
        return resolved_playstyle
    return f"{resolved_playstyle} ({requested_playstyle} requested, unsupported by this model)"


def format_character(character: Character) -> str:
    aliases = {
        Character.CPTFALCON: 'Captain Falcon',
        Character.JIGGLYPUFF: 'Jigglypuff',
        Character.GAMEANDWATCH: 'Mr. Game & Watch',
        Character.DK: 'Donkey Kong',
        Character.DOC: 'Dr. Mario',
        Character.YLINK: 'Young Link',
        Character.POPO: 'Ice Climbers',
    }
    if character in aliases:
        return aliases[character]
    return character.name.replace('_', ' ').title()


def get_allowed_characters_for_state(state: dict) -> Optional[list[Character]]:
    allowed_characters = (
        state.get('config', {})
        .get('dataset', {})
        .get('allowed_characters')
    )
    if not allowed_characters:
        return None
    return data_lib.chars_from_string(allowed_characters)


def get_playstyle_autocomplete_choices(
    state: dict,
    current: str,
    default_playstyle: Optional[str] = None,
) -> list[app_commands.Choice[str]]:
    supported_names = _supported_state_names(state)
    default_playstyle = (
        get_supported_playstyle_name(state, default_playstyle) or
        (supported_names[0] if supported_names else None)
    )
    current_normalized = current.strip().lower()
    choices = []
    default_label = (
        f'Default ({default_playstyle})'
        if default_playstyle
        else 'None'
    )
    if (
        not current_normalized or
        'default'.startswith(current_normalized) or
        'none'.startswith(current_normalized)
    ):
        choices.append(app_commands.Choice(
            name=default_label,
            value=DEFAULT_PLAYSTYLE_SENTINEL,
        ))
    choices.extend(_filter_autocomplete_choices(
        [
            (name, name) for name in supported_names
            if name != default_playstyle
        ],
        current,
    ))
    return choices[:25]


def get_character_autocomplete_choices(
    state: dict,
    current: str,
) -> list[app_commands.Choice[str]]:
    allowed_characters = get_allowed_characters_for_state(state)
    if allowed_characters is None:
        choices = [(display_name, character.name) for character, display_name in PLAYABLE_CHARACTER_CHOICES]
    else:
        allowed_set = set(allowed_characters)
        choices = [
            (display_name, character.name)
            for character, display_name in PLAYABLE_CHARACTER_CHOICES
            if character in allowed_set
        ]
    return _filter_autocomplete_choices(choices, current)


def format_character_list(characters: list[Character]) -> str:
    return ', '.join(format_character(character) for character in characters)


def format_launch_exception(exc: Exception) -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    message = lines[-1] if lines else repr(exc)
    return message[:1500]


def resolve_gpu_model_path(models_path: str, gpu_model: Optional[str]) -> Optional[str]:
    if not gpu_model:
        return None
    if os.path.isabs(gpu_model):
        return gpu_model
    return os.path.join(models_path, gpu_model)


def configure_tensorflow_gpu_limit(memory_limit_mb: Optional[int]):
    if memory_limit_mb is None:
        return
    if memory_limit_mb <= 0:
        raise ValueError(f'gpu_memory_limit_mb must be positive, got {memory_limit_mb}')
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        logging.warning(
            'Requested Discord GPU memory cap of %d MB, but no GPUs are visible.',
            memory_limit_mb,
        )
        return
    logical_config = [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
    for gpu in gpus:
        tf.config.set_logical_device_configuration(gpu, logical_config)
    logging.info(
        'Configured TensorFlow logical GPU memory limit to %d MB on %d visible GPU(s).',
        memory_limit_mb,
        len(gpus),
    )


def build_discord_player_order(
    my_port: int,
    present_ports: List[int],
    teammate_port: Optional[int],
) -> Tuple[int, int, int, int]:
    all_ports = (1, 2, 3, 4)
    present = [int(p) for p in present_ports]
    if my_port not in present:
        raise ValueError(f'my_port {my_port} not in present_ports {present}')
    if teammate_port is not None and teammate_port not in present:
        raise ValueError(
            f'teammate_port {teammate_port} not in present_ports {present}')

    opponents = [
        p for p in all_ports
        if p in present and p != my_port and p != teammate_port
    ]
    missing = [p for p in all_ports if p not in present]

    if teammate_port is None:
        # In a 3-player lobby without a teammate, keep p1 empty and place
        # the two humans into opponent slots p2/p3.
        ordered = [my_port]
        if missing:
            ordered.append(missing[0])
        ordered.extend(opponents)
    else:
        ordered = [my_port, teammate_port]
        ordered.extend(opponents)

    for port in all_ports:
        if port not in ordered:
            ordered.append(port)
    return tuple(ordered)


class FinalizedDelayBuffer:
    """Publishes exact delayed frames once netplay has finalized them."""

    def __init__(self, delay: int, max_size: int = 300):
        if delay < 0:
            raise ValueError(f"delay must be non-negative, got {delay}")
        self.delay = delay
        self.max_size = max_size
        self._frames = {}
        self._first_frame: Optional[int] = None
        self._next_frame: Optional[int] = None
        self.last_live_frame: Optional[int] = None
        self.last_gap_frame: Optional[int] = None
        self.last_lag_frame: Optional[int] = None
        self.last_lag: Optional[int] = None

    def clear(self):
        self._frames.clear()
        self._first_frame = None
        self._next_frame = None
        self.last_live_frame = None
        self.last_gap_frame = None
        self.last_lag_frame = None
        self.last_lag = None

    def push(self, gamestate: melee.GameState) -> list[melee.GameState]:
        finalized_frame = getattr(gamestate, 'finalized_frame', None)
        if finalized_frame is None:
            return []

        frame = int(gamestate.frame)
        finalized_frame = int(finalized_frame)
        previous_live_frame = self.last_live_frame
        if previous_live_frame is not None and frame > previous_live_frame + 1:
            self.last_gap_frame = previous_live_frame
        self.last_live_frame = frame
        live_lag = frame - finalized_frame
        if self.last_lag is None or live_lag > self.last_lag:
            self.last_lag = live_lag
            self.last_lag_frame = frame
        self._frames[frame] = gamestate
        if self._first_frame is None:
            self._first_frame = frame
            self._next_frame = frame

        target_frame = frame - self.delay
        if self._first_frame is not None:
            target_frame = max(target_frame, self._first_frame)

        ready = []
        while (
            self._next_frame is not None and
            self._next_frame <= target_frame and
            self._next_frame <= finalized_frame
        ):
            next_gamestate = self._frames.get(self._next_frame)
            if next_gamestate is None:
                break
            next_gamestate.custom['discordbot_delayed_finalized'] = True
            ready.append(next_gamestate)
            del self._frames[self._next_frame]
            self._next_frame += 1

        if len(self._frames) > self.max_size:
            min_keep = self._next_frame if self._next_frame is not None else frame
            for old_frame in sorted(self._frames):
                if len(self._frames) <= self.max_size:
                    break
                if old_frame < min_keep:
                    del self._frames[old_frame]

        return ready


class DiscordInGameFrameProcessor:
    """Converts non-menu Slippstream frames into Discord inference frames."""

    def __init__(self, *, logical_port: int, observation_delay: int):
        self.logical_port = logical_port
        self.observation_delay = int(observation_delay)
        self.finalized_buffer = FinalizedDelayBuffer(self.observation_delay)
        self.last_live_frame: Optional[int] = None
        self.last_logged_gap_frame: Optional[int] = None
        self.last_logged_missing_frame: Optional[int] = None
        self.last_logged_lag_bucket: Optional[int] = None
        self._logged_finalized_start = False

    def clear(self):
        self.finalized_buffer.clear()
        self.last_live_frame = None
        self.last_logged_gap_frame = None
        self.last_logged_missing_frame = None
        self.last_logged_lag_bucket = None

    def process(self, gamestate: melee.GameState) -> list[melee.GameState]:
        frame = int(gamestate.frame)
        if frame < 0:
            self.clear()
            gamestate.custom['discordbot_delayed_finalized'] = True
            return [gamestate]

        finalized_frame = getattr(gamestate, 'finalized_frame', None)
        if finalized_frame is None:
            raise RuntimeError(
                f'In-game frame missing finalized_frame for logical_port={self.logical_port} '
                f'frame={gamestate.frame}')

        if self.last_live_frame is not None:
            if frame <= self.last_live_frame:
                logging.warning(
                    "Discord bot live frame rollback/correction "
                    "logical_port=%s from=%s to=%s finalized=%s",
                    self.logical_port, self.last_live_frame, frame, finalized_frame)
            elif frame > self.last_live_frame + 1:
                logging.warning(
                    "Discord bot live frame gap logical_port=%s "
                    "from=%s to=%s skipped=%s finalized=%s",
                    self.logical_port, self.last_live_frame, frame,
                    frame - self.last_live_frame - 1, finalized_frame)
        self.last_live_frame = frame

        if self.finalized_buffer.delay != self.observation_delay:
            self.finalized_buffer = FinalizedDelayBuffer(self.observation_delay)
            self._logged_finalized_start = False

        if not self._logged_finalized_start:
            logging.info(
                "Using finalized Slippstream frames for logical_port=%s delay=%s",
                self.logical_port, self.observation_delay)
            self._logged_finalized_start = True

        lag = frame - int(finalized_frame)
        lag_bucket = lag // 5
        if lag >= 5 and lag_bucket != self.last_logged_lag_bucket:
            logging.info(
                "Discord bot finalized lag logical_port=%s "
                "live=%s finalized=%s lag=%s delay=%s",
                self.logical_port, frame, finalized_frame, lag,
                self.observation_delay)
            self.last_logged_lag_bucket = lag_bucket

        published = self.finalized_buffer.push(gamestate)
        if (
            self.finalized_buffer.last_gap_frame is not None and
            self.finalized_buffer.last_gap_frame != self.last_logged_gap_frame
        ):
            logging.warning(
                "Discord bot delayed-finalized missing frame logical_port=%s "
                "missing=%s target=%s live=%s finalized=%s",
                self.logical_port,
                self.finalized_buffer.last_gap_frame,
                self.finalized_buffer.next_frame,
                frame,
                finalized_frame,
            )
            self.last_logged_gap_frame = self.finalized_buffer.last_gap_frame
        return published

def _make_dummy_raw_game(batch_size: int) -> Game:
    game = utils.map_nt(
        lambda t: np.zeros([batch_size], dtype=t),
        env_lib.reified_game,
    )
    game = game._replace(
        stage=np.full([batch_size], melee.Stage.FINAL_DESTINATION.value, dtype=np.uint8),
        is_teams=np.full([batch_size], True, dtype=np.bool_),
    )
    for player_name in ('p0', 'p1', 'p2', 'p3'):
        player = getattr(game, player_name)
        player = player._replace(
            character=np.full([batch_size], melee.Character.FOX.value, dtype=np.uint8),
            stocks_left=np.full([batch_size], 4, dtype=np.uint8),
            is_dead=np.full([batch_size], False, dtype=np.bool_),
        )
        game = game._replace(**{player_name: player})
    return game


@dataclasses.dataclass
class _SessionGpuAgent:
    agent: Any
    batch_size: int


class LocalDiscordGpuInferenceServer:
    """Single-process GPU owner that uses the normal delayed-agent path."""

    def __init__(
        self,
        *,
        model_path: str,
        console_delay: int,
        max_sessions: int,
        max_local_bots: int,
        batch_window_ms: float,
        compile: bool,
        jit_compile: bool,
        sample_temperature: float,
        batch_steps: int,
    ):
        if batch_steps != 0:
            raise ValueError(
                'Discord GPU inference server currently requires batch_steps=0.')
        self.model_path = model_path
        self.console_delay = int(console_delay)
        self.max_total_batch = int(max_sessions) * int(max_local_bots)
        del batch_window_ms
        self._state = eval_lib.load_state(path=model_path)
        policy_delay = int(self._state['config']['policy']['delay'])
        if self.console_delay > policy_delay:
            raise ValueError(
                f'console_delay={self.console_delay} exceeds policy delay={policy_delay}.')
        self.delay = policy_delay - self.console_delay
        self._compile = compile
        self._jit_compile = jit_compile
        self._sample_temperature = sample_temperature
        self._closed = False
        self._session_agents: dict[str, _SessionGpuAgent] = {}
        self._infer_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._batches_processed = 0
        self._last_batch_size = 0
        self._max_batch_size_seen = 0

    def warmup(self):
        start_time = time.time()
        warm_names = [self._default_name()] * min(3, self.max_total_batch)
        agent = self._build_agent(warm_names)
        agent.start()
        self._warm_agent(agent, len(warm_names))
        agent.stop()
        logging.info(
            'Warmed local Discord GPU inference server for %s in %.3fs',
            os.path.basename(self.model_path),
            time.time() - start_time,
        )
        return dict(delay=self.delay, max_total_batch=self.max_total_batch)

    def model_metadata(self):
        return {
            key: copy.deepcopy(self._state[key])
            for key in ('config', 'name_map', 'rl_config', 'agent_config')
            if key in self._state
        }

    def register_session(self, session_key: str, names: list[str]):
        if not names:
            raise ValueError('GPU-served sessions require at least one agent name.')
        with self._lifecycle_lock:
            if session_key in self._session_agents:
                raise ValueError(f'Session {session_key} already registered.')
            agent = self._build_agent(names)
            agent.start()
            self._warm_agent(agent, len(names))
            self._session_agents[session_key] = _SessionGpuAgent(
                agent=agent,
                batch_size=len(names),
            )
        return dict(delay=self.delay, batch_size=len(names))

    def unregister_session(self, session_key: str):
        with self._lifecycle_lock:
            session = self._session_agents.pop(session_key, None)
        if session is not None:
            session.agent.stop()

    def infer(
        self,
        session_key: str,
        game: Game,
        needs_reset: np.ndarray,
    ):
        if self._closed:
            raise RuntimeError('Discord GPU inference server is closed.')
        session = self._session_agents.get(session_key)
        if session is None:
            raise ValueError(f'Unknown GPU session {session_key}.')
        if len(needs_reset) != session.batch_size:
            raise ValueError(
                f'Session {session_key} expected batch_size={session.batch_size}, '
                f'got {len(needs_reset)}')
        with self._infer_lock:
            self._batches_processed += 1
            self._last_batch_size = len(needs_reset)
            self._max_batch_size_seen = max(self._max_batch_size_seen, len(needs_reset))
            sample_outputs = session.agent.step_undelayed(game, needs_reset)
            return session.agent.embed_controller.decode(sample_outputs.controller_state)

    def stats(self):
        return dict(
            batches_processed=self._batches_processed,
            last_batch_size=self._last_batch_size,
            max_batch_size_seen=self._max_batch_size_seen,
        )

    def close(self):
        self._closed = True
        with self._lifecycle_lock:
            sessions = list(self._session_agents.values())
            self._session_agents.clear()
        for session in sessions:
            session.agent.stop()

    def _default_name(self) -> str:
        rl_name = eval_lib.get_name_from_rl_state(self._state)
        if rl_name:
            return rl_name[0]
        name_map = self._state.get('name_map', {})
        if name_map:
            return next(iter(name_map))
        return nametags.DEFAULT_NAME

    def _build_agent(self, names: list[str]):
        return eval_lib.build_delayed_agent(
            state=self._state,
            batch_size=len(names),
            console_delay=self.console_delay,
            name=names,
            async_inference=False,
            sample_temperature=self._sample_temperature,
            compile=self._compile,
            jit_compile=self._jit_compile,
            batch_steps=0,
            run_on_cpu=False,
        )

    def _warm_agent(self, agent: Any, batch_size: int):
        game = _make_dummy_raw_game(batch_size)
        needs_reset = np.zeros([batch_size], dtype=np.bool_)
        for _ in range(3):
            sample_outputs = agent.step_undelayed(game, needs_reset)
            agent.embed_controller.decode(sample_outputs.controller_state)


@dataclasses.dataclass
class BotSpec:
    """Configuration for one local bot inside a Discord session."""

    logical_port: int
    team_color: int
    character: Optional[Character]
    playstyle: Optional[str] = None
    user_json_path: Optional[str] = None

class DoublesSession:
    """Session actor that owns all local bots and one fused inference worker."""

    def __init__(
        self,
        dolphin_config: dolphin_lib.DolphinConfig,
        agent_kwargs: dict,
        connect_code: str,
        bot_specs: List[BotSpec],
        gpu_server: Optional[Any] = None,
        gpu_model_basename: Optional[str] = None,
        disable_process_gpus: bool = True,
    ):
        if disable_process_gpus:
            eval_lib.disable_gpus()
        self.dolphin_config = dolphin_config
        self.agent_kwargs = agent_kwargs
        self.connect_code = connect_code
        self.bot_specs = list(bot_specs)
        self._gpu_server = gpu_server
        self._gpu_model_basename = gpu_model_basename
        self._gpu_session_key: Optional[str] = None
        self._gpu_serving_enabled = False
        self._resolved_agent_names: list[str] = []
        self._policy_observation_delay = 0

        self.stop_requested = threading.Event()
        self._lock = threading.RLock()
        self._dolphins_lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._reader_threads: Dict[int, threading.Thread] = {}
        self._num_menu_frames = 0
        self._last_error: Optional[str] = None

        self._state: Optional[dict] = None
        self._agent = None
        self._current_agent_label: Optional[str] = None
        self._pending_agent_kwargs: Optional[dict] = None
        self._pending_agent_label: Optional[str] = None
        self._dolphins: Dict[int, dolphin_lib.Dolphin] = {}
        self._players: Dict[int, dolphin_lib.AI] = {}
        self._controllers: Dict[int, melee.Controller] = {}
        self._stopped_dolphins: set[int] = set()
        self._bot_codes: Dict[int, str] = {}
        self._player_orders: Dict[int, Tuple[int, ...]] = {}
        self._teammate_ports: Dict[int, Optional[int]] = {}
        self._no_teammate_signatures: Dict[int, Tuple[int, Tuple[int, ...]]] = {}
        self._player_order_signatures: Dict[int, Tuple[int, Optional[int], Tuple[int, ...]]] = {}
        self._dead_frames: Dict[int, int] = {spec.logical_port: 0 for spec in self.bot_specs}
        self._pressed_start: Dict[int, bool] = {spec.logical_port: False for spec in self.bot_specs}
        self._has_entered_game = False
        self._in_menu = True
        now = time.monotonic()
        self._last_activity_time = now
        self._last_frame_advance_time = now
        self._last_frame_signature: Optional[Tuple[int, ...]] = None
        
    def num_menu_frames(self) -> int:
        return self._num_menu_frames

    def status(self) -> dict:
        now = time.monotonic()
        return {
            'is_alive': self._thread.is_alive() if self._thread else False,
            'num_menu_frames': self._num_menu_frames,
            'last_error': self._last_error,
            'current_agent': self._current_agent_label,
            'pending_agent': self._pending_agent_label,
            'has_entered_game': self._has_entered_game,
            'in_menu': self._in_menu,
            'seconds_since_activity': now - self._last_activity_time,
            'seconds_since_frame_advance': now - self._last_frame_advance_time,
        }

    def set_agent(self, agent_kwargs: dict) -> dict:
        """Request a model switch at the next game boundary."""
        if self._gpu_serving_enabled:
            raise RuntimeError(
                'Active session model switching is not yet supported for GPU-served Discord sessions.')
        with self._lock:
            self._pending_agent_kwargs = agent_kwargs.copy()
            self._pending_agent_label = self._agent_label(agent_kwargs)
            logging.info(
                "Queued Discord fused session model switch to %s",
                self._pending_agent_label)
            return {
                'current_agent': self._current_agent_label,
                'pending_agent': self._pending_agent_label,
            }

    def start(self):
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("DoublesSession is already started.")

            try:
                self._set_state_and_agent(self.agent_kwargs)
                self._register_gpu_session_if_needed()
                if self._agent is not None:
                    warmup_start = time.time()
                    self._agent.warmup()
                    logging.info(
                        "Warmed Discord session agent %s in %.3fs",
                        self._current_agent_label,
                        time.time() - warmup_start,
                    )
                self._start_dolphins()
                if self._agent is not None:
                    self._agent.start()

                self._thread = threading.Thread(
                    target=self._run_loop,
                    name=f"DiscordDoublesSession-{self.connect_code}",
                    daemon=True,
                )
                self._thread.start()
            except Exception:
                if self._agent is not None:
                    self._agent.stop()
                    self._agent = None
                self._unregister_gpu_session_if_needed()
                self._stop_dolphins()
                raise

    def stop(self):
        self.stop_requested.set()
        self._stop_dolphins(clear=False)

        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=10)

        if self._agent is not None:
            self._agent.stop()
            self._agent = None
        self._unregister_gpu_session_if_needed()

    def _uses_gpu_model(self, source_agent_kwargs: dict) -> bool:
        if self._gpu_server is None or not self._gpu_model_basename:
            return False
        path = source_agent_kwargs.get('path')
        return bool(path) and os.path.basename(path) == self._gpu_model_basename

    def _agent_label(self, agent_kwargs: dict) -> str:
        path = agent_kwargs.get('path')
        tag = agent_kwargs.get('tag')
        if path:
            return os.path.basename(path)
        if tag:
            return str(tag)
        return '<unknown>'

    def _build_state_and_agent(self, source_agent_kwargs: dict):
        agent_kwargs = source_agent_kwargs.copy()
        path = agent_kwargs.pop('path', None)
        tag = agent_kwargs.pop('tag', None)
        use_gpu_serving = self._uses_gpu_model(source_agent_kwargs)
        name_change_mode = agent_kwargs.pop(
            'name_change_mode', eval_lib.NameChangeMode.FIXED)
        if name_change_mode != eval_lib.NameChangeMode.FIXED:
            logging.warning(
                "Discord fused sessions currently use a fixed agent name; got %s",
                name_change_mode)
        async_inference = agent_kwargs.pop('async_inference', False)
        if async_inference:
            logging.info(
                "Ignoring --agent.async_inference=True for Discord fused sessions; "
                "the Ray session actor is the single inference worker.")

        if use_gpu_serving:
            if tag is not None:
                raise ValueError('GPU-served Discord sessions currently require a local checkpoint path.')
            state = self._gpu_server.model_metadata()
        else:
            state = eval_lib.load_state(path=path, tag=tag)
        default_name = agent_kwargs.pop('name', None)
        resolved_names = [
            resolve_playstyle_for_state(spec.playstyle, default_name, state)
            for spec in self.bot_specs
        ]
        logging.info(
            "Discord fused session playstyles: %s",
            {
                spec.logical_port: name
                for spec, name in zip(self.bot_specs, resolved_names)
            },
        )
        self._resolved_agent_names = resolved_names
        if use_gpu_serving:
            return state, None, self._agent_label(source_agent_kwargs), True

        agent_kwargs['name'] = resolved_names
        agent = eval_lib.build_delayed_agent(
            state=state,
            batch_size=len(self.bot_specs),
            console_delay=self.dolphin_config.online_delay,
            run_on_cpu=True,
            async_inference=False,
            **agent_kwargs,
        )
        if agent.batch_steps > agent.delay + 1:
            raise ValueError(
                f"agent.batch_steps={agent.batch_steps} exceeds delay slack "
                f"for policy delay={agent.delay} after console delay.")
        return state, agent, self._agent_label(source_agent_kwargs), False

    def _set_state_and_agent(self, source_agent_kwargs: dict):
        state, agent, label, gpu_serving_enabled = self._build_state_and_agent(source_agent_kwargs)
        self._state = state
        self._agent = agent
        self._current_agent_label = label
        self._gpu_serving_enabled = gpu_serving_enabled
        self._policy_observation_delay = (
            int(self._state['config']['policy']['delay']) - self.dolphin_config.online_delay
        )
        self.agent_kwargs = source_agent_kwargs.copy()

    def _register_gpu_session_if_needed(self):
        if not self._gpu_serving_enabled or self._gpu_server is None:
            return
        if self._gpu_session_key is None:
            actor_id = ray.get_runtime_context().get_actor_id()
            if actor_id:
                self._gpu_session_key = f"{actor_id}:{self.connect_code}"
            else:
                self._gpu_session_key = f"local:{uuid.uuid4().hex}:{self.connect_code}"
        self._gpu_server.register_session(
            self._gpu_session_key,
            self._resolved_agent_names,
        )

    def _unregister_gpu_session_if_needed(self):
        if self._gpu_session_key is None or self._gpu_server is None:
            return
        try:
            self._gpu_server.unregister_session(self._gpu_session_key)
        except Exception:
            if not self.stop_requested.is_set():
                logging.exception("Failed to unregister Discord GPU inference session")
        finally:
            self._gpu_session_key = None

    def _apply_pending_agent_if_needed(self):
        with self._lock:
            pending_agent_kwargs = self._pending_agent_kwargs
            pending_agent_label = self._pending_agent_label
            if pending_agent_kwargs is None:
                return

        logging.info(
            "Applying pending Discord fused session model switch to %s",
            pending_agent_label)
        if self._gpu_serving_enabled or self._uses_gpu_model(pending_agent_kwargs):
            self._last_error = (
                f"Failed to switch agent to {pending_agent_label}: "
                "GPU-served Discord sessions do not support active model switching yet.")
            with self._lock:
                self._pending_agent_kwargs = None
                self._pending_agent_label = None
            return
        old_agent = self._agent
        try:
            state, agent, label, gpu_serving_enabled = self._build_state_and_agent(pending_agent_kwargs)
            agent.start()
        except Exception as exc:
            self._last_error = f"Failed to switch agent to {pending_agent_label}: {exc}"
            logging.exception(
                "Failed to apply pending Discord fused session model switch to %s",
                pending_agent_label)
            with self._lock:
                self._pending_agent_kwargs = None
                self._pending_agent_label = None
            return

        self._state = state
        self._agent = agent
        self._current_agent_label = label
        self._gpu_serving_enabled = gpu_serving_enabled
        self.agent_kwargs = pending_agent_kwargs.copy()
        with self._lock:
            self._pending_agent_kwargs = None
            self._pending_agent_label = None
        if old_agent is not None:
            old_agent.stop()
        logging.info("Switched Discord fused session model to %s", label)

    def _start_dolphins(self):
        for index, spec in enumerate(self.bot_specs):
            config = self._dolphin_config_for_spec(spec, save_replays=(index == 0))
            with open(config.user_json_path) as f:
                self._bot_codes[spec.logical_port] = json.load(f)['connectCode']

            player = dolphin_lib.AI()
            if spec.character is not None:
                player.character = spec.character
            eval_lib.update_character(player, self._state['config'])

            dolphin_kwargs = config.to_kwargs()
            dolphin_kwargs['headless'] = config.headless
            dolphin_kwargs['render'] = config.render

            logging.info(
                "Launching Discord bot Dolphin logical_port=%s save_replays=%s "
                "user_json=%s slippi_port=%s team=%s",
                spec.logical_port, config.save_replays, config.user_json_path,
                config.slippi_port, spec.team_color)
            dolphin = dolphin_lib.Dolphin(
                players={1: player},
                desired_teams={1: spec.team_color},
                **dolphin_kwargs,
            )
            log_path = os.path.join(
                dolphin.console._get_dolphin_home_path(), 'Logs', 'dolphin.log')
            logging.info(
                "Started Discord bot Dolphin logical_port=%s temp_dir=%s log=%s",
                spec.logical_port, dolphin.console.temp_dir, log_path)

            self._players[spec.logical_port] = player
            self._dolphins[spec.logical_port] = dolphin
            self._controllers[spec.logical_port] = dolphin.controllers[1]

    def _dolphin_config_for_spec(
        self,
        spec: BotSpec,
        save_replays: bool,
    ) -> dolphin_lib.DolphinConfig:
        config = dataclasses.replace(self.dolphin_config)
        config.slippi_port = utils.find_open_udp_port()
        config.connect_code = self.connect_code
        config.teams_connect_code = self.connect_code
        config.save_replays = save_replays
        if spec.user_json_path:
            config.user_json_path = spec.user_json_path
        return config

    def _run_loop(self):
        gamestate_queues: Dict[int, queue.Queue] = {
            logical_port: queue.Queue(maxsize=1)
            for logical_port in self._dolphins
        }
        advance_events: Dict[int, threading.Event] = {
            logical_port: threading.Event()
            for logical_port in self._dolphins
        }

        def read_gamestates(logical_port: int, dolphin: dolphin_lib.Dolphin):
            processor = DiscordInGameFrameProcessor(
                logical_port=logical_port,
                observation_delay=int(self._policy_observation_delay),
            )

            def publish_gamestate(gamestate: melee.GameState):
                while not self.stop_requested.is_set():
                    try:
                        gamestate_queues[logical_port].put(gamestate, timeout=0.5)
                        break
                    except queue.Full:
                        continue
                advance_event = advance_events[logical_port]
                while (
                    not self.stop_requested.is_set() and
                    not advance_event.wait(timeout=0.5)
                ):
                    pass
                advance_event.clear()

            try:
                for gamestate in dolphin.iter_gamestates(skip_menu_frames=False):
                    if self.stop_requested.is_set():
                        return
                    if dolphin_lib.is_menu_state(gamestate):
                        processor.clear()
                        publish_gamestate(gamestate)
                        continue

                    for published in processor.process(gamestate):
                        publish_gamestate(published)
            except Exception as exc:
                if not self.stop_requested.is_set():
                    logging.exception(
                        "Discord bot Dolphin reader crashed for logical_port=%s",
                        logical_port)
                    gamestate_queues[logical_port].put(exc)

        self._reader_threads = {
            logical_port: threading.Thread(
                target=read_gamestates,
                args=(logical_port, dolphin),
                name=f"DiscordDolphinReader-{logical_port}",
                daemon=True,
            )
            for logical_port, dolphin in self._dolphins.items()
        }
        for thread in self._reader_threads.values():
            thread.start()

        try:
            while not self.stop_requested.is_set():
                current = {}
                for logical_port, gamestate_queue in gamestate_queues.items():
                    while not self.stop_requested.is_set():
                        try:
                            item = gamestate_queue.get(timeout=0.5)
                            break
                        except queue.Empty:
                            continue
                    if self.stop_requested.is_set():
                        return
                    if isinstance(item, Exception):
                        raise item
                    current[logical_port] = item
                self._last_activity_time = time.monotonic()

                if any(dolphin_lib.is_menu_state(gs) for gs in current.values()):
                    self._in_menu = True
                    self._num_menu_frames += 1
                    for event in advance_events.values():
                        event.set()
                    continue

                self._in_menu = False
                self._num_menu_frames = 0
                self._has_entered_game = True
                frame_signature = tuple(current[p].frame for p in sorted(current))
                if frame_signature != self._last_frame_signature:
                    self._last_frame_signature = frame_signature
                    self._last_frame_advance_time = self._last_activity_time
                if any(gs.frame == -123 for gs in current.values()):
                    self._apply_pending_agent_if_needed()

                games = []
                needs_reset = []
                ordered_ports = [spec.logical_port for spec in self.bot_specs]

                for logical_port in ordered_ports:
                    gamestate = current[logical_port]
                    new_game = gamestate.frame == -123
                    if (
                        new_game or
                        logical_port not in self._player_orders or
                        self._teammate_ports.get(logical_port) is None
                    ):
                        self._set_player_ports(logical_port, gamestate)
                    game = get_game(gamestate, ports=self._player_orders[logical_port])
                    games.append(utils.map_nt(lambda x: np.expand_dims(x, 0), game))
                    needs_reset.append(new_game)

                batched_game = utils.map_nt(
                    lambda *xs: np.concatenate(xs, axis=0), *games)
                assert all(
                    gs.custom.get('discordbot_delayed_finalized', False)
                    for gs in current.values()
                ), 'Expected only finalized delayed in-game frames in Discord bot session.'
                if self._gpu_serving_enabled:
                    decoded = self._gpu_server.infer(
                        self._gpu_session_key,
                        batched_game,
                        np.array(needs_reset, dtype=np.bool_),
                    )
                else:
                    sample_outputs = self._agent.step_undelayed(
                        batched_game,
                        np.array(needs_reset, dtype=np.bool_),
                    )
                    decoded = self._agent.embed_controller.decode(
                        sample_outputs.controller_state)

                for index, logical_port in enumerate(ordered_ports):
                    controller = self._controllers[logical_port]
                    game = games[index]
                    gamestate = current[logical_port]
                    if self._is_game_over(game, gamestate):
                        controller.release_all()
                        continue

                    action = utils.map_single_structure(lambda x: x[index], decoded)
                    send_controller(controller, action)
                    self._maybe_stock_steal(logical_port, game, gamestate, controller)

                for event in advance_events.values():
                    event.set()

        except Exception as exc:
            if not self.stop_requested.is_set():
                self._last_error = str(exc)
                logging.exception("Discord fused session crashed")
        finally:
            if self._agent is not None:
                self._agent.stop()
                self._agent = None
            self._stop_dolphins()
            for thread in self._reader_threads.values():
                if thread.is_alive():
                    thread.join(timeout=2)

    def _set_player_ports(self, logical_port: int, gamestate: melee.GameState):
        my_port = getattr(gamestate, 'local_player_port', None)
        if my_port not in gamestate.players:
            bot_code = self._bot_codes[logical_port]
            matching_ports = [
                port for port, player in gamestate.players.items()
                if player.connectCode == bot_code
            ]
            if len(matching_ports) != 1:
                raise RuntimeError(
                    f"Could not uniquely identify bot port for {bot_code}: "
                    f"{matching_ports}")
            my_port = matching_ports[0]

        teammate_port = None
        for port, player in gamestate.players.items():
            if port == my_port:
                continue
            if player.team_id == gamestate.players[my_port].team_id:
                teammate_port = int(port)
                break

        if teammate_port is None:
            no_teammate_signature = (
                int(my_port),
                tuple(sorted(int(port) for port in gamestate.players)),
            )
            if self._no_teammate_signatures.get(logical_port) != no_teammate_signature:
                self._no_teammate_signatures[logical_port] = no_teammate_signature
                logging.warning(
                    "Discord bot could not identify teammate yet "
                    "logical_port=%s local_player_port=%s players=%s",
                    logical_port, my_port, sorted(gamestate.players))
        else:
            self._no_teammate_signatures.pop(logical_port, None)

        player_order = build_discord_player_order(
            int(my_port),
            [int(port) for port in gamestate.players],
            teammate_port,
        )
        self._player_orders[logical_port] = player_order
        self._teammate_ports[logical_port] = teammate_port
        signature = (int(my_port), teammate_port, player_order)
        if self._player_order_signatures.get(logical_port) != signature:
            self._player_order_signatures[logical_port] = signature
            logging.info(
                "Discord bot logical_port=%s local_player_port=%s teammate_port=%s "
                "player_order=%s",
                logical_port, my_port, teammate_port, player_order)

    def _maybe_stock_steal(
        self,
        logical_port: int,
        game,
        gamestate: melee.GameState,
        controller: melee.Controller,
    ):
        teammate_port = self._teammate_ports.get(logical_port)
        if self._as_bool(game.p0.is_dead):
            self._dead_frames[logical_port] += 1
            teammate_has_stock = (
                teammate_port in gamestate.players and
                gamestate.players[teammate_port].stock > 1
            )
            if (
                not self._pressed_start[logical_port] and
                self._dead_frames[logical_port] >= 120 and
                teammate_has_stock
            ):
                logging.info(
                    "logical_port=%s p0 is dead, stock stealing from %s",
                    logical_port, teammate_port)
                controller.press_button(melee.Button.BUTTON_START)
                self._pressed_start[logical_port] = True
            elif self._pressed_start[logical_port]:
                controller.release_button(melee.Button.BUTTON_START)
                self._pressed_start[logical_port] = False
        else:
            self._dead_frames[logical_port] = 0

    def _is_game_over(self, game, gamestate: melee.GameState) -> bool:
        left_team_out = (
            self._as_bool(game.p0.stocks_left == 0) and
            self._as_bool(game.p1.stocks_left == 0)
        )
        right_team_out = (
            self._as_bool(game.p2.stocks_left == 0) and
            self._as_bool(game.p3.stocks_left == 0)
        )
        return left_team_out or right_team_out or gamestate.frame >= 28799

    def _as_bool(self, value) -> bool:
        return bool(np.asarray(value).reshape(-1)[0])

    def _stop_dolphins(self, clear: bool = True):
        with self._dolphins_lock:
            dolphins = list(self._dolphins.items())
        for logical_port, dolphin in dolphins:
            with self._dolphins_lock:
                if logical_port in self._stopped_dolphins:
                    continue
                self._stopped_dolphins.add(logical_port)
            try:
                dolphin.stop()
            except Exception:
                if not self.stop_requested.is_set():
                    logging.exception("Failed to stop Discord bot Dolphin")
        if clear:
            with self._dolphins_lock:
                self._dolphins.clear()
                self._controllers.clear()


RemoteDoublesSession = ray.remote(DoublesSession)

class RemoteSessionHandle:

    def __init__(self, actor):
        self._actor = actor

    def start(self):
        return ray.get(self._actor.start.remote())

    def stop(self):
        return ray.get(self._actor.stop.remote())

    def status(self):
        return ray.get(self._actor.status.remote())

    def set_agent(self, agent_kwargs: dict):
        return ray.get(self._actor.set_agent.remote(agent_kwargs))

    def kill(self):
        return ray.kill(self._actor, no_restart=True)


class LocalSessionHandle:

    def __init__(self, session: DoublesSession):
        self._session = session

    def start(self):
        return self._session.start()

    def stop(self):
        return self._session.stop()

    def status(self):
        return self._session.status()

    def set_agent(self, agent_kwargs: dict):
        return self._session.set_agent(agent_kwargs)

    def kill(self):
        return None

@dataclasses.dataclass
class SessionInfo:
    session: Any  # RemoteDoublesSession
    start_time: datetime.datetime
    discord_name: str
    discord_id: int
    connect_code: str
    agents: Dict[int, str]  # port -> agent name
    team_colors: Dict[int, int]  # port -> team color
    playstyles: Dict[int, str]  # port -> requested playstyle
    bot_specs: List[BotSpec]
    launch_message: discord.Message
    playing_announced: bool = False


def format_team_name(team_color: int) -> str:
    return ["Red", "Blue", "Green"][team_color]


def format_team_marker(team_color: int) -> str:
    return ["🟥", "🟦", "🟩"][team_color]


def format_td(td: datetime.timedelta) -> str:
    """Chop off microseconds."""
    return str(td).split('.')[0]

def get_character_from_name(name: str) -> Character:
    """Convert a character name string to the corresponding Character enum value.
    
    Accepts partial names, ignores case, and handles some common aliases.
    """
    name = name.lower().strip()
    
    # Common aliases mapping
    aliases = {
        "falcon": Character.CPTFALCON,
        "captain falcon": Character.CPTFALCON,
        "doc": Character.DOC,
        "dr mario": Character.DOC,
        "ganon": Character.GANONDORF,
        "jiggs": Character.JIGGLYPUFF,
        "puff": Character.JIGGLYPUFF,
        "g&w": Character.GAMEANDWATCH,
        "game & watch": Character.GAMEANDWATCH,
        "game and watch": Character.GAMEANDWATCH,
        "young link": Character.YLINK,
        "ylink": Character.YLINK,
        "sheik": Character.SHEIK,
        "icies": Character.POPO,
        "ice climbers": Character.POPO,
        "dk": Character.DK,
        "peach": Character.PEACH
    }
    
    # Check if name is a direct alias
    if name in aliases:
        return aliases[name]
    
    # Try to find a match in Character enum
    for char in Character:
        if char == Character.WIREFRAME_MALE or char == Character.WIREFRAME_FEMALE or char == Character.GIGA_BOWSER or char == Character.SANDBAG or char == Character.UNKNOWN_CHARACTER or char == Character.NANA:
            continue
            
        char_name = char.name.lower()
        if name == char_name or name in char_name:
            return char
    
    # Default to Fox if no match found
    logging.warning(f"Could not find character matching '{name}', defaulting to Fox")
    return Character.FOX


def parse_character_config_key(name: str) -> Character:
    normalized = name.strip().upper().replace(' ', '').replace('-', '').replace('.', '')
    aliases = {
        'CAPTAINFALCON': Character.CPTFALCON,
        'FALCON': Character.CPTFALCON,
        'DRMARIO': Character.DOC,
        'DOC': Character.DOC,
        'DONKEYKONG': Character.DK,
        'DK': Character.DK,
        'MRGAMEANDWATCH': Character.GAMEANDWATCH,
        'GAMEANDWATCH': Character.GAMEANDWATCH,
        'ICECLIMBERS': Character.POPO,
        'ICIES': Character.POPO,
        'YOUNGLINK': Character.YLINK,
    }
    if normalized in aliases:
        return aliases[normalized]
    return Character[normalized]


def parse_default_playstyle_by_character(
    raw_config: str,
) -> Dict[str, Dict[Character, str]]:
    defaults: Dict[str, Dict[Character, str]] = {}
    if not raw_config.strip():
        return defaults

    for model_group in raw_config.split(';'):
        model_group = model_group.strip()
        if not model_group:
            continue
        if ':' not in model_group:
            raise ValueError(
                f'Default playstyle group "{model_group}" is missing model: prefix')
        model_name, assignments_text = model_group.split(':', 1)
        model_name = os.path.basename(model_name.strip())
        if not model_name:
            raise ValueError(f'Default playstyle group "{model_group}" has empty model')

        model_defaults = defaults.setdefault(model_name, {})
        for assignment in assignments_text.split(','):
            assignment = assignment.strip()
            if not assignment:
                continue
            if '=' not in assignment:
                raise ValueError(
                    f'Default playstyle assignment "{assignment}" is missing =')
            character_name, playstyle = assignment.split('=', 1)
            playstyle = playstyle.strip()
            if not playstyle:
                raise ValueError(
                    f'Default playstyle assignment "{assignment}" has empty playstyle')
            model_defaults[parse_character_config_key(character_name)] = playstyle

    return defaults


def get_character_default_playstyle(
    agent_name: str,
    character: Optional[Character],
    default_playstyles_by_character: Dict[str, Dict[Character, str]],
    state: dict,
) -> Optional[str]:
    if character is None:
        return None

    model_defaults = default_playstyles_by_character.get(os.path.basename(agent_name))
    if not model_defaults:
        return None

    configured_default = model_defaults.get(character)
    supported_default = get_supported_playstyle_name(state, configured_default)
    if supported_default is not None:
        return supported_default

    if configured_default:
        logging.warning(
            'Configured default playstyle %s for %s/%s is not supported by this model.',
            configured_default, agent_name, character.name)
    return None


def get_optional_character_from_name(name: Optional[str]) -> Optional[Character]:
    if not name:
        return None
    try:
        return parse_character_config_key(name)
    except KeyError:
        normalized = name.strip().lower()
        for character, display_name in PLAYABLE_CHARACTER_CHOICES:
            if normalized in (character.name.lower(), display_name.lower()):
                return character
    return None


def get_valid_character_choices():
    """Return a list of character choices for the Discord API."""
    return [
        app_commands.Choice(name=display_name, value=char.name)
        for char, display_name in PLAYABLE_CHARACTER_CHOICES
    ]

# Custom command tree that restricts commands to specific channels
class ChannelRestrictedCommandTree(app_commands.CommandTree):
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Check if the interaction is from an allowed channel"""
        # Allow commands through direct messages
        if interaction.guild is None:
            return True

        # Allow interactions in the 'phillip-connect' channel
        if interaction.channel and interaction.channel.name == 'phillip-connect':
            return True

        # Inform the user if they're in the wrong channel
        await interaction.response.send_message(
            "Commands can only be used in the #phillip-connect channel.", 
            ephemeral=True
        )
        return False

class DiscordBot(commands.Bot):
    def __init__(
        self,
        token: str,
        prefix: str,
        dolphin_config: dolphin_lib.DolphinConfig,
        agent_kwargs: dict,
        models_path: str,
        admin_role: str = "AI Admin",
        max_sessions: int = 4,
        menu_timeout: float = 3,  # in minutes
        gpu_server: Optional[Any] = None,
        gpu_model_basename: Optional[str] = None,
        default_playstyles_by_character: Optional[Dict[str, Dict[Character, str]]] = None,
    ):
        # Set up intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.guilds = True
        
        # Initialize bot with command prefix and intents, using our custom command tree
        super().__init__(
            command_prefix=prefix, 
            intents=intents,
            tree_cls=ChannelRestrictedCommandTree  # Use our custom command tree
        )
        
        self.token = token
        self.dolphin_config = dolphin_config
        self.agent_kwargs = agent_kwargs
        self.admin_role = admin_role
        self._max_sessions = max_sessions
        self._menu_timeout = menu_timeout
        self._gpu_server = gpu_server
        self._gpu_model_basename = gpu_model_basename
        self._default_playstyles_by_character = default_playstyles_by_character or {}

        self._sessions: Dict[int, SessionInfo] = {}  # User ID -> SessionInfo
        self.lock = threading.RLock()

        self._models_path = models_path
        self._reload_models()

        self._default_agent_name = os.path.basename(agent_kwargs['path'])
        self._requested_agents = {}  # user_id -> agent_name
        self._play_codes = {}  # user_id -> connect_code

    async def setup_hook(self):
        """Setup hook that runs when the bot is first connecting"""
        # Start the task when the bot is ready
        self._do_chores.start()

    async def on_ready(self):
        """Event triggered when the bot is ready"""
        logging.info(f'Logged in as {self.user}')
        logging.info('------')
        
        # Set up status
        await self.change_presence(activity=discord.Game(name=f"Use /help to learn commands"))
        
        # Setup slash commands
        await self.setup_slash_commands()
        
        logging.info("Slash commands synced")
    
    async def setup_slash_commands(self):
        """Setup and sync all slash commands"""
        # Create a command tree for the bot
        self.tree.clear_commands(guild=None)
        
        # Help command - can be simplified since Discord shows command descriptions
        @self.tree.command(name="help", description="Display help information about the Slippi AI Doubles Bot")
        async def help_command(interaction: discord.Interaction):
            embed = discord.Embed(
                title="Slippi AI Doubles Bot",
                description="Play against or with AI agents in doubles mode.",
                color=discord.Color.blue()
            )
            
            # Add sections with clearer formatting
            embed.add_field(
                name="Basic Commands",
                value="`/play1` - Start a game with one AI agent\n"
                      "`/play2` - Start a game with two AI agents\n"
                      "`/play3` - Start a game with three AI agents\n"
                      "`/stop` - Stop your current game session\n"
                      "`/status` - Show current bot status",
                inline=False
            )
            
            embed.add_field(
                name="Agent Selection",
                value="`/agents` - List available AI agents\n"
                      "`/agent` - Select a specific AI agent",
                inline=False
            )

            embed.add_field(
                name="Playstyles",
                value="The playstyle/nametag option is only supported by the "
                      "`rl_doubles_d18_v2_*` model line. The `rl_doubles_d21_v3_*` "
                      "models do not support playstyles and will use their default "
                      "nametag instead.",
                inline=False
            )
            
            embed.add_field(
                name="How to Play",
                value="Use `/play1`, `/play2`, or `/play3` with your connect code, create a direct lobby, and wait for the AI to join.",
                inline=False
            )
            
            embed.add_field(
                name="Info",
                value=f"Max concurrent sessions: {self._max_sessions}\n",
                inline=False
            )
            
            await interaction.response.send_message(embed=embed)
        
        # Reload models command (admin only)
        @self.tree.command(name="reload", description="Reload available models (Admin only)")
        @app_commands.describe(
            default_model="Optional: Set a new default model"
        )
        async def reload_command(interaction: discord.Interaction, default_model: str = None):
            if not self.is_admin(interaction):
                await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
                return
                
            with self.lock:
                self._reload_models()
                
                # Update default model if provided and valid
                if default_model:
                    if default_model in self._models:
                        self._default_agent_name = default_model
                        await interaction.response.send_message(f"Models reloaded. Default model set to {default_model}")
                    else:
                        # Invalid model name provided
                        models_str = ", ".join(self._models)
                        await interaction.response.send_message(f"Models reloaded. Invalid default model '{default_model}'. Available models: {models_str}")
                        return
                else:
                    # No default model provided, just show available models
                    models_str = ", ".join(self._models)
                    embed = discord.Embed(
                        title="Available Agents",
                        description=models_str,
                        color=discord.Color.green()
                    )
                    embed.add_field(
                        name="Default Agent",
                        value=self._default_agent_name,
                        inline=False
                    )
                    await interaction.response.send_message(embed=embed)
        
        # List agents command
        @self.tree.command(name="agents", description="List available AI agents")
        async def agents_command(interaction: discord.Interaction):
            models_str = ", ".join(self._models)
            embed = discord.Embed(
                title="Available Agents",
                description=models_str,
                color=discord.Color.green()
            )
            await interaction.response.send_message(embed=embed)
        
        # Select agent command
        @self.tree.command(name="agent", description="Select an agent to play against")
        @app_commands.describe(
            agent_name="Name of the agent to use"
        )
        async def agent_command(
            interaction: discord.Interaction, 
            agent_name: str
        ):
            if agent_name not in self._models:
                await interaction.response.send_message(
                    f'{agent_name} is not a valid agent. Available agents: {", ".join(self._models)}',
                    ephemeral=True
                )
                return
                
            user_id = interaction.user.id

            # Just store the agent name as the default agent for this user
            self._requested_agents[user_id] = agent_name  # Always use port 1 as default
            if user_id in self._sessions:
                session_info = self._sessions[user_id]
                try:
                    result = session_info.session.set_agent(
                        self._get_agent_kwargs(user_id, 1, agent_name))
                except Exception:
                    logging.exception("Failed to queue active Discord session agent switch")
                    await interaction.response.send_message(
                        f'Selected agent {agent_name} for future games, but failed to queue '
                        'the active session switch. Use /stop and start again if needed.',
                        ephemeral=True,
                    )
                    return

                await interaction.response.send_message(
                    f'Selected agent {agent_name}. Active session will switch next game '
                    f'(current: {result.get("current_agent")}, pending: {result.get("pending_agent")}).')
                return

            await interaction.response.send_message(f'Selected agent {agent_name} for your games')
        
        # Stop command
        @self.tree.command(name="stop", description="Stop the current game session")
        async def stop_command(interaction: discord.Interaction):
            with self.lock:
                user_id = interaction.user.id
                if user_id not in self._sessions:
                    await interaction.response.send_message(f"{interaction.user.name}, you're not playing right now.", ephemeral=True)
                    return

                self._stop_sessions([self._sessions[user_id]])
                await interaction.response.send_message(f'Stopped playing against {interaction.user.name}')
        
        @self.tree.command(name="play1", description="Start a game with one AI agent")
        @app_commands.describe(
            connect_code="Your Slippi lobby code (no # needed)",
            team_color="Team color for the AI agent",
            character="Character for the AI to play",
            playstyle="Optional nametag/playstyle for supported agents",
        )
        @app_commands.choices(team_color=[
            app_commands.Choice(name="Red", value="red"),
            app_commands.Choice(name="Blue", value="blue"),
            app_commands.Choice(name="Green", value="green"),
        ])
        async def play_command(
            interaction: discord.Interaction, 
            connect_code: str,
            character: str, 
            team_color: str = "red",
            playstyle: Optional[str] = None,
        ):
            with self.lock:
                user_id = interaction.user.id

                if user_id in self._sessions:
                    await interaction.response.send_message(
                        f'{interaction.user.name}, you are already playing', 
                        ephemeral=True
                    )
                    return

                await self._gc_sessions()

                if len(self._sessions) >= self._max_sessions:
                    await interaction.response.send_message(
                        'Sorry, too many sessions already active.', 
                        ephemeral=True
                    )
                    return

                # Validate the connect code
                is_valid, error_message = self._is_valid_connect_code(connect_code)
                if not is_valid:
                    await interaction.response.send_message(error_message, ephemeral=True)
                    return

                # Just use the connect code as provided - no # needed for doubles
                connect_code = connect_code.upper()
                self._play_codes[user_id] = connect_code
                
                # Set fixed port to 1
                port = 1
                
                # Convert team color string to number
                team_color_map = {"red": 0, "blue": 1, "green": 2}
                team_color_num = team_color_map.get(team_color.lower(), 0)  # Default to red if not found
                
                # Convert character string to enum if provided
                char_enum = None
                if character:
                    try:
                        char_enum = Character[character]
                        char_display = character.replace('_', ' ').title()
                    except (KeyError, ValueError):
                        char_enum = get_character_from_name(character)
                        char_display = char_enum.name.replace('_', ' ').title()
                else:
                    char_enum = Character.FOX
                    char_display = "Fox"
                
                agent_name = self._get_opponent(user_id)
                resolved_playstyle = self._resolve_playstyle(agent_name, playstyle, char_enum)
                playstyle_display = format_playstyle(playstyle, resolved_playstyle)

                team_colors = {port: team_color_num}
                agents = {port: agent_name}
                playstyles = {port: playstyle_display}
                
                bot_specs = [
                    BotSpec(
                        logical_port=port,
                        team_color=team_color_num,
                        character=char_enum,
                        playstyle=resolved_playstyle,
                        user_json_path=self.dolphin_config.user_json_path3,
                    ),
                ]
                try:
                    self._validate_bot_specs(agent_name, bot_specs)
                except SessionLaunchError as exc:
                    await interaction.response.send_message(str(exc), ephemeral=True)
                    return

                logging.info(
                    "Connecting to %s (%s) with agent %s on port %s with team color %s "
                    "playing %s using playstyle %s",
                    interaction.user.name, connect_code, agent_name, port,
                    team_color, char_display, playstyle_display)
                await interaction.response.defer()
                await interaction.edit_original_response(
                    embed=self._build_session_launch_embed(
                        discord_name=interaction.user.name,
                        connect_code=connect_code,
                        bot_specs=bot_specs,
                        agents=agents,
                        playstyles=playstyles,
                        status="Launching",
                    ))

                try:
                    session = self._start_session(
                        connect_code=connect_code,
                        agent_kwargs=self._get_agent_kwargs(user_id, port, agent_name),
                        bot_specs=bot_specs,
                    )
                except SessionLaunchError as exc:
                    await interaction.edit_original_response(
                        embed=self._build_session_launch_embed(
                            discord_name=interaction.user.name,
                            connect_code=connect_code,
                            bot_specs=bot_specs,
                            agents=agents,
                            playstyles=playstyles,
                            status="Failed",
                            success=False,
                            error=str(exc),
                        ))
                    return

                await interaction.edit_original_response(
                    embed=self._build_session_launch_embed(
                        discord_name=interaction.user.name,
                        connect_code=connect_code,
                        bot_specs=bot_specs,
                        agents=agents,
                        playstyles=playstyles,
                        status="Queueing",
                        success=True,
                    ))
                launch_message = await interaction.original_response()
                
                self._sessions[user_id] = SessionInfo(
                    session=session,
                    start_time=datetime.datetime.now(),
                    discord_name=interaction.user.name,
                    discord_id=user_id,
                    connect_code=connect_code,
                    agents=agents,
                    team_colors=team_colors,
                    playstyles=playstyles,
                    bot_specs=bot_specs,
                    launch_message=launch_message,
                )

        @play_command.autocomplete('character')
        async def play1_character_autocomplete(
            interaction: discord.Interaction,
            current: str,
        ):
            return await self._autocomplete_character_for_port(interaction, current, 1)

        @play_command.autocomplete('playstyle')
        async def play1_playstyle_autocomplete(
            interaction: discord.Interaction,
            current: str,
        ):
            return await self._autocomplete_playstyle_for_port(interaction, current, 1)
        
        @self.tree.command(name="play2", description="Start a game with two AI agents")
        @app_commands.describe(
            connect_code="Your Slippi lobby code",
            character1="Character for first agent",
            team1="Team color for first agent",
            character2="Character for second agent",
            team2="Team color for second agent",
            playstyle1="Optional nametag/playstyle for first agent",
            playstyle2="Optional nametag/playstyle for second agent",
        )
        @app_commands.choices(team1=[
            app_commands.Choice(name="Red", value="red"),
            app_commands.Choice(name="Blue", value="blue"),
            app_commands.Choice(name="Green", value="green"),
        ])
        @app_commands.choices(team2=[
            app_commands.Choice(name="Red", value="red"),
            app_commands.Choice(name="Blue", value="blue"),
            app_commands.Choice(name="Green", value="green"),
        ])
        async def play2_command(
            interaction: discord.Interaction, 
            connect_code: str, 
            character1: str, 
            team1: str,
            character2: str,
            team2: str,
            playstyle1: Optional[str] = None,
            playstyle2: Optional[str] = None,
        ):
            with self.lock:
                user_id = interaction.user.id

                if user_id in self._sessions:
                    await interaction.response.send_message(
                        f'{interaction.user.name}, you are already playing', 
                        ephemeral=True
                    )
                    return

                await self._gc_sessions()

                if len(self._sessions) >= self._max_sessions:
                    await interaction.response.send_message(
                        'Sorry, too many sessions already active.', 
                        ephemeral=True
                    )
                    return

                # Validate the connect code
                is_valid, error_message = self._is_valid_connect_code(connect_code)
                if not is_valid:
                    await interaction.response.send_message(error_message, ephemeral=True)
                    return

                # Just use the connect code as provided - no # needed for doubles
                connect_code = connect_code.upper()
                self._play_codes[user_id] = connect_code
                
                # Convert team color strings to numbers
                team_color_map = {"red": 0, "blue": 1, "green": 2}
                team1_num = team_color_map.get(team1.lower(), 0)  # Default to red if not found
                team2_num = team_color_map.get(team2.lower(), 0)  # Default to red if not found

                # Convert character strings to enums
                char1_enum = None
                if character1:
                    try:
                        char1_enum = Character[character1]
                        char1_display = character1.replace('_', ' ').title()
                    except (KeyError, ValueError):
                        char1_enum = get_character_from_name(character1)
                        char1_display = char1_enum.name.replace('_', ' ').title()
                else:
                    char1_enum = Character.FOX
                    char1_display = "Fox"
                    
                char2_enum = None
                if character2:
                    try:
                        char2_enum = Character[character2]
                        char2_display = character2.replace('_', ' ').title()
                    except (KeyError, ValueError):
                        char2_enum = get_character_from_name(character2)
                        char2_display = char2_enum.name.replace('_', ' ').title()
                else:
                    char2_enum = Character.FOX
                    char2_display = "Fox"
                
                # Get agent names (using port index only for selecting different agents)
                agent1_name = self._get_opponent(user_id, 1)
                agent2_name = self._get_opponent(user_id, 2)
                resolved_playstyle1 = self._resolve_playstyle(agent1_name, playstyle1, char1_enum)
                resolved_playstyle2 = self._resolve_playstyle(agent2_name, playstyle2, char2_enum)
                playstyle1_display = format_playstyle(playstyle1, resolved_playstyle1)
                playstyle2_display = format_playstyle(playstyle2, resolved_playstyle2)
                team_colors = {1: team1_num, 2: team2_num}
                agents = {1: agent1_name, 2: agent2_name}
                playstyles = {
                    1: playstyle1_display,
                    2: playstyle2_display,
                }

                bot_specs = [
                    BotSpec(
                        logical_port=1,
                        team_color=team1_num,
                        character=char1_enum,
                        playstyle=resolved_playstyle1,
                    ),
                    BotSpec(
                        logical_port=2,
                        team_color=team2_num,
                        character=char2_enum,
                        playstyle=resolved_playstyle2,
                        user_json_path=self.dolphin_config.user_json_path2,
                    ),
                ]
                try:
                    self._validate_bot_specs(agent1_name, bot_specs)
                except SessionLaunchError as exc:
                    await interaction.response.send_message(str(exc), ephemeral=True)
                    return

                logging.info(
                    "Connecting to %s (%s) with: agent %s team %s char %s playstyle %s; "
                    "agent %s team %s char %s playstyle %s",
                    interaction.user.name, connect_code,
                    agent1_name, team1, char1_display, playstyle1_display,
                    agent2_name, team2, char2_display, playstyle2_display)
                await interaction.response.defer()
                await interaction.edit_original_response(
                    embed=self._build_session_launch_embed(
                        discord_name=interaction.user.name,
                        connect_code=connect_code,
                        bot_specs=bot_specs,
                        agents=agents,
                        playstyles=playstyles,
                        status="Launching",
                    ))

                try:
                    session = self._start_session(
                        connect_code=connect_code,
                        agent_kwargs=self._get_agent_kwargs(user_id, 1, agent1_name),
                        bot_specs=bot_specs,
                    )
                except SessionLaunchError as exc:
                    await interaction.edit_original_response(
                        embed=self._build_session_launch_embed(
                            discord_name=interaction.user.name,
                            connect_code=connect_code,
                            bot_specs=bot_specs,
                            agents=agents,
                            playstyles=playstyles,
                            status="Failed",
                            success=False,
                            error=str(exc),
                        ))
                    return

                await interaction.edit_original_response(
                    embed=self._build_session_launch_embed(
                        discord_name=interaction.user.name,
                        connect_code=connect_code,
                        bot_specs=bot_specs,
                        agents=agents,
                        playstyles=playstyles,
                        status="Queueing",
                        success=True,
                    ))
                launch_message = await interaction.original_response()
                
                self._sessions[user_id] = SessionInfo(
                    session=session,
                    start_time=datetime.datetime.now(),
                    discord_name=interaction.user.name,
                    discord_id=user_id,
                    connect_code=connect_code,
                    agents=agents,
                    team_colors=team_colors,
                    playstyles=playstyles,
                    bot_specs=bot_specs,
                    launch_message=launch_message,
                )

        @play2_command.autocomplete('character1')
        async def play2_character1_autocomplete(
            interaction: discord.Interaction,
            current: str,
        ):
            return await self._autocomplete_character_for_port(interaction, current, 1)

        @play2_command.autocomplete('character2')
        async def play2_character2_autocomplete(
            interaction: discord.Interaction,
            current: str,
        ):
            return await self._autocomplete_character_for_port(interaction, current, 2)

        @play2_command.autocomplete('playstyle1')
        async def play2_playstyle1_autocomplete(
            interaction: discord.Interaction,
            current: str,
        ):
            return await self._autocomplete_playstyle_for_port(interaction, current, 1)

        @play2_command.autocomplete('playstyle2')
        async def play2_playstyle2_autocomplete(
            interaction: discord.Interaction,
            current: str,
        ):
            return await self._autocomplete_playstyle_for_port(interaction, current, 2)
        
        @self.tree.command(name="play3", description="Start a game with three AI agents")
        @app_commands.describe(
            connect_code="Your Slippi lobby code",
            character1="Character for first agent",
            team1="Team color for first agent",
            character2="Character for second agent",
            team2="Team color for second agent",
            character3="Character for third agent",
            team3="Team color for third agent",
            playstyle1="Optional nametag/playstyle for first agent",
            playstyle2="Optional nametag/playstyle for second agent",
            playstyle3="Optional nametag/playstyle for third agent",
        )
        @app_commands.choices(team1=[
            app_commands.Choice(name="Red", value="red"),
            app_commands.Choice(name="Blue", value="blue"),
            app_commands.Choice(name="Green", value="green"),
        ])
        @app_commands.choices(team2=[
            app_commands.Choice(name="Red", value="red"),
            app_commands.Choice(name="Blue", value="blue"),
            app_commands.Choice(name="Green", value="green"),
        ])
        @app_commands.choices(team3=[
            app_commands.Choice(name="Red", value="red"),
            app_commands.Choice(name="Blue", value="blue"),
            app_commands.Choice(name="Green", value="green"),
        ])
        async def play3_command(
            interaction: discord.Interaction, 
            connect_code: str,
            character1: str,
            team1: str,
            character2: str,
            team2: str,
            character3: str,
            team3: str,
            playstyle1: Optional[str] = None,
            playstyle2: Optional[str] = None,
            playstyle3: Optional[str] = None,
        ):
            with self.lock:
                user_id = interaction.user.id

                if user_id in self._sessions:
                    await interaction.response.send_message(
                        f'{interaction.user.name}, you are already playing', 
                        ephemeral=True
                    )
                    return

                await self._gc_sessions()

                if len(self._sessions) >= self._max_sessions:
                    await interaction.response.send_message(
                        'Sorry, too many sessions already active.', 
                        ephemeral=True
                    )
                    return

                # Validate the connect code
                is_valid, error_message = self._is_valid_connect_code(connect_code)
                if not is_valid:
                    await interaction.response.send_message(error_message, ephemeral=True)
                    return

                # Just use the connect code as provided - no # needed for doubles
                connect_code = connect_code.upper()
                self._play_codes[user_id] = connect_code
                
                # Convert team color strings to numbers
                team_color_map = {"red": 0, "blue": 1, "green": 2}
                team1_num = team_color_map.get(team1.lower(), 0)  # Default to red if not found
                team2_num = team_color_map.get(team2.lower(), 0)  # Default to red if not found
                team3_num = team_color_map.get(team3.lower(), 0)  # Default to red if not found

                # Convert character strings to enums
                char1_enum = None
                if character1:
                    try:
                        char1_enum = Character[character1]
                        char1_display = character1.replace('_', ' ').title()
                    except (KeyError, ValueError):
                        char1_enum = get_character_from_name(character1)
                        char1_display = char1_enum.name.replace('_', ' ').title()
                else:
                    char1_enum = Character.FOX
                    char1_display = "Fox"
                    
                char2_enum = None
                if character2:
                    try:
                        char2_enum = Character[character2]
                        char2_display = character2.replace('_', ' ').title()
                    except (KeyError, ValueError):
                        char2_enum = get_character_from_name(character2)
                        char2_display = char2_enum.name.replace('_', ' ').title()
                else:
                    char2_enum = Character.FOX
                    char2_display = "Fox"
                
                char3_enum = None
                if character3:
                    try:
                        char3_enum = Character[character3]
                        char3_display = character3.replace('_', ' ').title()
                    except (KeyError, ValueError):
                        char3_enum = get_character_from_name(character3)
                        char3_display = char3_enum.name.replace('_', ' ').title()
                else:
                    char3_enum = Character.FOX
                    char3_display = "Fox"
                
                # Get agent names (using port index only for selecting different agents)
                agent1_name = self._get_opponent(user_id, 1)
                agent2_name = self._get_opponent(user_id, 2)
                agent3_name = self._get_opponent(user_id, 3)
                resolved_playstyle1 = self._resolve_playstyle(agent1_name, playstyle1, char1_enum)
                resolved_playstyle2 = self._resolve_playstyle(agent2_name, playstyle2, char2_enum)
                resolved_playstyle3 = self._resolve_playstyle(agent3_name, playstyle3, char3_enum)
                playstyle1_display = format_playstyle(playstyle1, resolved_playstyle1)
                playstyle2_display = format_playstyle(playstyle2, resolved_playstyle2)
                playstyle3_display = format_playstyle(playstyle3, resolved_playstyle3)
                team_colors = {1: team1_num, 2: team2_num, 3: team3_num}
                agents = {1: agent1_name, 2: agent2_name, 3: agent3_name}
                playstyles = {
                    1: playstyle1_display,
                    2: playstyle2_display,
                    3: playstyle3_display,
                }

                bot_specs = [
                    BotSpec(
                        logical_port=1,
                        team_color=team1_num,
                        character=char1_enum,
                        playstyle=resolved_playstyle1,
                    ),
                    BotSpec(
                        logical_port=2,
                        team_color=team2_num,
                        character=char2_enum,
                        playstyle=resolved_playstyle2,
                        user_json_path=self.dolphin_config.user_json_path2,
                    ),
                    BotSpec(
                        logical_port=3,
                        team_color=team3_num,
                        character=char3_enum,
                        playstyle=resolved_playstyle3,
                        user_json_path=self.dolphin_config.user_json_path3,
                    ),
                ]
                try:
                    self._validate_bot_specs(agent1_name, bot_specs)
                except SessionLaunchError as exc:
                    await interaction.response.send_message(str(exc), ephemeral=True)
                    return

                logging.info(
                    "Connecting to %s (%s) with: agent %s team %s char %s playstyle %s; "
                    "agent %s team %s char %s playstyle %s; "
                    "agent %s team %s char %s playstyle %s",
                    interaction.user.name, connect_code,
                    agent1_name, team1, char1_display, playstyle1_display,
                    agent2_name, team2, char2_display, playstyle2_display,
                    agent3_name, team3, char3_display, playstyle3_display)
                await interaction.response.defer()
                await interaction.edit_original_response(
                    embed=self._build_session_launch_embed(
                        discord_name=interaction.user.name,
                        connect_code=connect_code,
                        bot_specs=bot_specs,
                        agents=agents,
                        playstyles=playstyles,
                        status="Launching",
                    ))

                try:
                    session = self._start_session(
                        connect_code=connect_code,
                        agent_kwargs=self._get_agent_kwargs(user_id, 1, agent1_name),
                        bot_specs=bot_specs,
                    )
                except SessionLaunchError as exc:
                    await interaction.edit_original_response(
                        embed=self._build_session_launch_embed(
                            discord_name=interaction.user.name,
                            connect_code=connect_code,
                            bot_specs=bot_specs,
                            agents=agents,
                            playstyles=playstyles,
                            status="Failed",
                            success=False,
                            error=str(exc),
                        ))
                    return

                await interaction.edit_original_response(
                    embed=self._build_session_launch_embed(
                        discord_name=interaction.user.name,
                        connect_code=connect_code,
                        bot_specs=bot_specs,
                        agents=agents,
                        playstyles=playstyles,
                        status="Queueing",
                        success=True,
                    ))
                launch_message = await interaction.original_response()
                
                self._sessions[user_id] = SessionInfo(
                    session=session,
                    start_time=datetime.datetime.now(),
                    discord_name=interaction.user.name,
                    discord_id=user_id,
                    connect_code=connect_code,
                    agents=agents,
                    team_colors=team_colors,
                    playstyles=playstyles,
                    bot_specs=bot_specs,
                    launch_message=launch_message,
                )

        @play3_command.autocomplete('character1')
        async def play3_character1_autocomplete(
            interaction: discord.Interaction,
            current: str,
        ):
            return await self._autocomplete_character_for_port(interaction, current, 1)

        @play3_command.autocomplete('character2')
        async def play3_character2_autocomplete(
            interaction: discord.Interaction,
            current: str,
        ):
            return await self._autocomplete_character_for_port(interaction, current, 2)

        @play3_command.autocomplete('character3')
        async def play3_character3_autocomplete(
            interaction: discord.Interaction,
            current: str,
        ):
            return await self._autocomplete_character_for_port(interaction, current, 3)

        @play3_command.autocomplete('playstyle1')
        async def play3_playstyle1_autocomplete(
            interaction: discord.Interaction,
            current: str,
        ):
            return await self._autocomplete_playstyle_for_port(interaction, current, 1)

        @play3_command.autocomplete('playstyle2')
        async def play3_playstyle2_autocomplete(
            interaction: discord.Interaction,
            current: str,
        ):
            return await self._autocomplete_playstyle_for_port(interaction, current, 2)

        @play3_command.autocomplete('playstyle3')
        async def play3_playstyle3_autocomplete(
            interaction: discord.Interaction,
            current: str,
        ):
            return await self._autocomplete_playstyle_for_port(interaction, current, 3)
        
        # Status command
        @self.tree.command(name="status", description="Show current bot status and active sessions")
        async def status_command(interaction: discord.Interaction):
            with self.lock:
                if not self._sessions:
                    await interaction.response.send_message('No active sessions.')
                    return

                embed = discord.Embed(
                    title="Active Sessions",
                    color=discord.Color.blue()
                )

                now = datetime.datetime.now()
                for user_id, session_info in self._sessions.items():
                    timedelta = format_td(now - session_info.start_time)
                    # Get the status using remote call
                    status = session_info.session.status()
                    menu_frames = status['num_menu_frames']
                    menu_time = format_td(datetime.timedelta(seconds=menu_frames / 60))
                    model_text = status.get('current_agent') or 'unknown'
                    if status.get('pending_agent'):
                        model_text += f" (switching to {status['pending_agent']} next game)"
                    
                    agents_info = []
                    for port, agent in session_info.agents.items():
                        team = session_info.team_colors.get(port, 0)
                        team_name = ["Red", "Blue", "Green"][team]
                        playstyle = session_info.playstyles.get(port, "default")
                        agents_info.append(
                            f"Port {port}: {agent} (Team {team_name}, Playstyle {playstyle})")
                    
                    agents_text = "\n".join(agents_info)
                    
                    embed.add_field(
                        name=f"Playing against {session_info.discord_name}",
                        value=f"Connect code: {session_info.connect_code}\n"
                              f"Duration: {timedelta}\n"
                              f"Menu time: {menu_time}\n"
                              f"Model: {model_text}\n"
                              f"Agents:\n{agents_text}",
                        inline=False
                    )

                await interaction.response.send_message(embed=embed)
        
        # GC command (admin only)
        @self.tree.command(name="gc", description="Clean up idle sessions (Admin only)")
        async def gc_command(interaction: discord.Interaction):
            if not self.is_admin(interaction):
                await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
                return
                
            infos = await self._gc_sessions()
            names = [info.discord_name for info in infos]
            names_str = ", ".join(names) if names else "none"
            await interaction.response.send_message(f"Stopped idle sessions: {names_str}")

        @self.tree.command(name="stop_session", description="Stop a specific session by Discord name (Admin only)")
        @app_commands.describe(user="Discord username for the session to stop")
        async def stop_session_command(interaction: discord.Interaction, user: str):
            if not self.is_admin(interaction):
                await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
                return

            with self.lock:
                matches = [
                    info for info in self._sessions.values()
                    if info.discord_name.lower() == user.lower()
                ]

            if not matches:
                await interaction.response.send_message(
                    f'No active session found for "{user}".',
                    ephemeral=True,
                )
                return

            self._stop_sessions(matches)
            names = ", ".join(info.discord_name for info in matches)
            await interaction.response.send_message(f"Stopped session(s): {names}")
            
        # Sync commands globally
        await self.tree.sync()

    def is_admin(self, interaction: discord.Interaction) -> bool:
        """Check if the user has admin permissions"""
        # Simply check if the username is 'enzyme_'
        return interaction.user.name == 'enzyme_'

    # Update the run method
    def run(self, token):
        """Run the bot with the given token."""
        super().run(token)

    # Keep the rest of the methods unchanged
    def _reload_models(self):
        self._models: Dict[str, dict] = {}
        agents = os.listdir(self._models_path)

        for agent in agents:
            path = os.path.join(self._models_path, agent)
            state = eval_lib.load_state(path=path)
            state = {k: state[k] for k in ['step', 'config', 'rl_config'] if k in state}
            self._models[agent] = state

    def _get_opponent(self, user_id: int, port: int = 1) -> str:
        """Get the agent name for the specified user and port"""
        if user_id in self._requested_agents:
            return self._requested_agents[user_id]
        return self._default_agent_name

    def _get_agent_kwargs(self, user_id: int, port: int, agent_name: str) -> dict:
        """Get agent kwargs for the specified agent"""
        agent_kwargs = self.agent_kwargs.copy()
        agent_kwargs['path'] = os.path.join(self._models_path, agent_name)
        return agent_kwargs

    def _get_effective_agent_name(self, user_id: int, port: int) -> str:
        return self._get_opponent(user_id, port)

    def _get_state_for_port(self, user_id: int, port: int) -> dict:
        agent_name = self._get_effective_agent_name(user_id, port)
        return self._models[agent_name]

    def _get_autocomplete_character(self, interaction: discord.Interaction, port: int) -> Optional[Character]:
        namespace = getattr(interaction, 'namespace', None)
        if namespace is None:
            return None

        names = [f'character{port}']
        if port == 1:
            names.append('character')
        for name in names:
            character = get_optional_character_from_name(getattr(namespace, name, None))
            if character is not None:
                return character
        return None

    def _character_default_playstyle(
        self,
        agent_name: str,
        character: Optional[Character],
        state: dict,
    ) -> Optional[str]:
        return get_character_default_playstyle(
            agent_name,
            character,
            self._default_playstyles_by_character,
            state,
        )

    async def _autocomplete_playstyle_for_port(
        self,
        interaction: discord.Interaction,
        current: str,
        port: int,
    ) -> list[app_commands.Choice[str]]:
        user_id = interaction.user.id
        agent_name = self._get_effective_agent_name(user_id, port)
        state = self._models[agent_name]
        character = self._get_autocomplete_character(interaction, port)
        default_playstyle = self._character_default_playstyle(agent_name, character, state)
        return get_playstyle_autocomplete_choices(state, current, default_playstyle)

    async def _autocomplete_character_for_port(
        self,
        interaction: discord.Interaction,
        current: str,
        port: int,
    ) -> list[app_commands.Choice[str]]:
        state = self._get_state_for_port(interaction.user.id, port)
        return get_character_autocomplete_choices(state, current)

    def _resolve_playstyle(
        self,
        agent_name: str,
        playstyle: Optional[str],
        character: Optional[Character] = None,
    ) -> str:
        state = self._models[agent_name]
        default_name = (
            self._character_default_playstyle(agent_name, character, state) or
            self.agent_kwargs.get('name')
        )
        return resolve_playstyle_for_state(
            playstyle,
            default_name,
            state,
        )

    def _validate_bot_specs(self, agent_name: str, bot_specs: List[BotSpec]):
        state = self._models[agent_name]
        allowed_characters = get_allowed_characters_for_state(state)
        if allowed_characters is None:
            return

        invalid_specs = [
            spec for spec in bot_specs
            if spec.character is not None and spec.character not in allowed_characters
        ]
        if not invalid_specs:
            return

        invalid = ', '.join(
            f"bot {spec.logical_port}: {format_character(spec.character)}"
            for spec in invalid_specs
            if spec.character is not None
        )
        valid = format_character_list(allowed_characters)
        raise SessionLaunchError(
            f"Can't launch {agent_name} with {invalid}. "
            f"Valid characters for this model: {valid}."
        )

    def _build_session_launch_embed(
        self,
        discord_name: str,
        connect_code: str,
        bot_specs: List[BotSpec],
        agents: Dict[int, str],
        playstyles: Dict[int, str],
        status: str,
        success: Optional[bool] = None,
        error: Optional[str] = None,
        title_override: Optional[str] = None,
    ) -> discord.Embed:
        title = title_override or "Session Launched"
        if success is None:
            color = discord.Color.orange()
        elif success:
            color = discord.Color.green()
        else:
            color = discord.Color.red()

        embed = discord.Embed(
            title=title,
            description=f"Opponent: `{discord_name}`\nConnect code: `{connect_code}`",
            color=color,
        )
        embed.add_field(name="Status", value=status, inline=False)

        for spec in bot_specs:
            team_marker = format_team_marker(spec.team_color)
            character = spec.character or Character.FOX
            playstyle = playstyles.get(spec.logical_port, "default")
            agent_name = agents.get(spec.logical_port, "unknown")
            embed.add_field(
                name=f"Agent {spec.logical_port} · `{agent_name}`",
                value=f"{team_marker} - {format_character(character)} - ({playstyle})",
                inline=False,
            )

        if error:
            embed.add_field(name="Error", value=error[:1024], inline=False)

        return embed

    def _start_session(
        self,
        connect_code: str,
        agent_kwargs: dict,
        bot_specs: List[BotSpec],
    ) -> DoublesSession:
        """Start one fused doubles session actor for all local bots."""
        use_local_gpu_session = (
            self._gpu_server is not None and
            self._gpu_model_basename is not None and
            os.path.basename(agent_kwargs.get('path', '')) == self._gpu_model_basename
        )
        if use_local_gpu_session:
            session = LocalSessionHandle(DoublesSession(
                self.dolphin_config,
                agent_kwargs,
                connect_code,
                bot_specs,
                self._gpu_server,
                self._gpu_model_basename,
                disable_process_gpus=False,
            ))
        else:
            session = RemoteSessionHandle(RemoteDoublesSession.remote(
                self.dolphin_config,
                agent_kwargs,
                connect_code,
                bot_specs,
                None,
                None,
            ))
        try:
            session.start()
        except Exception as exc:
            try:
                session.kill()
            except Exception:
                logging.exception("Failed to clean up Discord session after launch failure")
            raise SessionLaunchError(
                f"Failed to launch the bot session: {format_launch_exception(exc)}"
            ) from exc
        return session

    def _stop_sessions(self, infos: List[SessionInfo]):
        """Stop the specified sessions"""
        with self.lock:
            for info in infos:
                info.session.stop()

            for info in infos:
                if info.discord_id in self._sessions:
                    del self._sessions[info.discord_id]

    async def _gc_sessions(self) -> List[SessionInfo]:
        """Stop sessions that have been in the menu for too long."""
        with self.lock:
            to_gc: List[SessionInfo] = []
            for info in self._sessions.values():
                # Get the status using remote call and ray.get
                status = info.session.status()
                menu_minutes = status['num_menu_frames'] / (60 * 60)
                stale_activity = status.get('seconds_since_activity', 0) > STALL_TIMEOUT_SECONDS
                stale_frame = (
                    status.get('has_entered_game') and
                    not status.get('in_menu', True) and
                    status.get('seconds_since_frame_advance', 0) > STALL_TIMEOUT_SECONDS
                )
                if (
                    not status['is_alive'] or
                    menu_minutes > self._menu_timeout or
                    stale_activity or
                    stale_frame
                ):
                    to_gc.append(info)

            self._stop_sessions(to_gc)
            if to_gc:
                names = ", ".join([info.discord_name for info in to_gc])
                logging.info(f'GCed sessions: {names}')
            return to_gc

    async def _update_playing_embeds(self):
        with self.lock:
            infos = [info for info in self._sessions.values() if not info.playing_announced]

        for info in infos:
            status = info.session.status()
            if not status.get('is_alive') or not status.get('has_entered_game'):
                continue
            try:
                await info.launch_message.edit(
                    embed=self._build_session_launch_embed(
                        discord_name=info.discord_name,
                        connect_code=info.connect_code,
                        bot_specs=info.bot_specs,
                        agents=info.agents,
                        playstyles=info.playstyles,
                        status="Playing",
                        success=True,
                        title_override="Playing",
                    ))
                info.playing_announced = True
            except Exception:
                logging.exception(
                    "Failed to update Discord session embed to Playing for %s",
                    info.discord_name)

    @tasks.loop(seconds=5)
    async def _do_chores(self):
        await self._update_playing_embeds()
        await self._gc_sessions()

    def shutdown(self):
        with self.lock:
            self._stop_sessions(list(self._sessions.values()))
            self._do_chores.cancel()

    async def close(self):
        self.shutdown()
        await super().close()

    def _is_valid_connect_code(self, connect_code: str) -> Tuple[bool, str]:
        """
        Validate the connect code.
        Returns a tuple of (is_valid, error_message).
        If the code is valid, error_message will be empty.
        """
        connect_code = connect_code.upper()
        
        # Check if the connect code starts with EC or WC
        if connect_code.startswith(('EC', 'WC')):
            return False, f"Connect codes starting with EC or WC are not allowed. Please use a different connect code."
            
        return True, ""

# Modify the main function to use the bot's run method
def main(_):
    ray.init()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    agent_kwargs = AGENT.value
    if not agent_kwargs['path']:
        raise ValueError('Must provide agent path.')
    default_playstyles_by_character = parse_default_playstyle_by_character(
        DEFAULT_PLAYSTYLE_BY_CHARACTER.value)

    gpu_server = None
    gpu_model_basename = None
    gpu_model_path = resolve_gpu_model_path(MODELS_PATH.value, GPU_MODEL.value)
    if not gpu_model_path:
        eval_lib.disable_gpus()
    if gpu_model_path:
        configure_tensorflow_gpu_limit(GPU_MEMORY_LIMIT_MB.value)
        gpu_model_basename = os.path.basename(gpu_model_path)
        logging.info(
            'Starting shared Discord GPU inference server for %s',
            gpu_model_basename)
        gpu_server = LocalDiscordGpuInferenceServer(
            model_path=gpu_model_path,
            console_delay=DOLPHIN.value['online_delay'],
            max_sessions=MAX_SESSIONS.value,
            max_local_bots=3,
            batch_window_ms=GPU_MICROBATCH_MS.value,
            compile=agent_kwargs.get('compile', True),
            jit_compile=agent_kwargs.get('jit_compile', False),
            sample_temperature=agent_kwargs.get('sample_temperature', 1.0),
            batch_steps=agent_kwargs.get('batch_steps', 0),
        )
        gpu_server.warmup()

    bot = DiscordBot(
        token=BOT_TOKEN.value,
        prefix=COMMAND_PREFIX.value,
        admin_role=ADMIN_ROLE.value,
        models_path=MODELS_PATH.value,
        dolphin_config=flag_utils.dataclass_from_dict(
            dolphin_lib.DolphinConfig, DOLPHIN.value),
        agent_kwargs=agent_kwargs,
        max_sessions=MAX_SESSIONS.value,
        menu_timeout=MENU_TIMEOUT.value,
        gpu_server=gpu_server,
        gpu_model_basename=gpu_model_basename,
        default_playstyles_by_character=default_playstyles_by_character,
    )

    try:
        bot.run(bot.token)
    finally:
        bot.shutdown()
        if gpu_server is not None:
            try:
                gpu_server.close()
            except Exception:
                logging.exception('Failed to close Discord GPU inference server')

if __name__ == '__main__':
    app.run(main)
