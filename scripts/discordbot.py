"""Bot that runs on Discord and lets people play against phillip 2 in doubles matches."""

import dataclasses
import datetime
import json
import logging
import os
import queue
import threading
import time
from typing import Optional, Dict, List, Tuple, Any

from absl import app, flags
import fancyflags as ff
import discord
from discord import app_commands
from discord.ext import commands, tasks
import numpy as np
import portpicker
import ray

from slippi_ai import data as data_lib
from slippi_ai import flag_utils, eval_lib, nametags, utils
from slippi_ai import dolphin as dolphin_lib
from slippi_ai.controller_lib import send_controller
from slippi_db.parse_libmelee import get_game
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

# Session management settings
MENU_TIMEOUT = flags.DEFINE_float(
    'menu_timeout', 3, 'Minutes before timing out a session in menu')
MAX_SESSIONS = flags.DEFINE_integer(
    'max_sessions', 4, 'Maximum number of concurrent sessions')
STALL_TIMEOUT_SECONDS = 120.0

SUPPORTED_PLAYSTYLES = (
    'Master Player',
    'Ralph',
    'Darkatma',
    'Dragunov',
    'Tempo',
    'xRunRiot',
)


class SessionLaunchError(Exception):
    """User-facing session launch failure."""


def get_playstyle_choices():
    return [
        app_commands.Choice(name=name, value=name)
        for name in SUPPORTED_PLAYSTYLES
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

    normalized_to_supported = {
        nametags.normalize_name(name): name for name in supported_names
    }

    for candidate in (requested_playstyle, default_name, nametags.DEFAULT_NAME):
        if not candidate:
            continue
        normalized = nametags.normalize_name(candidate)
        if normalized in normalized_to_supported:
            return normalized_to_supported[normalized]

    fallback = supported_names[0]
    if requested_playstyle:
        logging.warning(
            'Requested playstyle %s is not supported by this model; using %s.',
            requested_playstyle, fallback)
    return fallback


def format_playstyle(requested_playstyle: Optional[str], resolved_playstyle: str) -> str:
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


def format_character_list(characters: list[Character]) -> str:
    return ', '.join(format_character(character) for character in characters)


def format_launch_exception(exc: Exception) -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    message = lines[-1] if lines else repr(exc)
    return message[:1500]


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
    ):
        eval_lib.disable_gpus()
        self.dolphin_config = dolphin_config
        self.agent_kwargs = agent_kwargs
        self.connect_code = connect_code
        self.bot_specs = list(bot_specs)

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
        self._dead_frames: Dict[int, int] = {spec.logical_port: 0 for spec in self.bot_specs}
        self._pressed_start: Dict[int, bool] = {spec.logical_port: False for spec in self.bot_specs}
        self._has_entered_game = False
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
            'seconds_since_activity': now - self._last_activity_time,
            'seconds_since_frame_advance': now - self._last_frame_advance_time,
        }

    def set_agent(self, agent_kwargs: dict) -> dict:
        """Request a model switch at the next game boundary."""
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
                self._start_dolphins()
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

        state = eval_lib.load_state(path=path, tag=tag)
        default_name = agent_kwargs.pop('name', None)
        agent_kwargs['name'] = [
            resolve_playstyle_for_state(spec.playstyle, default_name, state)
            for spec in self.bot_specs
        ]
        logging.info(
            "Discord fused session playstyles: %s",
            {
                spec.logical_port: name
                for spec, name in zip(self.bot_specs, agent_kwargs['name'])
            },
        )
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
        return state, agent, self._agent_label(source_agent_kwargs)

    def _set_state_and_agent(self, source_agent_kwargs: dict):
        state, agent, label = self._build_state_and_agent(source_agent_kwargs)
        self._state = state
        self._agent = agent
        self._current_agent_label = label
        self.agent_kwargs = source_agent_kwargs.copy()

    def _apply_pending_agent_if_needed(self):
        with self._lock:
            pending_agent_kwargs = self._pending_agent_kwargs
            pending_agent_label = self._pending_agent_label
            if pending_agent_kwargs is None:
                return

        logging.info(
            "Applying pending Discord fused session model switch to %s",
            pending_agent_label)
        old_agent = self._agent
        try:
            state, agent, label = self._build_state_and_agent(pending_agent_kwargs)
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
        config.slippi_port = portpicker.pick_unused_port()
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
            finalized_buffer: Optional[FinalizedDelayBuffer] = None
            using_finalized_metadata = False
            last_live_frame: Optional[int] = None
            last_logged_gap_frame: Optional[int] = None
            last_logged_missing_frame: Optional[int] = None
            last_logged_lag_bucket: Optional[int] = None

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
                        if finalized_buffer is not None:
                            finalized_buffer.clear()
                        publish_gamestate(gamestate)
                        continue

                    finalized_frame = getattr(gamestate, 'finalized_frame', None)
                    if finalized_frame is None:
                        publish_gamestate(gamestate)
                        continue
                    frame = int(gamestate.frame)
                    if last_live_frame is not None:
                        if frame <= last_live_frame:
                            logging.warning(
                                "Discord bot live frame rollback/correction "
                                "logical_port=%s from=%s to=%s finalized=%s",
                                logical_port, last_live_frame, frame, finalized_frame)
                        elif frame > last_live_frame + 1:
                            logging.warning(
                                "Discord bot live frame gap logical_port=%s "
                                "from=%s to=%s skipped=%s finalized=%s",
                                logical_port, last_live_frame, frame,
                                frame - last_live_frame - 1, finalized_frame)
                    last_live_frame = frame

                    if not using_finalized_metadata:
                        logging.info(
                            "Using finalized Slippstream frames for logical_port=%s",
                            logical_port)
                        using_finalized_metadata = True

                    observation_delay = int(getattr(self._agent, 'delay', 0))
                    if (
                        finalized_buffer is None or
                        finalized_buffer.delay != observation_delay
                    ):
                        finalized_buffer = FinalizedDelayBuffer(observation_delay)

                    lag = frame - int(finalized_frame)
                    lag_bucket = lag // 5
                    if lag >= 5 and lag_bucket != last_logged_lag_bucket:
                        logging.info(
                            "Discord bot finalized lag logical_port=%s "
                            "live=%s finalized=%s lag=%s delay=%s",
                            logical_port, frame, finalized_frame, lag,
                            observation_delay)
                        last_logged_lag_bucket = lag_bucket

                    for published in finalized_buffer.push(gamestate):
                        if (
                            finalized_buffer.last_gap_frame is not None and
                            finalized_buffer.last_gap_frame != last_logged_gap_frame
                        ):
                            logging.warning(
                                "Discord bot delayed-finalized source gap "
                                "logical_port=%s after_frame=%s next_published=%s",
                                logical_port, finalized_buffer.last_gap_frame,
                                published.frame)
                            last_logged_gap_frame = finalized_buffer.last_gap_frame
                        publish_gamestate(published)

                    next_frame = finalized_buffer._next_frame
                    if (
                        next_frame is not None and
                        next_frame <= min(frame - observation_delay, int(finalized_frame)) and
                        next_frame not in finalized_buffer._frames and
                        next_frame != last_logged_missing_frame
                    ):
                        logging.warning(
                            "Discord bot delayed-finalized missing frame "
                            "logical_port=%s missing=%s live=%s finalized=%s "
                            "delay=%s",
                            logical_port, next_frame, frame, finalized_frame,
                            observation_delay)
                        last_logged_missing_frame = next_frame
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
                    self._num_menu_frames += 1
                    for event in advance_events.values():
                        event.set()
                    continue

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
                all_delayed_finalized = all(
                    gs.custom.get('discordbot_delayed_finalized', False)
                    for gs in current.values())
                agent_step = (
                    self._agent.step_undelayed
                    if all_delayed_finalized
                    else self._agent.step
                )
                sample_outputs = agent_step(
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
            logging.warning(
                "Discord bot could not identify teammate yet "
                "logical_port=%s local_player_port=%s players=%s",
                logical_port, my_port, sorted(gamestate.players))

        known_order = [int(my_port)]
        if teammate_port is not None and teammate_port not in known_order:
            known_order.append(teammate_port)
        player_order = tuple(known_order)
        player_order += tuple(p for p in (1, 2, 3, 4) if p not in player_order)
        self._player_orders[logical_port] = player_order
        self._teammate_ports[logical_port] = teammate_port
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

def get_valid_character_choices():
    """Return a list of character choices for the Discord API."""
    choices = []
    
    # List of playable characters - excluding certain characters
    playable_characters = [
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
    
    for char, display_name in playable_characters:
        choices.append(app_commands.Choice(name=display_name, value=char.name))
    
    return choices

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
                    result = ray.get(session_info.session.set_agent.remote(
                        self._get_agent_kwargs(user_id, 1, agent_name)))
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
        @app_commands.choices(character=get_valid_character_choices())
        @app_commands.choices(playstyle=get_playstyle_choices())
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
                resolved_playstyle = self._resolve_playstyle(agent_name, playstyle)
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
        @app_commands.choices(character1=get_valid_character_choices())
        @app_commands.choices(character2=get_valid_character_choices())
        @app_commands.choices(playstyle1=get_playstyle_choices())
        @app_commands.choices(playstyle2=get_playstyle_choices())
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
                resolved_playstyle1 = self._resolve_playstyle(agent1_name, playstyle1)
                resolved_playstyle2 = self._resolve_playstyle(agent1_name, playstyle2)
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
        @app_commands.choices(character1=get_valid_character_choices())
        @app_commands.choices(character2=get_valid_character_choices())
        @app_commands.choices(character3=get_valid_character_choices())
        @app_commands.choices(playstyle1=get_playstyle_choices())
        @app_commands.choices(playstyle2=get_playstyle_choices())
        @app_commands.choices(playstyle3=get_playstyle_choices())
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
                resolved_playstyle1 = self._resolve_playstyle(agent1_name, playstyle1)
                resolved_playstyle2 = self._resolve_playstyle(agent1_name, playstyle2)
                resolved_playstyle3 = self._resolve_playstyle(agent1_name, playstyle3)
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
                    status = ray.get(session_info.session.status.remote())
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

    def _resolve_playstyle(self, agent_name: str, playstyle: Optional[str]) -> str:
        state = self._models[agent_name]
        return resolve_playstyle_for_state(
            playstyle,
            self.agent_kwargs.get('name'),
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
        session = RemoteDoublesSession.remote(
            self.dolphin_config,
            agent_kwargs,
            connect_code,
            bot_specs,
        )
        # Call the remote method and get the result
        try:
            ray.get(session.start.remote())
        except Exception as exc:
            try:
                ray.kill(session, no_restart=True)
            except Exception:
                logging.exception("Failed to clean up Discord session after launch failure")
            raise SessionLaunchError(
                f"Failed to launch the bot session: {format_launch_exception(exc)}"
            ) from exc
        return session

    def _stop_sessions(self, infos: List[SessionInfo]):
        """Stop the specified sessions"""
        with self.lock:
            # Create list of tasks and wait for them to complete
            stop_tasks = [info.session.stop.remote() for info in infos]
            if stop_tasks:
                ray.get(stop_tasks)

            for info in infos:
                if info.discord_id in self._sessions:
                    del self._sessions[info.discord_id]

    async def _gc_sessions(self) -> List[SessionInfo]:
        """Stop sessions that have been in the menu for too long."""
        with self.lock:
            to_gc: List[SessionInfo] = []
            for info in self._sessions.values():
                # Get the status using remote call and ray.get
                status = ray.get(info.session.status.remote())
                menu_minutes = status['num_menu_frames'] / (60 * 60)
                stale_activity = status.get('seconds_since_activity', 0) > STALL_TIMEOUT_SECONDS
                stale_frame = (
                    status.get('has_entered_game') and
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
            status = ray.get(info.session.status.remote())
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
    eval_lib.disable_gpus()
    ray.init()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    agent_kwargs = AGENT.value
    if not agent_kwargs['path']:
        raise ValueError('Must provide agent path.')

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
    )

    try:
        bot.run(bot.token)
    finally:
        bot.shutdown()

if __name__ == '__main__':
    app.run(main)
