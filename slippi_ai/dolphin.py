import abc
import atexit
import configparser
import dataclasses
import logging
import os
import random
from typing import Dict, Mapping, Optional, Iterator

import fancyflags as ff
import portpicker

import melee
from melee.console import get_dolphin_version, DumpConfig, DolphinBuild

from slippi_ai import instant_match as instant_match_lib

class Player(abc.ABC):

  @abc.abstractmethod
  def controller_type(self) -> melee.ControllerType:
    pass

  @abc.abstractmethod
  def menuing_kwargs(self) -> Dict:
    pass


class Human(Player):

  def controller_type(self) -> melee.ControllerType:
    return melee.ControllerType.GCN_ADAPTER

  def menuing_kwargs(self) -> Dict:
      return {}

@dataclasses.dataclass
class CPU(Player):
  character: melee.Character = melee.Character.FOX
  level: int = 9

  def controller_type(self) -> melee.ControllerType:
    return melee.ControllerType.STANDARD

  def menuing_kwargs(self) -> Dict:
      return dict(character_selected=self.character, cpu_level=self.level)

@dataclasses.dataclass
class AI(Player):
  character: melee.Character = melee.Character.FOX
  character_weight_table: Optional[dict[melee.Character, int]] = None

  def shuffle_character(self):
    if self.character_weight_table is not None:
      self.character = random.choices(
          list(self.character_weight_table.keys()),
          weights=list(self.character_weight_table.values()),
      )[0]

  def controller_type(self) -> melee.ControllerType:
    return melee.ControllerType.STANDARD

  def menuing_kwargs(self) -> Dict:
      return dict(character_selected=self.character)


@dataclasses.dataclass
class ScheduledAI(AI):
  """AI player whose character is set explicitly each match."""

  _next_character: Optional[melee.Character] = dataclasses.field(default=None, init=False)

  def set_next_character(self, character: melee.Character):
    if not isinstance(character, melee.Character):
      character = melee.Character(character)
    self._next_character = character

  def shuffle_character(self):
    if self._next_character is None:
      raise RuntimeError('ScheduledAI requires set_next_character before shuffle_character().')
    self.character = self._next_character
    self._next_character = None

class RemoteAI(Player):

  def controller_type(self) -> melee.ControllerType:
    return melee.ControllerType.STANDARD

  def menuing_kwargs(self) -> Dict:
      return {}

def is_menu_state(gamestate: melee.GameState) -> bool:
  return gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]

class ConnectFailed(Exception):
  """Raised when we fail to connect to the console."""


def _enable_gecko_cheats(console: melee.Console):
  config_path = os.path.join(console._get_dolphin_home_path(), 'Config', 'Dolphin.ini')
  config = configparser.ConfigParser()
  config.read(config_path)
  if not config.has_section('Core'):
    config.add_section('Core')
  config.set('Core', 'EnableCheats', 'True')
  with open(config_path, 'w') as f:
    config.write(f)

class Dolphin:

  def __init__(
      self,
      path: str,
      iso: str,
      players: Mapping[int, Player],
      stage: melee.Stage = melee.Stage.RANDOM_STAGE,
      online_delay: int = 0,  # overrides Console's default of 2
      blocking_input: bool = True,
      console_timeout: Optional[float] = None,
      slippi_port: Optional[int] = None,  # Picked automatically if None
      save_replays=False,  # Override default in Console
      env_vars: Optional[dict] = None,
      headless: bool = False,
      custom_headless: bool = False,
      render: Optional[bool] = None,  # Render even when running headless.
      connect_code: Optional[str] = None,
      teams_connect_code: Optional[str] = None,
      desired_teams: Mapping[int, int] = {},
      existing_dolphin: bool = False,
      starting_stocks: int = 0,
      instant_match: bool = False,
      instant_match_character_pool: Optional[list[str]] = None,
      instant_match_stage_pool: Optional[list[str]] = None,
      **console_kwargs,
  ) -> None:
    self._players = players
    self._stage = stage
    self._instant_match_config: Optional[instant_match_lib.InstantMatchConfig] = None

    platform = None
    version = get_dolphin_version(path)

    if render is None:
      render = not headless

    if not render and not custom_headless:
      console_kwargs.update(gfx_backend='Null')

    if headless:
      console_kwargs.update(
          disable_audio=True,
      )
      if version.mainline:
        platform = 'headless'
        # console_kwargs.update(emulation_speed=0)

      if version.build is DolphinBuild.EXI_AI:
        console_kwargs.update(
            use_exi_inputs=True,
            enable_ffw=True,
        )
      elif custom_headless:
        logging.info(
            'Allowing custom headless Dolphin build: %s (%s)',
            path, version)
      elif not version.mainline:
        raise ValueError(
            'Headless requires mainline dolphin or a custom dolphin build. '
            'See https://github.com/vladfi1/libmelee?tab=readme-ov-file#setup-instructions')

    slippi_port = slippi_port or portpicker.pick_unused_port()

    # if we have remote players, dont wait for them to select characters
    remote_players = [port for port, player in players.items() if isinstance(player, RemoteAI)]
    self.menu_helper = melee.MenuHelper(is_singles=len(players) == 2, remote_players=remote_players)
    if starting_stocks and console_kwargs.get('infinite_time', False):
      raise ValueError('starting_stocks is incompatible with infinite_time=True.')
    if instant_match:
      if existing_dolphin:
        raise ValueError('instant_match requires launching a fresh Dolphin instance.')
      self._instant_match_config = instant_match_lib.resolve_config(
          players=players,
          stage=stage,
          character_pool=instant_match_character_pool,
          stage_pool=instant_match_stage_pool,
          starting_stocks=starting_stocks,
      )

    console_starting_stocks = starting_stocks
    console = melee.Console(
        path=path,
        online_delay=online_delay,
        blocking_input=blocking_input,
        polling_mode=console_timeout is not None,
        polling_timeout=console_timeout,
        slippi_port=slippi_port,
        copy_home_directory=False,
        setup_gecko_codes=True,
        save_replays=save_replays,
        slippi_starting_stocks=console_starting_stocks,
        **console_kwargs,
    )
    _enable_gecko_cheats(console)
    if self._instant_match_config is not None:
      instant_match_lib.inject_gecko_codes(console, self._instant_match_config)
      logging.info(
          'Enabled instant_match with chars=%s stages=%s stocks=%d',
          [c.name for c in self._instant_match_config.character_pool],
          [s.name for s in self._instant_match_config.stage_pool],
          self._instant_match_config.starting_stocks,
      )
    atexit.register(console.stop)
    self.console = console

    self.controllers: Mapping[int, melee.Controller] = {}
    self._menuing_controllers: list[tuple[melee.Controller, Player]] = []
    self._autostart = True
    self._connect_code = connect_code
    self._teams_connect_code = teams_connect_code
    self._desired_teams = desired_teams
    self._prev_menu_state = False
    for port, player in players.items():
      controller = melee.Controller(
          console, port, player.controller_type())
      if not isinstance(player, RemoteAI):
        self.controllers[port] = controller
      if isinstance(player, Human):
        self._autostart = False
      elif not isinstance(player, RemoteAI):
        self._menuing_controllers.append((controller, player))

    if not existing_dolphin:
      console.run(
          iso_path=iso,
          environment_vars=env_vars,
          platform=platform,
      )

    logging.info('Connecting to console...')

    if not console.connect():
      import os
      logging.error(
          f"PID {os.getpid()}: failed to connect to the console"
          f" {console.temp_dir} on port {slippi_port}")

      raise ConnectFailed(f"Failed to connect to the console on port {slippi_port}.")
    logging.info('Connected to console')

    for controller in self.controllers.values():
      if not controller.connect():
        self.stop()
        raise ConnectFailed("Failed to connect the controller.")

  def next_gamestate(self) -> melee.GameState:
    gamestate = self.console.step()
    if gamestate is None:
      raise TimeoutError('Console timed out.')
    return gamestate

  def step(self) -> melee.GameState:
    gamestate = self.next_gamestate()

    # The console object keeps track of how long your bot is taking to process frames
    #   And can warn you if it's taking too long
    #if self.console.processingtime * 1000 > 12:
    #    print("WARNING: Last frame took " + str(self.console.processingtime*1000) + "ms to process.")

    if is_menu_state(gamestate) and not self._prev_menu_state:
      self.menu_helper.done_selecting_character = {}
      for port, player in self._players.items():
        if isinstance(player, AI):
          self.menu_helper.done_selecting_character[port] = False

      if self._instant_match_config is not None:
        self._stage = self._instant_match_config.choose_initial_stage()

      new_characters = []
      for i, (controller, player) in enumerate(self._menuing_controllers):
        if isinstance(player, AI):
          if (self._instant_match_config is not None and
              self._instant_match_config.character_pool):
            player.character = self._instant_match_config.choose_initial_character()
            if isinstance(player, ScheduledAI):
              player._next_character = None
          else:
            player.shuffle_character()
          new_characters.append(player.character)
      logging.debug(
          "Shuffled characters for next game on port %s: %s",
          self.console.slippi_port,
          new_characters,
      )

    self._prev_menu_state = is_menu_state(gamestate)

    menu_frames = 0
    while is_menu_state(gamestate):
      for i, (controller, player) in enumerate(self._menuing_controllers):
        autostart_enabled = False
        if self._autostart and menu_frames > 180:
          if i == 0 or getattr(self.menu_helper, 'stage_selected', False):
            autostart_enabled = True

        self.menu_helper.menu_helper_simple(
            gamestate, controller,
            stage_selected=self._stage,
            connect_code=self._connect_code,
            teams_connect_code=self._teams_connect_code,
            desired_teams=self._desired_teams,
            offline_teams=self._desired_teams and not self._teams_connect_code and len(self._players) == 4,
            autostart=autostart_enabled,
            swag=False,
            costume=i,
            **player.menuing_kwargs())

      gamestate = self.next_gamestate()
      menu_frames += 1

    return gamestate

  def iter_gamestates(self, skip_menu_frames: bool = True) -> Iterator[melee.GameState]:
    while True:
      gamestate = self.next_gamestate()

      menu_frames = 0
      while is_menu_state(gamestate):
        if not skip_menu_frames:
          yield gamestate

        for i, (controller, player) in enumerate(self._menuing_controllers):
          autostart_enabled = False
          if self._autostart and menu_frames > 180:
            if i == 0 or getattr(self.menu_helper, 'stage_selected', False):
              autostart_enabled = True

          self.menu_helper.menu_helper_simple(
              gamestate, controller,
              stage_selected=self._stage,
              connect_code=self._connect_code,
              teams_connect_code=self._teams_connect_code,
              desired_teams=self._desired_teams,
              offline_teams=self._desired_teams and not self._teams_connect_code and len(self._players) == 4,
              autostart=autostart_enabled,
              swag=False,
              costume=i,
              **player.menuing_kwargs())

        gamestate = self.next_gamestate()
        menu_frames += 1

      yield gamestate

  def stop(self):
    for controller in getattr(self, 'controllers', {}).values():
      controller.disconnect()
    if hasattr(self, 'console'):
      self.console.stop()

  def __del__(self):
    self.stop()

  def multi_step(self, n: int):
    for _ in range(n):
      self.step()

_field = lambda f: dataclasses.field(default_factory=f)

@dataclasses.dataclass
class DolphinConfig:
  """Configure dolphin for evaluation."""
  path: Optional[str] = None  # Path to folder containing the dolphin executable
  iso: Optional[str] = None  # Path to melee 1.02 iso.
  stage: melee.Stage = melee.Stage.RANDOM_STAGE  # Which stage to play on.
  online_delay: int = 0  # Simulate online delay.
  blocking_input: bool = True  # Have game wait for AIs to send inputs.
  console_timeout: Optional[float] = None  # Seconds to wait for console inputs before throwing an error.
  slippi_port: Optional[int] = None  # Local ip port to communicate with dolphin.
  fullscreen: bool = False # Run dolphin in full screen mode
  render: Optional[bool] = None  # Render frames. Only disable if using vladfi1\'s slippi fork.
  save_replays: bool = False  # Save slippi replays to the usual location.
  replay_dir: Optional[str] = None  # Directory to save replays to.
  gfx_backend: str = ''  # Graphics backend to use.
  disable_audio: bool = False  # Disable dolphin audio.
  headless: bool = True  # Headless configuration: exi + ffw, no graphics or audio.
  custom_headless: bool = False  # Allow non-mainline custom Ishiiruka headless builds.
  emulation_speed: float = 1.0  # Set to 0 for unlimited speed. Mainline only.
  infinite_time: bool = True  # Infinite time no stocks.
  starting_stocks: int = 0  # Set >0 to override starting stocks on supported Ishiiruka builds.
  instant_match: bool = False  # Reload local VS instantly and randomize future matches via Gecko.
  instant_match_character_pool: list[str] = dataclasses.field(default_factory=list)
  instant_match_stage_pool: list[str] = dataclasses.field(default_factory=list)
  log_level: int = 3  # WARN; 0 to disable
  log_types: list[str] = dataclasses.field(default_factory=['SLIPPI'].copy)
  dump: DumpConfig = _field(DumpConfig)  # For framedumping.
  existing_dolphin: bool = False  # If true, don't run dolphin. Use existing dolphin instance.
  force_lan_ip: Optional[str] = None  # Force Slippi LAN IP

  # For online play
  connect_code: Optional[str] = None
  user_json_path: Optional[str] = None
  user_json_path2: Optional[str] = None

  def to_kwargs(self) -> dict:
    kwargs = dataclasses.asdict(self)
    del kwargs['dump']
    kwargs['dump_config'] = self.dump
    return kwargs

  @classmethod
  def kwargs_from_flags(cls, flags: dict) -> dict:
    kwargs = flags.copy()
    del kwargs['dump']
    kwargs['dump_config'] = DumpConfig(**flags['dump'])
    return kwargs

# TODO: replace usage with the above dataclass
DOLPHIN_FLAGS = dict(
    path=ff.String(None, 'Path to folder containing the dolphin executable.'),
    iso=ff.String(None, 'Path to melee 1.02 iso.'),
    stage=ff.EnumClass(melee.Stage.RANDOM_STAGE, melee.Stage, 'Which stage to play on.'),
    online_delay=ff.Integer(0, 'Simulate online delay.'),
    blocking_input=ff.Boolean(True, 'Have game wait for AIs to send inputs.'),
    slippi_port=ff.Integer(None, 'Local ip port to communicate with dolphin.'),
    fullscreen=ff.Boolean(False, 'Run dolphin in full screen mode.'),
    render=ff.Boolean(None, 'Render frames. Only disable if using vladfi1\'s slippi fork.'),
    save_replays=ff.Boolean(False, 'Save slippi replays to the usual location.'),
    replay_dir=ff.String(None, 'Directory to save replays to.'),
    headless=ff.Boolean(
        False, 'Headless configuration: exi + ffw, no graphics or audio.'),
    custom_headless=ff.Boolean(
        False, 'Allow non-mainline custom Ishiiruka headless builds.'),
    emulation_speed=ff.Float(1.0),
    infinite_time=ff.Boolean(False, 'Infinite time no stocks.'),
    starting_stocks=ff.Integer(0, 'Set >0 to override starting stocks on supported Ishiiruka builds.'),
    instant_match=ff.Boolean(False, 'Instantly reload local VS matches and randomize rematches via Gecko.'),
    instant_match_character_pool=ff.StringList(
        [], 'Character pool for instant_match, e.g. fox,falco,marth,sheik.'),
    instant_match_stage_pool=ff.StringList(
        [], 'Stage pool for instant_match, e.g. battlefield,final_destination,yoshis_story.'),
    log_level=ff.Integer(3, 'Dolphin log level, defaults to WARN.'),
    log_types=ff.StringList(['SLIPPI'], 'Enabled logging categories.'),
    disable_audio=ff.Boolean(False, 'Disable dolphin audio.'),
    force_lan_ip=ff.String(None, 'Force Slippi LAN IP'),
)
