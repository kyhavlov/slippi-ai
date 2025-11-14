import abc
import atexit
import dataclasses
import logging
import random
from typing import Dict, Mapping, Optional, Iterator

import fancyflags as ff
import portpicker

import melee
from melee.console import get_dolphin_version, DumpConfig, DolphinBuild

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

class RemoteAI(Player):

  def controller_type(self) -> melee.ControllerType:
    return melee.ControllerType.STANDARD

  def menuing_kwargs(self) -> Dict:
      return {}

def is_menu_state(gamestate: melee.GameState) -> bool:
  return gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]

class ConnectFailed(Exception):
  """Raised when we fail to connect to the console."""

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
      render: Optional[bool] = None,  # Render even when running headless.
      connect_code: Optional[str] = None,
      teams_connect_code: Optional[str] = None,
      desired_teams: Mapping[int, int] = {},
      existing_dolphin: bool = False,
      **console_kwargs,
  ) -> None:
    self._players = players
    self._stage = stage

    platform = None
    version = get_dolphin_version(path)

    if render is None:
      render = not headless

    if not render:
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
      elif not version.mainline:
        raise ValueError(
            'Headless requires mainline dolphin or a custom dolphin build. '
            'See https://github.com/vladfi1/libmelee?tab=readme-ov-file#setup-instructions')

    slippi_port = slippi_port or portpicker.pick_unused_port()

    # if we have remote players, dont wait for them to select characters
    remote_players = [port for port, player in players.items() if isinstance(player, RemoteAI)]
    self.menu_helper = melee.MenuHelper(is_singles=len(players) == 2, remote_players=remote_players)

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
        **console_kwargs,
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

      new_characters = []
      for i, (controller, player) in enumerate(self._menuing_controllers):
        if isinstance(player, AI):
          player.shuffle_character()
          new_characters.append(player.character)
      
      print(f"shuffled characters for next game: {new_characters} port: {self.console.slippi_port}")

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
    for controller in self.controllers.values():
      controller.disconnect()
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
  emulation_speed: float = 1.0  # Set to 0 for unlimited speed. Mainline only.
  infinite_time: bool = True  # Infinite time no stocks.
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
    emulation_speed=ff.Float(1.0),
    infinite_time=ff.Boolean(False, 'Infinite time no stocks.'),
    log_level=ff.Integer(3, 'Dolphin log level, defaults to WARN.'),
    log_types=ff.StringList(['SLIPPI'], 'Enabled logging categories.'),
    disable_audio=ff.Boolean(False, 'Disable dolphin audio.'),
    force_lan_ip=ff.String(None, 'Force Slippi LAN IP'),
)
