"""Bot that runs on Discord and lets people play against phillip 2 in doubles matches."""

import dataclasses
import datetime
import json
import logging
import os
import time
import threading
from typing import Optional, Dict, List, Tuple, Any

from absl import app, flags
import fancyflags as ff
import discord
from discord import app_commands
from discord.ext import commands, tasks
import portpicker
import ray

from slippi_ai import train_lib
from slippi_ai import flag_utils, eval_lib
from slippi_ai import dolphin as dolphin_lib
from melee.enums import Character

# Discord settings
default_bot_token = os.environ.get('DISCORD_BOT_TOKEN')
BOT_TOKEN = flags.DEFINE_string(
    'token', default_bot_token, 'Discord bot token',
    required=default_bot_token is None)
COMMAND_PREFIX = flags.DEFINE_string('prefix', '!', 'Command prefix for the bot')
ADMIN_ROLE = flags.DEFINE_string('admin_role', 'AI Admin', 'Role name for admin commands')

# Bot settings
_DOLPHIN_CONFIG = dolphin_lib.DolphinConfig(
    infinite_time=False,
    save_replays=True,
    replay_dir='discordbot/replays',
    disable_audio=True,
    headless=False,  # Ensure headless is explicitly set here
    log_types=[],
    render=True,
)
DOLPHIN = ff.DEFINE_dict(
    'dolphin', **flag_utils.get_flags_from_default(_DOLPHIN_CONFIG))

MODELS_PATH = flags.DEFINE_string('models', 'discordbot/models', 'Path to models')

# Serves as the default agent people play against
agent_flags = eval_lib.AGENT_FLAGS.copy()
agent_flags.update(
    async_inference=ff.Boolean(True),
    jit_compile=ff.Boolean(False),
)
AGENT = ff.DEFINE_dict('agent', **agent_flags)

# Session management settings
MENU_TIMEOUT = flags.DEFINE_float(
    'menu_timeout', 3, 'Minutes before timing out a session in menu')
MAX_SESSIONS = flags.DEFINE_integer(
    'max_sessions', 4, 'Maximum number of concurrent sessions')

# Add this near the top with the other flags
USER_JSON_PATH2 = flags.DEFINE_string(
    'dolphin.user_json_path2', None, 'Path to the second user.json file for Dolphin')

class AgentInstance:
    """Manages a single agent instance with its own Dolphin."""

    def __init__(
        self,
        dolphin_config: dolphin_lib.DolphinConfig,
        agent_kwargs: dict,
        team_color: int,  # 0=red, 1=blue, 2=green
        character: Character = None,  # Add character parameter
        extra_dolphin_kwargs: dict = {},
    ):
        eval_lib.disable_gpus()
        self.dolphin_config = dolphin_config
        self.stop_requested = threading.Event()
        self.team_color = team_color

        # Log the dolphin config to debug
        logging.info(f"AgentInstance init with config: headless={dolphin_config.headless}, render={dolphin_config.render}, teams_connect_code={dolphin_config.teams_connect_code}, team_color={team_color}")

        dolphin_kwargs = dolphin_config.to_kwargs()
        # Make sure these values are explicitly set in the kwargs
        dolphin_kwargs['headless'] = dolphin_config.headless
        dolphin_kwargs['render'] = dolphin_config.render
        # Ensure teams_connect_code is passed to Dolphin
        if hasattr(dolphin_config, 'teams_connect_code') and dolphin_config.teams_connect_code:
            dolphin_kwargs['teams_connect_code'] = dolphin_config.teams_connect_code
            
        dolphin_kwargs.update(extra_dolphin_kwargs)

        # Log the final dolphin kwargs
        logging.info(f"Final dolphin kwargs: {dolphin_kwargs}")

        player = dolphin_lib.AI()
        
        # Set the character if provided
        if character is not None:
            player.character = character
            logging.info(f"Setting character to {character.name}")
        
        dolphin = dolphin_lib.Dolphin(
            players={1: player},
            desired_teams={1: team_color},  # This is how we set team colors properly
            **dolphin_kwargs,
        )

        # Set initial opponent port to None, will be determined during gameplay
        # Check if the Dolphin object has controllers attribute, otherwise try alternate approach
        controller = dolphin.controllers[1]

        agent = eval_lib.build_agent(
            console_delay=15,
            controller=controller,
            opponent_port=None,
            run_on_cpu=True,
            **agent_kwargs,
        )

        eval_lib.update_character(player, agent.config)

        self._num_menu_frames = 0
        self._thread = None
        self._dolphin = dolphin
        self._agent = agent
        self._player = player

    def start(self):
        """Start the agent thread."""
        def run():
            # Don't block in the menu so that we can stop if asked to.
            gamestates = self._dolphin.iter_gamestates(skip_menu_frames=False)

            # This gets us through the menus and into the first frame of the actual game
            for gamestate in gamestates:
                if self.stop_requested.is_set():
                    self._dolphin.stop()
                    return

                if not dolphin_lib.is_menu_state(gamestate):
                    self._num_menu_frames = 0
                    break

                self._num_menu_frames += 1

            # Once in game, determine opponent ports based on teams
            opponent_ports = []
            agent_team = self._player.team
            for p in gamestate.players:
                if p != self.port and gamestate.players[p].team != agent_team:
                    opponent_ports.append(p)
            
            # If we found opponents, use the first one
            if opponent_ports:
                self._agent.players = (self.port, opponent_ports[0])

            # Main loop
            self._agent.start()
            try:
                while not self.stop_requested.is_set():
                    gamestate = next(gamestates)
                    if not dolphin_lib.is_menu_state(gamestate):
                        self._agent.step(gamestate)
                        self._num_menu_frames = 0
                    else:
                        self._num_menu_frames += 1
            finally:
                self._agent.stop()
                self._dolphin.stop()

        self._thread = threading.Thread(target=run)
        self._thread.start()

    def num_menu_frames(self) -> int:
        return self._num_menu_frames

    def status(self) -> dict:
        return {
            'num_menu_frames': self._num_menu_frames,
            'is_alive': self._thread.is_alive() if self._thread else False,
        }

    def stop(self):
        if self._thread:
            self.stop_requested.set()
            self._thread.join()

RemoteAgentInstance = ray.remote(AgentInstance)

class DoublesSession:
    """Session for doubles matches with 1-2 AI agents."""

    def __init__(
        self,
        agents: Dict[int, AgentInstance],  # port -> AgentInstance
    ):
        self.agents = agents
        
    def num_menu_frames(self) -> int:
        """Return the maximum number of menu frames across all agents."""
        frames = [agent.num_menu_frames.remote() for agent in self.agents.values()]
        if frames:
            frames_values = ray.get(frames)
            return max(frames_values)
        return 0

    def status(self) -> dict:
        """Return the combined status of all agents."""
        # Fix: use .remote() for Ray actor method calls
        agent_status_refs = {port: agent.status.remote() for port, agent in self.agents.items()}
        agent_statuses = {port: ray.get(status_ref) for port, status_ref in agent_status_refs.items()}
        all_alive = all(status['is_alive'] for status in agent_statuses.values())
        max_menu_frames = max([status['num_menu_frames'] for status in agent_statuses.values()], default=0)
        
        return {
            'agent_statuses': agent_statuses,
            'is_alive': all_alive,
            'num_menu_frames': max_menu_frames,
        }

    def start(self):
        """Start all agents."""
        for port, agent in self.agents.items():
            agent.start.remote()  # Ray actor reference

    def stop(self):
        """Stop all agents."""
        for port, agent in self.agents.items():
            agent.stop.remote()  # Ray actor reference
            

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
        (Character.MARIO, "Mario"),
        (Character.FOX, "Fox"),
        (Character.CPTFALCON, "Captain Falcon"),
        (Character.DK, "Donkey Kong"),
        (Character.BOWSER, "Bowser"),
        (Character.LINK, "Link"),
        (Character.SHEIK, "Sheik"),
        (Character.NESS, "Ness"),
        (Character.PEACH, "Peach"),
        (Character.POPO, "Ice Climbers"),
        (Character.PIKACHU, "Pikachu"),
        (Character.SAMUS, "Samus"),
        (Character.YOSHI, "Yoshi"),
        (Character.JIGGLYPUFF, "Jigglypuff"),
        (Character.MEWTWO, "Mewtwo"),
        (Character.LUIGI, "Luigi"),
        (Character.MARTH, "Marth"),
        (Character.ZELDA, "Zelda"),
        (Character.YLINK, "Young Link"),
        (Character.DOC, "Dr. Mario"),
        (Character.FALCO, "Falco"),
        (Character.PICHU, "Pichu"),
        (Character.GAMEANDWATCH, "Mr. Game & Watch"),
        (Character.GANONDORF, "Ganondorf"),
        (Character.ROY, "Roy"),
    ]
    
    for char, display_name in playable_characters:
        choices.append(app_commands.Choice(name=display_name, value=char.name))
    
    return choices

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
        
        # Initialize bot with command prefix and intents
        super().__init__(command_prefix=prefix, intents=intents)
        
        self.token = token
        self.dolphin_config = dolphin_config
        self.agent_kwargs = agent_kwargs
        self.admin_role = admin_role
        self._max_sessions = max_sessions
        self._menu_timeout = menu_timeout

        self._sessions: Dict[int, SessionInfo] = {}  # User ID -> SessionInfo
        self.lock = threading.RLock()

        with open(dolphin_config.user_json_path) as f:
            user_json = json.load(f)

        self.bot_code = user_json['connectCode']

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
                value="`/play` - Start a game with one AI agent\n"
                      "`/play2` - Start a game with two AI agents\n"
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
                name="How to Play",
                value="Use `/play` with your connect code, create a direct lobby, and wait for the AI to join.",
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
        async def reload_command(interaction: discord.Interaction):
            if not self.is_admin(interaction):
                await interaction.response.send_message("You don't have permission to use this command.", ephemeral=True)
                return
                
            with self.lock:
                self._reload_models()
            models_str = ", ".join(self._models)
            embed = discord.Embed(
                title="Available Agents",
                description=models_str,
                color=discord.Color.green()
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
        
        # Update the /play command to make the character parameter required
        @self.tree.command(name="play", description="Start a game with one AI agent")
        @app_commands.describe(
            connect_code="Your Slippi lobby code (no # needed)",
            team_color="Team color for the AI agent",
            character="Character for the AI to play"  # Removed the default note
        )
        @app_commands.choices(team_color=[
            app_commands.Choice(name="Red", value="red"),
            app_commands.Choice(name="Blue", value="blue"),
            app_commands.Choice(name="Green", value="green"),
        ])
        @app_commands.choices(character=get_valid_character_choices())
        async def play_command(
            interaction: discord.Interaction, 
            connect_code: str,
            character: str, 
            team_color: str = "red",
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
                message = f"Connecting to {interaction.user.name} ({connect_code}) with agent {agent_name} on port {port} with team color {team_color} playing {char_display}"
                logging.info(message)
                
                # Tell the user we're processing
                await interaction.response.send_message(message)

                team_colors = {port: team_color_num}
                
                # Create and start agent instance
                agent_instance = self._create_agent_instance(
                    connect_code, 
                    port, 
                    team_color_num,
                    self._get_agent_kwargs(user_id, port, agent_name),
                    character=char_enum
                )
                
                # Create session with single agent
                agents = {port: agent_instance}
                session = self._start_session(agents)
                
                self._sessions[user_id] = SessionInfo(
                    session=session,
                    start_time=datetime.datetime.now(),
                    discord_name=interaction.user.name,
                    discord_id=user_id,
                    connect_code=connect_code,
                    agents={port: agent_name},
                    team_colors=team_colors
                )
        
        # Update the /play2 command to correctly handle team colors for both agents
        @self.tree.command(name="play2", description="Start a game with two AI agents")
        @app_commands.describe(
            connect_code="Your Slippi lobby code",
            character1="Character for first agent",
            team1="Team color for first agent",
            character2="Character for second agent",
            team2="Team color for second agent",
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
        async def play2_command(
            interaction: discord.Interaction, 
            connect_code: str, 
            character1: str,
            team1: str,
            character2: str,
            team2: str,
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
                
                message = f"Connecting to {interaction.user.name} ({connect_code}) with:\n" \
                         f"- Agent {agent1_name} with team color {team1} playing {char1_display}\n" \
                         f"- Agent {agent2_name} with team color {team2} playing {char2_display}"
                logging.info(message)
                
                # Tell the user we're processing
                await interaction.response.send_message(message)

                # Create agent instances (each with its own Dolphin)
                agent1 = self._create_agent_instance(
                    connect_code,
                    team1_num,
                    self._get_agent_kwargs(user_id, 1, agent1_name),  # Use index 1 just for agent selection
                    character=char1_enum
                )
                
                agent2 = self._create_agent_instance(
                    connect_code,
                    team2_num,
                    self._get_agent_kwargs(user_id, 2, agent2_name),  # Use index 2 just for agent selection
                    character=char2_enum,
                    is_second_agent=True  # Mark this as the second agent
                )
                
                # Create session with both agents (using logical ports 1 and 2 as keys)
                agents = {1: agent1, 2: agent2}
                session = self._start_session(agents)
                
                # Store the team colors keyed by logical ports (these are just for display purposes)
                team_colors = {1: team1_num, 2: team2_num}
                
                self._sessions[user_id] = SessionInfo(
                    session=session,
                    start_time=datetime.datetime.now(),
                    discord_name=interaction.user.name,
                    discord_id=user_id,
                    connect_code=connect_code,
                    agents={1: agent1_name, 2: agent2_name},  # Using logical ports as keys
                    team_colors=team_colors
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
                    
                    agents_info = []
                    for port, agent in session_info.agents.items():
                        team = session_info.team_colors.get(port, 0)
                        team_name = ["Red", "Blue", "Green"][team]
                        agents_info.append(f"Port {port}: {agent} (Team {team_name})")
                    
                    agents_text = "\n".join(agents_info)
                    
                    embed.add_field(
                        name=f"Playing against {session_info.discord_name}",
                        value=f"Connect code: {session_info.connect_code}\n"
                              f"Duration: {timedelta}\n"
                              f"Menu time: {menu_time}\n"
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

    def _create_agent_instance(
        self,
        connect_code: str,
        team_color: int,
        agent_kwargs: dict,
        character: Character = None,
        is_second_agent: bool = False,
    ) -> AgentInstance:
        """Create a new agent instance with its own Dolphin"""
        # Create a deep copy of the dolphin config
        config = dataclasses.replace(self.dolphin_config)
        
        # Set specific parameters for this instance
        config.slippi_port = portpicker.pick_unused_port()
        config.connect_code = connect_code
        
        # Make sure teams_connect_code is set from the connect_code
        # This is critical for doubles mode to work properly
        config.teams_connect_code = connect_code
        
        # For the discordbot, ensure these are explicitly set as needed
        config.render = True     # So we can see what's happening
        config.headless = False  # Ensure we show the Dolphin window
        
        # Print the config for debugging
        logging.info(f"Creating agent with dolphin config: connect_code={connect_code}, teams_connect_code={config.teams_connect_code}, team_color={team_color}")
        
        # Use the second user.json path if this is the second agent and the path is provided
        if is_second_agent and USER_JSON_PATH2.value:
            config.user_json_path = USER_JSON_PATH2.value
        
        # Explicitly add the teams_connect_code to ensure it's passed through
        extra_kwargs = {}
        if hasattr(config, 'teams_connect_code') and config.teams_connect_code:
            extra_kwargs['teams_connect_code'] = config.teams_connect_code
        
        # Create and return the remote agent instance
        return RemoteAgentInstance.remote(
            config, agent_kwargs, team_color, character, extra_kwargs
        )

    def _start_session(self, agents: Dict[int, AgentInstance]) -> DoublesSession:
        """Start a new doubles session with the specified agents"""
        session = RemoteDoublesSession.remote(agents)
        # Call the remote method and get the result
        ray.get(session.start.remote())
        return session

    def _stop_sessions(self, infos: List[SessionInfo]):
        """Stop the specified sessions"""
        with self.lock:
            # Create list of tasks and wait for them to complete
            stop_tasks = [info.session.stop.remote() for info in infos]
            if stop_tasks:
                ray.wait(stop_tasks)

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
                if not status['is_alive'] or menu_minutes > self._menu_timeout:
                    to_gc.append(info)

            self._stop_sessions(to_gc)
            if to_gc:
                names = ", ".join([info.discord_name for info in to_gc])
                logging.info(f'GCed sessions: {names}')
            return to_gc

    @tasks.loop(minutes=1)
    async def _do_chores(self):
        with self.lock:
            await self._gc_sessions()

    def shutdown(self):
        with self.lock:
            self._stop_sessions(list(self._sessions.values()))
            self._do_chores.cancel()

    async def close(self):
        self.shutdown()
        await super().close()

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
