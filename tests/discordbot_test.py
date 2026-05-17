import dataclasses
import types
import unittest
from unittest import mock
from typing import Optional

import numpy as np

from scripts import discordbot
from slippi_ai.controller_heads import SampleOutputs


@dataclasses.dataclass
class FakeGameState:
  frame: int
  finalized_frame: Optional[int]
  custom: dict = dataclasses.field(default_factory=dict)


class FinalizedDelayBufferTest(unittest.TestCase):

  def test_waits_for_delay_target_instead_of_latest_finalized(self):
    buffer = discordbot.FinalizedDelayBuffer(delay=3)

    outputs = []
    for frame in range(100, 106):
      outputs.extend(buffer.push(FakeGameState(frame, finalized_frame=105)))

    self.assertEqual([state.frame for state in outputs], [100, 101, 102])
    self.assertNotIn(105, [state.frame for state in outputs])

  def test_does_not_publish_speculative_target(self):
    buffer = discordbot.FinalizedDelayBuffer(delay=3)

    outputs = []
    for frame in range(100, 111):
      outputs.extend(buffer.push(FakeGameState(frame, finalized_frame=106)))

    self.assertEqual([state.frame for state in outputs], list(range(100, 107)))

  def test_publishes_backlog_in_order_when_finalization_jumps(self):
    buffer = discordbot.FinalizedDelayBuffer(delay=2)

    outputs = []
    for frame in range(100, 103):
      outputs.extend(buffer.push(FakeGameState(frame, finalized_frame=99)))
    outputs.extend(buffer.push(FakeGameState(103, finalized_frame=103)))

    self.assertEqual([state.frame for state in outputs], [100, 101])

  def test_marks_published_frames_as_delayed_finalized(self):
    buffer = discordbot.FinalizedDelayBuffer(delay=1)

    outputs = []
    for frame in range(100, 102):
      outputs.extend(buffer.push(FakeGameState(frame, finalized_frame=101)))

    self.assertEqual([state.frame for state in outputs], [100])
    self.assertTrue(outputs[0].custom['discordbot_delayed_finalized'])

  def test_clear_prevents_old_game_frames_from_publishing(self):
    buffer = discordbot.FinalizedDelayBuffer(delay=3)
    buffer.push(FakeGameState(100, finalized_frame=99))
    buffer.push(FakeGameState(101, finalized_frame=99))
    buffer.clear()

    outputs = []
    for frame in range(200, 204):
      outputs.extend(buffer.push(FakeGameState(frame, finalized_frame=203)))

    self.assertEqual([state.frame for state in outputs], [200])


class DiscordInGameFrameProcessorTest(unittest.TestCase):

  def test_negative_reset_frame_does_not_require_finalized_frame(self):
    processor = discordbot.DiscordInGameFrameProcessor(
        logical_port=1,
        observation_delay=0,
    )

    outputs = processor.process(FakeGameState(frame=-123, finalized_frame=None))

    self.assertEqual([state.frame for state in outputs], [-123])
    self.assertTrue(outputs[0].custom['discordbot_delayed_finalized'])

  def test_non_negative_frame_requires_finalized_frame(self):
    processor = discordbot.DiscordInGameFrameProcessor(
        logical_port=1,
        observation_delay=0,
    )

    with self.assertRaisesRegex(RuntimeError, 'missing finalized_frame'):
      processor.process(FakeGameState(frame=0, finalized_frame=None))

  def test_game_starts_after_negative_reset_marker(self):
    processor = discordbot.DiscordInGameFrameProcessor(
        logical_port=1,
        observation_delay=0,
    )

    reset_outputs = processor.process(FakeGameState(frame=-123, finalized_frame=None))
    game_outputs = processor.process(FakeGameState(frame=0, finalized_frame=0))

    self.assertEqual([state.frame for state in reset_outputs], [-123])
    self.assertEqual([state.frame for state in game_outputs], [0])
    self.assertTrue(game_outputs[0].custom['discordbot_delayed_finalized'])


class DiscordAutocompleteHelperTest(unittest.TestCase):

  def test_playstyle_autocomplete_uses_supported_names(self):
    state = {
        'name_map': {
            'Master Player': 0,
            'Cody': 1,
            'Amsa': 2,
        },
    }

    choices = discordbot.get_playstyle_autocomplete_choices(state, 'co')

    self.assertEqual([(choice.name, choice.value) for choice in choices], [('Cody', 'Cody')])

  def test_playstyle_autocomplete_includes_default_option(self):
    state = {'name_map': {'Master Player': 0, 'Cody': 1}}

    choices = discordbot.get_playstyle_autocomplete_choices(state, '')

    self.assertEqual(
        (choices[0].name, choices[0].value),
        ('Default (Master Player)', discordbot.DEFAULT_PLAYSTYLE_SENTINEL),
    )
    self.assertEqual(
        [(choice.name, choice.value) for choice in choices[1:]],
        [('Cody', 'Cody')],
    )

  def test_playstyle_autocomplete_uses_character_default_option(self):
    state = {
        'name_map': {
            'Master Player': 0,
            'Dragunov': 1,
            'Ralph': 2,
        },
    }

    choices = discordbot.get_playstyle_autocomplete_choices(state, '', 'Dragunov')

    self.assertEqual(
        (choices[0].name, choices[0].value),
        ('Default (Dragunov)', discordbot.DEFAULT_PLAYSTYLE_SENTINEL),
    )
    self.assertEqual(
        [(choice.name, choice.value) for choice in choices[1:]],
        [('Master Player', 'Master Player'), ('Ralph', 'Ralph')],
    )

  def test_playstyle_autocomplete_uses_none_for_models_without_names(self):
    state = {}

    choices = discordbot.get_playstyle_autocomplete_choices(state, '')

    self.assertEqual(
        [(choice.name, choice.value) for choice in choices],
        [('None', discordbot.DEFAULT_PLAYSTYLE_SENTINEL)],
    )

  def test_default_playstyle_sentinel_resolves_to_model_default(self):
    state = {'name_map': {'Master Player': 0, 'Cody': 1}}

    resolved = discordbot.resolve_playstyle_for_state(
        discordbot.DEFAULT_PLAYSTYLE_SENTINEL,
        'Master Player',
        state,
    )

    self.assertEqual(resolved, 'Master Player')

  def test_missing_playstyle_resolves_to_character_default(self):
    state = {'name_map': {'Master Player': 0, 'Dragunov': 1}}

    resolved = discordbot.resolve_playstyle_for_state(
        None,
        'Dragunov',
        state,
    )

    self.assertEqual(resolved, 'Dragunov')

  def test_parse_model_scoped_character_playstyle_defaults(self):
    defaults = discordbot.parse_default_playstyle_by_character(
        'rl_doubles_d21_v4_latest:MARTH=Dragunov,FOX=Ralph,SHEIK=Darkatma')

    self.assertEqual(
        defaults,
        {
            'rl_doubles_d21_v4_latest': {
                discordbot.Character.MARTH: 'Dragunov',
                discordbot.Character.FOX: 'Ralph',
                discordbot.Character.SHEIK: 'Darkatma',
            },
        },
    )

  def test_character_default_playstyle_validates_supported_name(self):
    state = {'name_map': {'Master Player': 0, 'Dragunov': 1}}
    defaults = {
        'rl_doubles_d21_v4_latest': {
            discordbot.Character.MARTH: 'dragunov',
        },
    }

    default = discordbot.get_character_default_playstyle(
        'rl_doubles_d21_v4_latest',
        discordbot.Character.MARTH,
        defaults,
        state,
    )

    self.assertEqual(default, 'Dragunov')

  def test_character_autocomplete_uses_allowed_characters(self):
    state = {
        'config': {
            'dataset': {
                'allowed_characters': 'fox,marth',
            },
        },
    }

    choices = discordbot.get_character_autocomplete_choices(state, '')

    self.assertEqual(
        [(choice.name, choice.value) for choice in choices],
        [('Fox', 'FOX'), ('Marth', 'MARTH')],
    )

  def test_character_autocomplete_filters_by_query(self):
    state = {'config': {'dataset': {'allowed_characters': 'fox,marth'}}}

    choices = discordbot.get_character_autocomplete_choices(state, 'mar')

    self.assertEqual([(choice.name, choice.value) for choice in choices], [('Marth', 'MARTH')])


class DiscordPlayerOrderTest(unittest.TestCase):

  def test_three_player_lobby_without_teammate_keeps_p1_empty(self):
    order = discordbot.build_discord_player_order(
        my_port=2,
        present_ports=[2, 3, 4],
        teammate_port=None,
    )

    self.assertEqual(order, (2, 1, 3, 4))

  def test_three_player_lobby_with_teammate_puts_human_opponent_after_team(self):
    order = discordbot.build_discord_player_order(
        my_port=2,
        present_ports=[1, 2, 3],
        teammate_port=3,
    )

    self.assertEqual(order, (2, 3, 1, 4))


class DiscordBotPlaystyleDefaultTest(unittest.IsolatedAsyncioTestCase):

  def _bot(self):
    bot = object.__new__(discordbot.DiscordBot)
    bot._default_agent_name = 'rl_doubles_d21_v4_latest'
    bot._requested_agents = {}
    bot.agent_kwargs = {'name': 'Master Player'}
    bot._models = {
        'rl_doubles_d21_v4_latest': {
            'name_map': {
                'Master Player': 0,
                'Dragunov': 1,
                'Ralph': 2,
            },
        },
    }
    bot._default_playstyles_by_character = {
        'rl_doubles_d21_v4_latest': {
            discordbot.Character.MARTH: 'Dragunov',
            discordbot.Character.FOX: 'Ralph',
        },
    }
    return bot

  async def test_playstyle_autocomplete_uses_selected_character_default(self):
    bot = self._bot()
    interaction = types.SimpleNamespace(
        user=types.SimpleNamespace(id=1),
        namespace=types.SimpleNamespace(character1='MARTH'),
    )

    choices = await discordbot.DiscordBot._autocomplete_playstyle_for_port(
        bot, interaction, '', 1)

    self.assertEqual(
        (choices[0].name, choices[0].value),
        ('Default (Dragunov)', discordbot.DEFAULT_PLAYSTYLE_SENTINEL),
    )

  def test_missing_playstyle_uses_character_default_at_launch(self):
    bot = self._bot()

    resolved = discordbot.DiscordBot._resolve_playstyle(
        bot,
        'rl_doubles_d21_v4_latest',
        None,
        discordbot.Character.FOX,
    )

    self.assertEqual(resolved, 'Ralph')


class FakeEmbedController:

  def decode(self, controller_state):
    return np.array(controller_state, copy=True) + 1000


class FakeDelayedAgent:

  def __init__(self, batch_size: int, names: list[str]):
    self.batch_size = batch_size
    self.names = list(names)
    self.embed_controller = FakeEmbedController()
    self.started = 0
    self.stopped = 0
    self.step_calls = 0
    self.base = ord(self.names[0][0])

  def start(self):
    self.started += 1

  def stop(self):
    self.stopped += 1

  def step_undelayed(self, game, needs_reset):
    del game, needs_reset
    self.step_calls += 1
    controller = np.full(
        [self.batch_size],
        self.base + self.step_calls,
        dtype=np.int32,
    )
    return SampleOutputs(controller_state=controller, logits=controller)


class LocalDiscordGpuInferenceServerTest(unittest.TestCase):

  def setUp(self):
    self.fake_state = {
        'name_map': {'A': 0, 'B': 1},
        'config': {'policy': {'delay': 18}},
        'agent_config': {'path': 'unused'},
    }
    self.created_agents = []

    def build_agent(*, batch_size, name, **kwargs):
      del kwargs
      agent = FakeDelayedAgent(batch_size=batch_size, names=name)
      self.created_agents.append(agent)
      return agent

    self.load_state_patcher = mock.patch.object(
        discordbot.eval_lib, 'load_state', return_value=self.fake_state)
    self.build_agent_patcher = mock.patch.object(
        discordbot.eval_lib, 'build_delayed_agent', side_effect=build_agent)
    self.load_state_patcher.start()
    self.build_agent_patcher.start()
    self.addCleanup(self.load_state_patcher.stop)
    self.addCleanup(self.build_agent_patcher.stop)

  def _server(self):
    return discordbot.LocalDiscordGpuInferenceServer(
        model_path='unused',
        console_delay=15,
        max_sessions=4,
        max_local_bots=3,
        batch_window_ms=2.0,
        compile=True,
        jit_compile=True,
        sample_temperature=1.0,
        batch_steps=0,
    )

  def test_register_session_prewarms_agent_before_infer(self):
    server = self._server()
    self.addCleanup(server.close)

    server.register_session('s1', ['A', 'A', 'A'])
    session_agent = server._session_agents['s1'].agent
    self.assertEqual(session_agent.started, 1)
    self.assertEqual(session_agent.step_calls, 3)

    game = discordbot._make_dummy_raw_game(3)
    result = server.infer('s1', game, np.zeros([3], dtype=np.bool_))

    self.assertEqual(session_agent.step_calls, 4)
    np.testing.assert_array_equal(
        result,
        np.full([3], ord('A') + 1004, dtype=np.int32),
    )

    server.unregister_session('s1')
    self.assertEqual(session_agent.stopped, 1)

  def test_sessions_use_independent_agents(self):
    server = self._server()
    self.addCleanup(server.close)
    game = discordbot._make_dummy_raw_game(1)
    needs_reset = np.zeros([1], dtype=np.bool_)

    server.register_session('s1', ['A'])
    server.register_session('s2', ['B'])

    out1 = server.infer('s1', game, needs_reset)
    out2 = server.infer('s2', game, needs_reset)
    agent1 = server._session_agents['s1'].agent
    agent2 = server._session_agents['s2'].agent

    self.assertIsNot(agent1, agent2)
    self.assertEqual(agent1.step_calls, 4)
    self.assertEqual(agent2.step_calls, 4)
    self.assertEqual(int(out1[0]), ord('A') + 1004)
    self.assertEqual(int(out2[0]), ord('B') + 1004)

  def test_duplicate_session_is_rejected(self):
    server = self._server()
    self.addCleanup(server.close)

    server.register_session('s1', ['A'])
    with self.assertRaisesRegex(ValueError, 'already registered'):
      server.register_session('s1', ['A'])


class DiscordGcTest(unittest.IsolatedAsyncioTestCase):

  async def test_gc_does_not_treat_menu_wait_as_stale_in_game(self):
    bot = object.__new__(discordbot.DiscordBot)
    bot.lock = discordbot.threading.RLock()
    bot._sessions = {}
    bot._menu_timeout = 3
    bot._stop_sessions = mock.Mock()

    session = mock.Mock()
    session.status.return_value = {
        'is_alive': True,
        'num_menu_frames': 60,
        'has_entered_game': True,
        'in_menu': True,
        'seconds_since_activity': 10,
        'seconds_since_frame_advance': discordbot.STALL_TIMEOUT_SECONDS + 30,
    }
    info = discordbot.SessionInfo(
        session=session,
        start_time=discordbot.datetime.datetime.now(),
        discord_name='vet0_',
        discord_id=1,
        connect_code='BABA',
        agents={},
        team_colors={},
        playstyles={},
        bot_specs=[],
        launch_message=None,
    )
    bot._sessions[1] = info

    gc_infos = await discordbot.DiscordBot._gc_sessions(bot)

    self.assertEqual(gc_infos, [])
    bot._stop_sessions.assert_called_once_with([])


if __name__ == '__main__':
  unittest.main(failfast=True)
