import unittest
import copy
import numpy as np

from melee import Character

from slippi_ai import dolphin as dolphin_lib
from slippi_ai import eval_lib, evaluators
from slippi_ai.rl import run_lib


class BasicNameLayoutTest(unittest.TestCase):

  def test_basic_name_layout_is_deterministic(self):
    names = ['Master Player', 'Cody', 'Dragunov', 'Darkatma']
    layout_a = run_lib._basic_name_layout(names, 8, 4, 17)
    layout_b = run_lib._basic_name_layout(names, 8, 4, 17)
    layout_c = run_lib._basic_name_layout(names, 8, 4, 18)

    self.assertEqual(layout_a, layout_b)
    self.assertNotEqual(layout_a, layout_c)
    self.assertEqual(len(layout_a), 8)
    self.assertTrue(all(len(slot_names) == 4 for slot_names in layout_a))

  def test_basic_name_layout_supports_all_env_widths(self):
    names = ['Master Player', 'Cody', 'Dragunov']

    singles = run_lib._basic_name_layout(names, 3, 2, 1)
    twovone = run_lib._basic_name_layout(names, 3, 3, 2)
    doubles = run_lib._basic_name_layout(names, 3, 4, 3)

    self.assertTrue(all(len(slot_names) == 2 for slot_names in singles))
    self.assertTrue(all(len(slot_names) == 3 for slot_names in twovone))
    self.assertTrue(all(len(slot_names) == 4 for slot_names in doubles))

  def test_port_name_batches_follow_layout_order(self):
    env_name_layout = [
        ('A', 'B', 'C'),
        ('D', 'E', 'F'),
    ]
    batches = run_lib._port_name_batches_from_layout(
        env_name_layout, (1, 3, 4))

    self.assertEqual(batches, {
        1: ['A', 'D'],
        3: ['B', 'E'],
        4: ['C', 'F'],
    })


class BasicSchedulingFakeEnvSmokeTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    demo_state = eval_lib.load_state(path='slippi_ai/data/checkpoints/demo')
    cls.state = copy.deepcopy(demo_state)
    cls.state['config'] = copy.deepcopy(demo_state['config'])
    cls.state['config']['max_names'] = 4
    cls.state['name_map'] = {
        'Master Player': 0,
        'Cody': 1,
        'Dragunov': 2,
        'Darkatma': 3,
    }
    cls.character_pool = {
        Character.FOX: 1,
        Character.FALCO: 1,
    }

  def _build_worker(
      self,
      active_ports: tuple[int, ...],
      env_name_layout: list[tuple[str, ...]],
  ) -> evaluators.RolloutWorker:
    players = {
        port: dolphin_lib.AI(character_weight_table=self.character_pool)
        for port in active_ports
    }
    agent_kwargs = {}
    for idx, port in enumerate(active_ports):
      agent_kwargs[port] = dict(
          state=self.state,
          name=[slot_names[idx] for slot_names in env_name_layout],
          compile=False,
          batch_steps=0,
      )
    return evaluators.RolloutWorker(
        agent_kwargs=agent_kwargs,
        dolphin_kwargs=dict(players=players, online_delay=0),
        num_envs=len(env_name_layout),
        use_fake_envs=True,
        use_gpu=False,
        agent_names=env_name_layout,
    )

  def _assert_history_matches_layout(
      self,
      histories: list[list[dict]],
      env_name_layout: list[tuple[str, ...]],
      width: int,
  ) -> None:
    self.assertEqual(len(histories), len(env_name_layout))
    for env_index, history in enumerate(histories):
      self.assertGreaterEqual(len(history), 2)
      expected_names = tuple(env_name_layout[env_index])
      self.assertTrue(all(item['names'] == expected_names for item in history))
      unique_characters = {item['characters'] for item in history}
      self.assertGreater(len(unique_characters), 1)
      self.assertTrue(all(len(chars) == width for chars in unique_characters))

  def test_fake_env_smoke_doubles(self):
    env_name_layout = run_lib._basic_name_layout(
        ['Master Player', 'Cody', 'Dragunov', 'Darkatma'],
        num_envs=3,
        slots_per_env=4,
        layout_seed=7,
    )
    worker = self._build_worker((1, 2, 3, 4), env_name_layout)
    with worker.run():
      worker.rollout(16)
      histories = worker._env.debug_assignment_history()
    self._assert_history_matches_layout(histories, env_name_layout, 4)

  def test_fake_env_smoke_singles(self):
    env_name_layout = run_lib._basic_name_layout(
        ['Master Player', 'Cody', 'Dragunov'],
        num_envs=3,
        slots_per_env=2,
        layout_seed=11,
    )
    worker = self._build_worker((1, 2), env_name_layout)
    with worker.run():
      worker.rollout(16)
      histories = worker._env.debug_assignment_history()
    self._assert_history_matches_layout(histories, env_name_layout, 2)

  def test_fake_env_smoke_twovone(self):
    env_name_layout = run_lib._basic_name_layout(
        ['Master Player', 'Cody', 'Dragunov', 'Darkatma'],
        num_envs=3,
        slots_per_env=3,
        layout_seed=13,
    )
    worker = self._build_worker((1, 2, 3), env_name_layout)
    with worker.run():
      worker.rollout(16)
      histories = worker._env.debug_assignment_history()
    self._assert_history_matches_layout(histories, env_name_layout, 3)


class MixedFusedRolloutWorkerTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    demo_state = eval_lib.load_state(path='slippi_ai/data/checkpoints/demo')
    cls.state = copy.deepcopy(demo_state)
    cls.state['config'] = copy.deepcopy(demo_state['config'])
    cls.state['config']['max_names'] = 4
    cls.state['name_map'] = {
        'Master Player': 0,
        'Cody': 1,
        'Dragunov': 2,
        'Darkatma': 3,
    }
    cls.character_pool = {
        Character.FOX: 1,
        Character.FALCO: 1,
    }

  def _make_group_spec(
      self,
      *,
      label: str,
      ports: tuple[int, ...],
      env_name_layout: list[tuple[str, ...]],
      enable_singles: bool = False,
  ) -> evaluators.RolloutGroupSpec:
    players = {
        port: dolphin_lib.AI(character_weight_table=self.character_pool)
        for port in ports
    }
    agent_kwargs = {}
    for idx, port in enumerate(ports):
      agent_kwargs[port] = dict(
          state=self.state,
          name=[slot_names[idx] for slot_names in env_name_layout],
          fake=True,
          compile=False,
          batch_steps=0,
      )
    return evaluators.RolloutGroupSpec(
        label=label,
        ports=ports,
        agent_kwargs=agent_kwargs,
        dolphin_kwargs=dict(players=players, online_delay=0),
        num_envs=len(env_name_layout),
        async_envs=False,
        env_kwargs=dict(enable_singles=enable_singles, swap_ports=False),
        use_gpu=False,
        use_fake_envs=True,
        use_ray_envs=False,
        agent_names=env_name_layout,
        scheduler=None,
    )

  def _make_specs(self) -> list[evaluators.RolloutGroupSpec]:
    return [
        self._make_group_spec(
            label='doubles',
            ports=(1, 2, 3, 4),
            env_name_layout=run_lib._basic_name_layout(
                ['Master Player', 'Cody', 'Dragunov', 'Darkatma'],
                num_envs=2,
                slots_per_env=4,
                layout_seed=7,
            ),
        ),
        self._make_group_spec(
            label='twovone',
            ports=(1, 2, 3),
            env_name_layout=run_lib._basic_name_layout(
                ['Master Player', 'Cody', 'Dragunov', 'Darkatma'],
                num_envs=2,
                slots_per_env=3,
                layout_seed=11,
            ),
        ),
        self._make_group_spec(
            label='singles',
            ports=(1, 2),
            env_name_layout=run_lib._basic_name_layout(
                ['Master Player', 'Cody', 'Dragunov', 'Darkatma'],
                num_envs=2,
                slots_per_env=2,
                layout_seed=13,
            ),
            enable_singles=True,
        ),
    ]

  def _assert_nested_equal(self, a, b):
    if isinstance(a, dict):
      self.assertEqual(set(a), set(b))
      for key in a:
        self._assert_nested_equal(a[key], b[key])
      return
    if hasattr(a, '_fields'):
      self.assertEqual(a._fields, b._fields)
      for field in a._fields:
        self._assert_nested_equal(getattr(a, field), getattr(b, field))
      return
    if isinstance(a, (list, tuple)):
      self.assertEqual(len(a), len(b))
      for left, right in zip(a, b):
        self._assert_nested_equal(left, right)
      return
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

  def test_mixed_fused_matches_legacy_mixed_worker_on_fake_envs(self):
    specs = self._make_specs()

    legacy_workers = [
        run_lib.TrajectoryRolloutWorker(
            evaluators.RolloutWorker(
                agent_kwargs=spec.agent_kwargs,
                dolphin_kwargs=spec.dolphin_kwargs,
                num_envs=spec.num_envs,
                async_envs=spec.async_envs,
                env_kwargs=spec.env_kwargs,
                use_gpu=spec.use_gpu,
                damage_ratio=spec.damage_ratio,
                use_fake_envs=spec.use_fake_envs,
                use_ray_envs=spec.use_ray_envs,
                agent_names=spec.agent_names,
                scheduler=spec.scheduler,
                fuse_ports_inference=True,
            ),
            ports=spec.ports,
            label=spec.label,
        )
        for spec in specs
    ]
    legacy = run_lib.MixedTrajectoryRolloutWorker(legacy_workers)
    mixed = evaluators.MixedFusedRolloutWorker(specs)

    try:
      legacy.start()
      mixed.start()
      legacy_traj, legacy_metrics = legacy.rollout(16)
      mixed_traj, mixed_metrics = mixed.rollout(16)
    finally:
      legacy.stop()
      mixed.stop()

    self._assert_nested_equal(legacy_traj, mixed_traj)
    self._assert_nested_equal(
        legacy_metrics['unexpected_reset'],
        mixed_metrics['unexpected_reset'],
    )


if __name__ == '__main__':
  unittest.main(failfast=True)
