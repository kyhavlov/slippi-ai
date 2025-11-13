import unittest

import numpy as np
import multiprocessing as mp

from slippi_ai import envs


class SinglesChunkMergeTest(unittest.TestCase):

  def test_merge_scalar_outputs(self):
    left = envs.EnvOutput({1: 'l1', 2: 'l2'}, False, None)
    right = envs.EnvOutput({3: 'r3', 4: 'r4'}, True, None)

    merged = envs._merge_singles_env_outputs(left, right)

    self.assertEqual(
        merged.gamestates,
        {1: 'l1', 2: 'l2', 3: 'r3', 4: 'r4'},
    )
    self.assertTrue(merged.needs_reset)
    self.assertFalse(merged.needs_reset_ports[1])
    self.assertTrue(merged.needs_reset_ports[3])

  def test_merge_array_outputs(self):
    left_needs = np.array([False, True])
    right_needs = np.array([True, False])

    left = envs.EnvOutput({1: 'l'}, left_needs, {1: left_needs})
    right = envs.EnvOutput({3: 'r'}, right_needs, None)

    merged = envs._merge_singles_env_outputs(left, right)

    np.testing.assert_array_equal(
        merged.needs_reset,
        np.array([True, True]),
    )
    np.testing.assert_array_equal(merged.needs_reset_ports[1], left_needs)
    np.testing.assert_array_equal(merged.needs_reset_ports[3], right_needs)

  def test_chunk_recv_batch_outputs(self):
    class _StubEnv:

      def __init__(self, outputs):
        self._outputs = list(outputs)

      def recv(self):
        return self._outputs.pop(0)

    left_batch = [
        envs.EnvOutput({1: 'l0'}, False, None),
        envs.EnvOutput({1: 'l1'}, True, None),
    ]
    right_batch = [
        envs.EnvOutput({3: 'r0'}, False, None),
        envs.EnvOutput({3: 'r1'}, False, None),
    ]

    chunk = envs._SinglesChunk.__new__(envs._SinglesChunk)
    chunk._left_env = _StubEnv([left_batch])
    chunk._right_env = _StubEnv([right_batch])
    chunk._left_conn = None
    chunk._right_conn = None

    merged = chunk.recv()

    self.assertIsInstance(merged, list)
    self.assertEqual(len(merged), 2)
    self.assertEqual(
        merged[0].gamestates,
        {1: 'l0', 3: 'r0'},
    )
    self.assertTrue(merged[1].needs_reset)

  def test_chunk_recv_skips_empty_batches(self):
    class _StubEnv:

      def __init__(self, outputs):
        self._outputs = list(outputs)

      def recv(self):
        return self._outputs.pop(0)

    left_batches = [
        [],
        [envs.EnvOutput({1: 'l'}, False, {1: False})],
    ]
    right_batches = [
        [],
        [envs.EnvOutput({3: 'r'}, False, {3: False})],
    ]

    chunk = envs._SinglesChunk.__new__(envs._SinglesChunk)
    chunk._left_env = _StubEnv(left_batches)
    chunk._right_env = _StubEnv(right_batches)
    chunk._left_conn = None
    chunk._right_conn = None

    merged = chunk.recv()

    self.assertIsInstance(merged, list)
    self.assertEqual(len(merged), 1)
    self.assertEqual(merged[0].gamestates, {1: 'l', 3: 'r'})

  def test_chunk_send_handles_batched_controllers(self):
    class _SinkEnv:

      def __init__(self):
        self.payloads = []

      def send(self, controllers):
        self.payloads.append(controllers)

      def recv(self):
        raise AssertionError('recv should not be called')

      def begin_stop(self):
        pass

      def ensure_stopped(self):
        pass

    left_env = _SinkEnv()
    right_env = _SinkEnv()
    chunk = envs._SinglesChunk(left_env, right_env)

    controllers = [
        {1: 'a', 2: 'b', 3: 'c', 4: 'd'},
        {1: 'e', 2: 'f', 3: 'g', 4: 'h'},
    ]

    chunk.send(controllers)

    self.assertEqual(left_env.payloads, [[{1: 'a', 2: 'b'}, {1: 'e', 2: 'f'}]])
    self.assertEqual(right_env.payloads, [[{3: 'c', 4: 'd'}, {3: 'g', 4: 'h'}]])

  def test_chunk_recv_uses_connection_wait(self):
    def make_conn_env(payloads):
      parent_conn, child_conn = mp.Pipe()
      for payload in payloads:
        child_conn.send(payload)

      class _ConnEnv:
        connection = parent_conn
        def recv(self_inner):
          return parent_conn.recv()
        def send(self_inner, controllers):
          pass
        def begin_stop(self_inner):
          pass
        def ensure_stopped(self_inner):
          pass

      return _ConnEnv()

    left_payload = envs.EnvOutput({1: 'L'}, False, {1: False})
    right_payload = envs.EnvOutput({3: 'R'}, False, {3: False})

    chunk = envs._SinglesChunk(
        make_conn_env([left_payload]),
        make_conn_env([right_payload]),
    )

    merged = chunk.recv()
    self.assertEqual(merged.gamestates, {1: 'L', 3: 'R'})


if __name__ == '__main__':
  unittest.main()
