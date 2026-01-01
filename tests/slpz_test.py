import os
import struct
import subprocess
import tempfile
import unittest

from slippi_db import slpz as slpz_lib
from slippi_db import utils as db_utils


def _zstd_compress(data: bytes) -> bytes:
  proc = subprocess.run(
      ["zstd", "-q", "-c"],
      input=data,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      check=True,
  )
  return proc.stdout


def _make_event_sizes_event(sizes: dict[int, int]) -> bytes:
  entries = sorted(sizes.items())
  payload_len = 1 + 3 * len(entries)
  out = bytearray()
  out.append(0x35)  # Event Payloads
  out.append(payload_len)
  for cmd, size in entries:
    out.append(cmd)
    out.extend(struct.pack(">H", size))
  return bytes(out)


def _make_slpz(
    *,
    event_sizes_event: bytes,
    game_start_event: bytes,
    metadata_event: bytes,
    unordered_events: list[tuple[int, bytes]],
    sizes: dict[int, int],
) -> bytes:
  event_order = bytes([cmd for cmd, _ in unordered_events])

  by_cmd: list[list[bytes]] = [[] for _ in range(256)]
  for cmd, payload in unordered_events:
    self_size = sizes.get(cmd, 0)
    if len(payload) != self_size:
      raise ValueError(f"payload len mismatch for {cmd}: {len(payload)} != {self_size}")
    by_cmd[cmd].append(payload)

  event_data = bytearray()
  for cmd in range(256):
    per_event_size = sizes.get(cmd, 0)
    if per_event_size == 0:
      continue
    events = by_cmd[cmd]
    if not events:
      continue
    for byte_pos in range(per_event_size):
      for payload in events:
        event_data.append(payload[byte_pos])

  decompressed_events = struct.pack(">I", len(event_order)) + event_order + bytes(event_data)
  compressed_events = _zstd_compress(decompressed_events)

  header_len = 24
  event_sizes_offset = header_len
  game_start_offset = event_sizes_offset + len(event_sizes_event)
  metadata_offset = game_start_offset + len(game_start_event)
  compressed_events_offset = metadata_offset + len(metadata_event)
  events_size = len(decompressed_events)  # matches the format field but not needed for decoding

  header = struct.pack(
      ">6I",
      0,  # version
      event_sizes_offset,
      game_start_offset,
      metadata_offset,
      compressed_events_offset,
      events_size,
  )

  return header + event_sizes_event + game_start_event + metadata_event + compressed_events


class SlpzTest(unittest.TestCase):

  def test_decompress_round_trip_matches_expected_layout(self):
    # Two commands with different payload sizes.
    sizes = {0x10: 2, 0x20: 3}
    event_sizes_event = _make_event_sizes_event(sizes)

    game_start_event = b"\x36GAME_START"
    metadata_event = b"\x37META"

    unordered_events = [
        (0x10, b"AB"),
        (0x20, b"XYZ"),
        (0x10, b"CD"),
    ]

    slpz_bytes = _make_slpz(
        event_sizes_event=event_sizes_event,
        game_start_event=game_start_event,
        metadata_event=metadata_event,
        unordered_events=unordered_events,
        sizes=sizes,
    )

    slp_bytes = slpz_lib.decompress_slpz_bytes(slpz_bytes)

    self.assertTrue(slp_bytes.startswith(slpz_lib.RAW_HEADER))
    raw_len = struct.unpack(">I", slp_bytes[11:15])[0]
    # raw_len measures the bytes between the raw_len field and the metadata.
    metadata_offset = 15 + raw_len
    self.assertLess(metadata_offset, len(slp_bytes))

    payload = slp_bytes[15:]
    expected_unordered = b"".join(bytes([cmd]) + p for cmd, p in unordered_events)
    self.assertEqual(
        event_sizes_event + game_start_event + expected_unordered + metadata_event,
        payload,
    )

  def test_localfile_wrapper_reads_as_slp_bytes(self):
    sizes = {0x10: 1}
    event_sizes_event = _make_event_sizes_event(sizes)
    game_start_event = b"\x36GS"
    metadata_event = b"\x37MD"
    unordered_events = [(0x10, b"A")]

    slpz_bytes = _make_slpz(
        event_sizes_event=event_sizes_event,
        game_start_event=game_start_event,
        metadata_event=metadata_event,
        unordered_events=unordered_events,
        sizes=sizes,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "test.slpz")
      with open(path, "wb") as f:
        f.write(slpz_bytes)

      lf = db_utils.SlpzFile(tmpdir, "test.slpz")
      slp_bytes = lf.read()

    self.assertTrue(slp_bytes.startswith(slpz_lib.RAW_HEADER))


if __name__ == "__main__":
  unittest.main(failfast=True)
