import dataclasses
import struct
import subprocess


RAW_HEADER = bytes([
    0x7B, 0x55, 0x03, 0x72, 0x61, 0x77, 0x5B, 0x24, 0x55, 0x23, 0x6C,
])


class SlpzDecodeError(ValueError):
  """Failed to decode a .slpz file."""


@dataclasses.dataclass(frozen=True)
class SlpzHeader:
  version: int
  event_sizes_offset: int
  game_start_offset: int
  metadata_offset: int
  compressed_events_offset: int
  events_size: int


_SLPZ_HEADER_LEN = 24


def _parse_header(data: bytes) -> SlpzHeader:
  if len(data) < _SLPZ_HEADER_LEN:
    raise SlpzDecodeError(f"slpz too small: {len(data)} bytes")

  # Big-endian u32 fields, matching https://github.com/project-slippi/slpz
  # (note: no magic string; offsets are absolute from the file start).
  fields = struct.unpack(">6I", data[:_SLPZ_HEADER_LEN])
  version = fields[0]
  event_sizes_offset = fields[1]
  game_start_offset = fields[2]
  metadata_offset = fields[3]
  compressed_events_offset = fields[4]
  events_size = fields[5]

  if version != 0:
    raise SlpzDecodeError(f"unsupported slpz version: {version}")

  if event_sizes_offset != _SLPZ_HEADER_LEN:
    raise SlpzDecodeError(
        f"unexpected event_sizes_offset: {event_sizes_offset} (expected {_SLPZ_HEADER_LEN})"
    )

  if not (event_sizes_offset <= game_start_offset <= metadata_offset <= compressed_events_offset <= len(data)):
    raise SlpzDecodeError("invalid slpz section offsets")

  return SlpzHeader(
      version=version,
      event_sizes_offset=event_sizes_offset,
      game_start_offset=game_start_offset,
      metadata_offset=metadata_offset,
      compressed_events_offset=compressed_events_offset,
      events_size=events_size,
  )


def _slice(data: bytes, *, offset: int, length: int, label: str) -> bytes:
  end = offset + length
  if offset < 0 or length < 0 or end > len(data):
    raise SlpzDecodeError(
        f"{label} section out of bounds: offset={offset} length={length} file={len(data)}"
    )
  return data[offset:end]


def _event_sizes_to_map(event_sizes_event: bytes) -> list[int]:
  # Matches the Rust implementation in https://github.com/project-slippi/slpz:
  # this is the SLP "Event Payloads" event (0x35), which begins with a single
  # byte indicating the payload length (not counting the command byte), and
  # then a list of (command: u8, payload_size: u16be) entries.
  if len(event_sizes_event) < 2:
    raise SlpzDecodeError(f"event_sizes too small: {len(event_sizes_event)} bytes")
  if event_sizes_event[0] != 0x35:
    raise SlpzDecodeError("event_sizes section does not start with 0x35")

  payload_len = event_sizes_event[1]
  total_len = payload_len + 1  # includes command byte
  if len(event_sizes_event) < total_len:
    raise SlpzDecodeError(
        f"event_sizes truncated: {len(event_sizes_event)} bytes (need {total_len})"
    )

  if payload_len < 1 or (payload_len - 1) % 3 != 0:
    raise SlpzDecodeError(f"invalid event_sizes payload_len={payload_len}")

  event_type_count = (payload_len - 1) // 3
  sizes: list[int] = [0] * 256
  for i in range(event_type_count):
    offset = i * 3 + 2
    cmd = event_sizes_event[offset]
    size = struct.unpack(">H", event_sizes_event[offset + 1:offset + 3])[0]
    sizes[cmd] = size

  return sizes


def _zstd_decompress(data: bytes) -> bytes:
  # Avoid adding a Python zstd dependency; use the system `zstd` binary.
  # `-q`: quiet, `-d`: decompress, `-c`: write to stdout.
  try:
    proc = subprocess.run(
        ["zstd", "-q", "-d", "-c"],
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
  except subprocess.CalledProcessError as e:
    raise SlpzDecodeError(f"zstd failed: {e.stderr.decode('utf-8', errors='replace')}") from e
  return proc.stdout


def _unorder_events(compressed_events: bytes, event_sizes: list[int]) -> bytes:
  decompressed = _zstd_decompress(compressed_events)
  if len(decompressed) < 4:
    raise SlpzDecodeError("compressed events payload too small")

  event_count = struct.unpack(">I", decompressed[:4])[0]
  if len(decompressed) < 4 + event_count:
    raise SlpzDecodeError("compressed events missing event order section")

  event_order = decompressed[4:4 + event_count]
  event_data = decompressed[4 + event_count:]

  events_per_command = [0] * 256
  for cmd in event_order:
    events_per_command[cmd] += 1

  command_data_lengths = [
      events_per_command[cmd] * event_sizes[cmd] for cmd in range(256)
  ]

  command_data_offsets = [0] * 256
  running = 0
  for cmd in range(256):
    command_data_offsets[cmd] = running
    running += command_data_lengths[cmd]

  if len(event_data) < running:
    raise SlpzDecodeError("compressed events payload truncated")

  events_by_command: list[list[bytes]] = [[] for _ in range(256)]
  for cmd in range(256):
    per_event_size = event_sizes[cmd]
    if per_event_size == 0:
      continue
    cmd_event_count = events_per_command[cmd]
    if cmd_event_count == 0:
      continue

    cmd_offset = command_data_offsets[cmd]
    for event_idx in range(cmd_event_count):
      payload = bytearray(per_event_size)
      for byte_pos in range(per_event_size):
        data_idx = cmd_offset + byte_pos * cmd_event_count + event_idx
        payload[byte_pos] = event_data[data_idx]
      events_by_command[cmd].append(bytes(payload))

    # We'll pop() during reconstruction; reverse so we emit events in order.
    events_by_command[cmd].reverse()

  out = bytearray()
  out_extend = out.extend
  for cmd in event_order:
    out.append(cmd)
    per_event_size = event_sizes[cmd]
    if per_event_size == 0:
      continue
    payloads = events_by_command[cmd]
    if not payloads:
      raise SlpzDecodeError(f"missing payloads for command {cmd}")
    out_extend(payloads.pop())
  return bytes(out)


def decompress_slpz_bytes(data: bytes) -> bytes:
  """Convert a .slpz file (bytes) into a standard .slp file (bytes)."""
  header = _parse_header(data)

  event_sizes_event = data[header.event_sizes_offset:header.game_start_offset]
  game_start_event = data[header.game_start_offset:header.metadata_offset]
  metadata_event = data[header.metadata_offset:header.compressed_events_offset]
  compressed_events = data[header.compressed_events_offset:]

  sizes = _event_sizes_to_map(event_sizes_event)
  unordered_events = _unorder_events(compressed_events, sizes)

  result = bytearray()
  result.extend(RAW_HEADER)
  result.extend(b"\x00\x00\x00\x00")  # raw length placeholder
  result.extend(event_sizes_event)
  result.extend(game_start_event)
  result.extend(unordered_events)
  metadata_offset_in_slp = len(result)
  result.extend(metadata_event)

  # raw_len counts bytes after the 11-byte RAW_HEADER and 4-byte raw_len field.
  raw_len = metadata_offset_in_slp - 15
  result[len(RAW_HEADER):len(RAW_HEADER) + 4] = struct.pack(">I", raw_len)
  return bytes(result)
