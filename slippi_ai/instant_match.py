import dataclasses
import os
import random
from typing import Any, Mapping, Sequence

import melee

_REMATCH_FALSE_HOOK_ADDRESS = 0x801A5C14
_REMATCH_TRUE_HOOK_ADDRESS = 0x801A5C20
_GECKO_NAME = 'slippi-ai: Instant Match Randomizer'

_COMPETITIVE_STAGE_POOL = (
    melee.Stage.BATTLEFIELD,
    melee.Stage.FINAL_DESTINATION,
    melee.Stage.DREAMLAND,
    melee.Stage.POKEMON_STADIUM,
    melee.Stage.YOSHIS_STORY,
    melee.Stage.FOUNTAIN_OF_DREAMS,
)

_CHARACTER_NAME_MAP = {
    ''.join(ch for ch in c.name.lower() if ch.isalnum()): c
    for c in melee.Character
}
_STAGE_NAME_MAP = {
    ''.join(ch for ch in s.name.lower() if ch.isalnum()): s
    for s in melee.Stage
}

_CONFLICT_HOOK_ADDRESSES = (
    _REMATCH_FALSE_HOOK_ADDRESS,
    _REMATCH_TRUE_HOOK_ADDRESS,
    0x801B15A0,
    0x801A5AC8,
    0x801A5E90,
    0x801A5EB4,
)

# Overwrite the pending-scene id with CSS (0), then return to the original flow.
_RETURN_TO_CSS_BYTES = bytes.fromhex('38600000')


@dataclasses.dataclass(frozen=True)
class InstantMatchConfig:
  character_pool: tuple[melee.Character, ...]
  stage_pool: tuple[melee.Stage, ...]
  starting_stocks: int

  def choose_initial_character(self) -> melee.Character:
    return random.choice(self.character_pool)

  def choose_initial_stage(self) -> melee.Stage:
    return random.choice(self.stage_pool)


def resolve_config(
    *,
    players: Mapping[int, Any],
    stage: melee.Stage,
    character_pool: Sequence[str] | None = None,
    stage_pool: Sequence[str] | None = None,
    starting_stocks: int = 0,
) -> InstantMatchConfig:
  if starting_stocks < 0 or starting_stocks > 99:
    raise ValueError('instant_match starting_stocks must be between 0 and 99.')

  resolved_characters = _parse_character_pool(character_pool)
  if character_pool is None and not resolved_characters:
    resolved_characters = _derive_character_pool(players)
  if character_pool is None and not resolved_characters:
    raise ValueError(
        'instant_match requires either --dolphin.instant_match_character_pool '
        'or AI players with configured characters.')

  resolved_stages = _parse_stage_pool(stage_pool)
  if not resolved_stages:
    resolved_stages = _derive_stage_pool(stage)

  return InstantMatchConfig(
      character_pool=tuple(resolved_characters),
      stage_pool=tuple(resolved_stages),
      starting_stocks=starting_stocks,
  )


def inject_gecko_codes(console: melee.Console, config: InstantMatchConfig):
  del config
  ini_path = os.path.join(console._get_dolphin_home_path(), 'GameSettings', 'GALE01r2.ini')
  with open(ini_path) as f:
    ini_text = f.read()
  ini_text = _upsert_gecko_code(
      ini_text,
      _GECKO_NAME,
      _build_gecko_code(),
  )
  with open(ini_path, 'w') as f:
    f.write(ini_text)


def _normalize_name(name: str) -> str:
  return ''.join(ch for ch in name.strip().lower() if ch.isalnum())


def _parse_character_pool(character_pool: Sequence[str] | None) -> list[melee.Character]:
  if not character_pool:
    return []
  result = []
  for name in character_pool:
    normalized = _normalize_name(name)
    normalized = normalized.replace('captainfalcon', 'cptfalcon')
    normalized = normalized.replace('younglink', 'ylink')
    normalized = normalized.replace('drmario', 'doc')
    normalized = normalized.replace('mrgameandwatch', 'gameandwatch')
    if normalized == 'gamewatch':
      normalized = 'gameandwatch'
    if normalized not in _CHARACTER_NAME_MAP:
      raise ValueError(f'Unknown instant_match character {name!r}.')
    result.append(_CHARACTER_NAME_MAP[normalized])
  return result


def _parse_stage_pool(stage_pool: Sequence[str] | None) -> list[melee.Stage]:
  if not stage_pool:
    return []
  result = []
  for name in stage_pool:
    normalized = _normalize_name(name)
    if normalized == 'dreamland64':
      normalized = 'dreamland'
    if normalized not in _STAGE_NAME_MAP:
      raise ValueError(f'Unknown instant_match stage {name!r}.')
    stage = _STAGE_NAME_MAP[normalized]
    if stage in (melee.Stage.RANDOM_STAGE, melee.Stage.NO_STAGE):
      raise ValueError(f'Invalid instant_match stage {name!r}.')
    result.append(stage)
  return result


def _derive_character_pool(players: Mapping[int, Any]) -> list[melee.Character]:
  pool: list[melee.Character] = []
  seen: set[melee.Character] = set()
  for player in players.values():
    if not hasattr(player, 'character'):
      continue
    weight_table = getattr(player, 'character_weight_table', None)
    characters = weight_table.keys() if weight_table else (player.character,)
    for character in characters:
      if character not in seen:
        seen.add(character)
        pool.append(character)
  return pool


def _derive_stage_pool(stage: melee.Stage) -> list[melee.Stage]:
  if stage == melee.Stage.RANDOM_STAGE:
    return list(_COMPETITIVE_STAGE_POOL)
  if stage == melee.Stage.NO_STAGE:
    raise ValueError('instant_match requires a concrete stage or RANDOM_STAGE.')
  return [stage]


def _upsert_gecko_code(ini_text: str, name: str, code_body: str) -> str:
  ini_text = _remove_conflicting_c2_blocks(ini_text, _CONFLICT_HOOK_ADDRESSES)
  enabled_line = f'${name}'
  if enabled_line not in ini_text:
    ini_text = ini_text.replace('[Gecko_Enabled]\n', f'[Gecko_Enabled]\n{enabled_line}\n', 1)
  block = f'${name}\n{code_body.rstrip()}\n'
  if block not in ini_text:
    if not ini_text.endswith('\n'):
      ini_text += '\n'
    ini_text += '\n' + block
  return ini_text


def _remove_conflicting_c2_blocks(ini_text: str, addresses: Sequence[int]) -> str:
  headers = {f'C2{address & 0xFFFFFF:06X} ' for address in addresses}
  output_lines: list[str] = []
  lines = ini_text.splitlines()
  i = 0
  while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    if stripped.startswith('$') and i + 1 < len(lines):
      next_line = lines[i + 1].strip()
      if any(next_line.startswith(header) for header in headers):
        i += 2
        while i < len(lines):
          next_stripped = lines[i].strip()
          if not next_stripped or next_stripped.startswith('$') or next_stripped.startswith('['):
            break
          i += 1
        continue
    output_lines.append(line)
    i += 1
  return '\n'.join(output_lines) + ('\n' if ini_text.endswith('\n') else '')


def _build_gecko_code() -> str:
  return '\n'.join([
      _format_c2_code(_REMATCH_FALSE_HOOK_ADDRESS, _RETURN_TO_CSS_BYTES),
      _format_c2_code(_REMATCH_TRUE_HOOK_ADDRESS, _RETURN_TO_CSS_BYTES),
  ])


def _format_c2_code(address: int, text_bytes: bytes) -> str:
  if len(text_bytes) % 4:
    raise ValueError('Gecko text payload must be word-aligned.')
  words = [
      int.from_bytes(text_bytes[i:i + 4], 'big')
      for i in range(0, len(text_bytes), 4)
  ]
  if len(words) % 2:
    words.append(0x60000000)
  header = f'C2{address & 0xFFFFFF:06X} {len(words) // 2:08X}'
  lines = [header]
  for i in range(0, len(words), 2):
    lines.append(f'{words[i]:08X} {words[i + 1]:08X}')
  return '\n'.join(lines)
