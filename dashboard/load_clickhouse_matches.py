#!/usr/bin/env python3
"""Utility for loading melee match JSONL data into ClickHouse.

Usage example (dev server):
    source .linuxvenv/bin/activate
    python dashboard/load_clickhouse_matches.py \
        --input-path melee_data/testdata.jsonl \
        --host gigaserver --database dashboard --create-schema

After ingestion you can plug Grafana directly into ClickHouse and query either
`dashboard.match_results` (match-level context) or `dashboard.match_players`
(per-player rows emitted by the materialized view) using `$__timeFilter`.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from clickhouse_driver import Client


DEFAULT_DB = "dashboard"
MATCH_TABLE = "match_results"
PLAYER_TABLE = "match_players"
MV_NAME = "match_players_mv"


@dataclass
class ClickHouseConfig:
    host: str
    port: int
    user: str
    password: Optional[str]
    database: str
    secure: bool = False

    def build_client(self) -> Client:
        return Client(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password or "",
            secure=self.secure,
            verify=False,
            database=self.database,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True,
                        help="Path to JSONL file containing match records")
    parser.add_argument("--host", default="gigaserver", help="ClickHouse host")
    parser.add_argument("--port", type=int, default=9000, help="ClickHouse port")
    parser.add_argument("--user", default="default", help="ClickHouse user")
    parser.add_argument("--password", default="", help="ClickHouse password")
    parser.add_argument("--database", default=DEFAULT_DB, help="Target database")
    parser.add_argument("--batch-size", type=int, default=500, help="Insert batch size")
    parser.add_argument("--create-schema", action="store_true",
                        help="Create database/tables/materialized view before loading")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse input only; do not write to ClickHouse")
    parser.add_argument("--use-ssl", action="store_true",
                        help="Connect with TLS (off by default)")
    return parser.parse_args()


def ensure_schema(cfg: ClickHouseConfig) -> None:
    client = cfg.build_client()
    db = cfg.database
    client.execute(f"CREATE DATABASE IF NOT EXISTS {db}")

    client.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {db}.{MATCH_TABLE} (
            match_id UUID,
            timestamp DateTime64(6, 'UTC'),
            ingested_at DateTime DEFAULT now(),
            mode LowCardinality(String),
            winner UInt8,
            team1_names Array(LowCardinality(String)),
            team1_characters Array(LowCardinality(String)),
            team2_names Array(LowCardinality(String)),
            team2_characters Array(LowCardinality(String)),
            team1_comp String,
            team2_comp String,
            team1_players UInt8,
            team2_players UInt8,
            is_doubles UInt8 MATERIALIZED multiIf(team1_players = 2 AND team2_players = 2, 1, 0)
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMMDD(timestamp)
        ORDER BY (timestamp, match_id)
        SETTINGS index_granularity = 8192
        """
    )

    client.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {db}.{PLAYER_TABLE} (
            match_id UUID,
            timestamp DateTime64(6, 'UTC'),
            mode LowCardinality(String),
            team_index UInt8,
            slot UInt8,
            player_name LowCardinality(String),
            character LowCardinality(String),
            won UInt8,
            team_comp String
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMMDD(timestamp)
        ORDER BY (timestamp, match_id, team_index, slot)
        SETTINGS index_granularity = 8192
        """
    )

    client.execute(
        f"""
        CREATE MATERIALIZED VIEW IF NOT EXISTS {db}.{MV_NAME}
        TO {db}.{PLAYER_TABLE}
        AS
        SELECT
            match_id,
            timestamp,
            mode,
            toUInt8(1) AS team_index,
            toUInt8(slot) AS slot,
            player_name,
            character,
            toUInt8(winner = 1) AS won,
            team1_comp AS team_comp
        FROM {db}.{MATCH_TABLE}
        ARRAY JOIN arrayEnumerate(team1_names) AS slot,
                  team1_names AS player_name,
                  team1_characters AS character

        UNION ALL

        SELECT
            match_id,
            timestamp,
            mode,
            toUInt8(2) AS team_index,
            toUInt8(slot) AS slot,
            player_name,
            character,
            toUInt8(winner = 2) AS won,
            team2_comp AS team_comp
        FROM {db}.{MATCH_TABLE}
        ARRAY JOIN arrayEnumerate(team2_names) AS slot,
                  team2_names AS player_name,
                  team2_characters AS character
        """
    )

    client.disconnect()


def load_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON on line {line_no}: {exc}") from exc


def parse_timestamp(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def canonical_comp(characters: List[str]) -> str:
    chars = [c.strip() for c in characters if c]
    return "+".join(sorted(chars)) if chars else ""


def normalize_match(raw: Dict[str, object]) -> Dict[str, object]:
    timestamp = parse_timestamp(str(raw["timestamp"]))
    winner = int(raw["winner"])
    mode = str(raw["mode"]).lower()

    team_payload = {}
    for team_index in (1, 2):
        names: List[str] = []
        chars: List[str] = []
        for slot in (1, 2):
            key = f"team{team_index}_player{slot}"
            player = raw.get(key)
            if not isinstance(player, dict):
                continue
            name = (player.get("name") or "").strip()
            character = (player.get("character") or "").strip()
            if not name and not character:
                continue
            names.append(name)
            chars.append(character)
        team_payload[team_index] = (names, chars)

    match_uuid = uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(raw, sort_keys=True))

    team1_names, team1_chars = team_payload.get(1, ([], []))
    team2_names, team2_chars = team_payload.get(2, ([], []))

    record = {
        "match_id": match_uuid,
        "timestamp": timestamp,
        "mode": mode,
        "winner": winner,
        "team1_names": team1_names,
        "team1_characters": team1_chars,
        "team2_names": team2_names,
        "team2_characters": team2_chars,
        "team1_comp": canonical_comp(team1_chars),
        "team2_comp": canonical_comp(team2_chars),
        "team1_players": len(team1_names),
        "team2_players": len(team2_names),
    }
    return record


def insert_matches(client: Client, database: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    columns = [
        "match_id",
        "timestamp",
        "mode",
        "winner",
        "team1_names",
        "team1_characters",
        "team2_names",
        "team2_characters",
        "team1_comp",
        "team2_comp",
        "team1_players",
        "team2_players",
    ]
    data = [[row[col] for col in columns] for row in rows]
    client.execute(
        f"INSERT INTO {database}.{MATCH_TABLE} ({', '.join(columns)}) VALUES",
        data,
    )


def main() -> int:
    args = parse_args()
    cfg = ClickHouseConfig(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
        secure=args.use_ssl,
    )

    if args.create_schema:
        ensure_schema(cfg)

    parsed_rows: List[Dict[str, object]] = []
    count = 0
    client: Optional[Client] = None
    if not args.dry_run:
        client = cfg.build_client()

    for raw in load_jsonl(args.input_path):
        row = normalize_match(raw)
        parsed_rows.append(row)
        if len(parsed_rows) >= args.batch_size:
            if not args.dry_run:
                assert client is not None
                insert_matches(client, cfg.database, parsed_rows)
            count += len(parsed_rows)
            parsed_rows.clear()

    if parsed_rows and not args.dry_run:
        assert client is not None
        insert_matches(client, cfg.database, parsed_rows)
        count += len(parsed_rows)
    elif parsed_rows and args.dry_run:
        count += len(parsed_rows)

    if client is not None:
        client.disconnect()

    print(f"Processed {count} matches from {args.input_path}")
    if args.dry_run:
        print("Dry run complete, nothing inserted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
