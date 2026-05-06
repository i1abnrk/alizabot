import os
import sqlite3
from typing import Dict, Iterable, List, Tuple

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;
PRAGMA mmap_size=30000000000;

CREATE TABLE IF NOT EXISTS tokens (
	id INTEGER PRIMARY KEY,
	text TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS cooccurrence (
	token_id INTEGER NOT NULL,
	neighbor_id INTEGER NOT NULL,
	distance INTEGER NOT NULL CHECK(distance BETWEEN 1 AND 5),
	count INTEGER NOT NULL,
	PRIMARY KEY (token_id, neighbor_id, distance),
	FOREIGN KEY(token_id) REFERENCES tokens(id) ON DELETE CASCADE,
	FOREIGN KEY(neighbor_id) REFERENCES tokens(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_co_token_distance ON cooccurrence(token_id, distance);
CREATE INDEX IF NOT EXISTS idx_co_neighbor ON cooccurrence(neighbor_id);
"""


def ensure_parent_dir(path: str) -> None:
	parent = os.path.dirname(os.path.abspath(path))
	if parent and not os.path.exists(parent):
		os.makedirs(parent, exist_ok=True)


def connect_db(db_path: str) -> sqlite3.Connection:
	ensure_parent_dir(db_path)
	conn = sqlite3.connect(db_path)
	conn.execute("PRAGMA foreign_keys=ON;")
	return conn


def init_schema(conn: sqlite3.Connection) -> None:
	conn.executescript(SCHEMA_SQL)
	conn.commit()


def get_or_create_token_ids(
	conn: sqlite3.Connection, tokens: Iterable[str]
) -> Dict[str, int]:
	unique = list(dict.fromkeys(tokens))
	if not unique:
		return {}
	with conn:
		conn.executemany("INSERT OR IGNORE INTO tokens(text) VALUES (?)", ((t,) for t in unique))
	cur = conn.execute(
		"SELECT text, id FROM tokens WHERE text IN ({})".format(",".join("?" * len(unique))),
		unique,
	)
	return {row[0]: row[1] for row in cur.fetchall()}


def upsert_cooccurrence_batch(
	conn: sqlite3.Connection, rows: Iterable[Tuple[int, int, int, int]]
) -> None:
	# rows: (token_id, neighbor_id, distance, count_delta)
	with conn:
		conn.executemany(
			"""
			INSERT INTO cooccurrence(token_id, neighbor_id, distance, count)
			VALUES (?, ?, ?, ?)
			ON CONFLICT(token_id, neighbor_id, distance)
			DO UPDATE SET count = count + excluded.count
			""",
			rows,
		)

