import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Sequence, Tuple

# SQL Injection Protection: parameterized queries only - user input never touches SQL directly

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

-- Convenience table for fast bayes_weight / relevance lookups
-- Populated at index time from cooccurrence
CREATE TABLE IF NOT EXISTS token_pair_stats (
    token_a INTEGER NOT NULL,
    token_b INTEGER NOT NULL,
    distance_sum REAL NOT NULL,           -- sum(distance * count) for directed pair (a as neighbor, b as target)
    PRIMARY KEY (token_a, token_b)
);

CREATE INDEX IF NOT EXISTS idx_token_pair_b ON token_pair_stats(token_b);
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


def build_token_pair_stats(conn: sqlite3.Connection) -> None:
    """Populate token_pair_stats from existing cooccurrence data.
    This creates fast lookup rows for bayes_weight calculations.
    Should be called after major indexing / rebuild.
    """
    conn.execute("DELETE FROM token_pair_stats")

    conn.execute("""
        INSERT INTO token_pair_stats (token_a, token_b, distance_sum)
        SELECT neighbor_id, token_id, SUM(distance * count)
        FROM cooccurrence
        GROUP BY neighbor_id, token_id
    """)
    conn.commit()


def safe_execute(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> sqlite3.Cursor:
	return conn.execute(sql, params)


def get_or_create_token_ids(
	conn: sqlite3.Connection, tokens: Iterable[str]
) -> Dict[str, int]:
	"""Safe version: batches the lookup to avoid SQLite variable limit."""
	unique = list(dict.fromkeys(tokens))
	if not unique:
		return {}

	result: Dict[str, int] = {}

	# First, insert all (safe, executemany handles large lists)
	with conn:
		conn.executemany("INSERT OR IGNORE INTO tokens(text) VALUES (?)", ((t,) for t in unique))

	# Then fetch in safe batches (SQLite typically limits ~999 variables per query)
	BATCH = 400
	for i in range(0, len(unique), BATCH):
		batch = unique[i : i + BATCH]
		if not batch:
			continue
		placeholders = ",".join("?" * len(batch))
		cur = conn.execute(
			f"SELECT text, id FROM tokens WHERE text IN ({placeholders})",
			batch,
		)
		for text, tid in cur.fetchall():
			result[text] = tid

	return result


def get_or_create_token_id(conn: sqlite3.Connection, token: str) -> int:
	"""Single token version for streaming use cases. Returns the integer id."""
	with conn:
		conn.execute("INSERT OR IGNORE INTO tokens(text) VALUES (?)", (token,))
	row = conn.execute("SELECT id FROM tokens WHERE text = ?", (token,)).fetchone()
	return int(row[0])


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



