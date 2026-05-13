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


def safe_execute(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> sqlite3.Cursor:
	return conn.execute(sql, params)


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


# ---------------------------------------------------------------------------
# Live console v0.1.2 — logical schema (token, count / from, to, distance,
# occurrences) is stored in ``chat_*`` tables so this file can coexist with the
# corpus indexer ``tokens`` / ``cooccurrence`` layout above.
# ---------------------------------------------------------------------------

CHAT_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS chat_tokens (
	id INTEGER PRIMARY KEY,
	token TEXT UNIQUE NOT NULL,
	count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chat_cooccurrence (
	from_token_id INTEGER NOT NULL,
	to_token_id INTEGER NOT NULL,
	distance INTEGER NOT NULL CHECK(distance BETWEEN 1 AND 5),
	occurrences INTEGER NOT NULL DEFAULT 0,
	PRIMARY KEY (from_token_id, to_token_id, distance)
);

CREATE INDEX IF NOT EXISTS idx_chat_co_from_dist ON chat_cooccurrence(from_token_id, distance);
"""


def ensure_parent_path(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def chat_connection(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
	ensure_parent_path(db_path)
	conn = sqlite3.connect(db_path)
	conn.execute("PRAGMA foreign_keys=ON;")
	try:
		conn.executescript(CHAT_SCHEMA_SQL)
		yield conn
		conn.commit()
	finally:
		conn.close()


def init_chat_schema(conn: sqlite3.Connection) -> None:
	conn.executescript(CHAT_SCHEMA_SQL)
	conn.commit()


def get_or_create_chat_token(conn: sqlite3.Connection, token: str) -> int:
	conn.execute(
		"INSERT OR IGNORE INTO chat_tokens(token, count) VALUES (?, 0)",
		(token,),
	)
	conn.execute("UPDATE chat_tokens SET count = count + 1 WHERE token = ?", (token,))
	row = conn.execute("SELECT id FROM chat_tokens WHERE token = ?", (token,)).fetchone()
	assert row is not None
	return int(row[0])


def chat_token_id(conn: sqlite3.Connection, token: str) -> Optional[int]:
	row = conn.execute("SELECT id FROM chat_tokens WHERE token = ?", (token,)).fetchone()
	return int(row[0]) if row else None


def update_chat_cooccurrence(
	conn: sqlite3.Connection,
	from_id: int,
	to_id: int,
	dist: int,
	increment: int = 1,
) -> None:
	if not (1 <= dist <= 5):
		raise ValueError("distance must be 1..5")
	conn.execute(
		"""
		INSERT OR IGNORE INTO chat_cooccurrence(from_token_id, to_token_id, distance, occurrences)
		VALUES (?, ?, ?, 0)
		""",
		(from_id, to_id, dist),
	)
	conn.execute(
		"""
		UPDATE chat_cooccurrence SET occurrences = occurrences + ?
		WHERE from_token_id = ? AND to_token_id = ? AND distance = ?
		""",
		(increment, from_id, to_id, dist),
	)


def get_chat_cooccurrences(
	conn: sqlite3.Connection,
	from_ids: Sequence[int],
	max_dist: int = 5,
) -> List[Tuple[int, int, int, int]]:
	if not from_ids:
		return []
	ph = ",".join("?" * len(from_ids))
	rows = conn.execute(
		"""
		SELECT from_token_id, to_token_id, distance, occurrences
		FROM chat_cooccurrence
		WHERE from_token_id IN ({}) AND distance <= ?
		""".format(ph),
		(*from_ids, max_dist),
	).fetchall()
	return [(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in rows]


def chat_totals_by_from_and_distance(
	conn: sqlite3.Connection, from_ids: Iterable[int], max_dist: int = 5
) -> dict[Tuple[int, int], int]:
	from_id_list = list(from_ids)
	if not from_id_list:
		return {}
	ph = ",".join("?" * len(from_id_list))
	cur = conn.execute(
		"""
		SELECT from_token_id, distance, SUM(occurrences)
		FROM chat_cooccurrence
		WHERE from_token_id IN ({}) AND distance <= ?
		GROUP BY from_token_id, distance
		""".format(ph),
		(*from_id_list, max_dist),
	)
	return {(int(r[0]), int(r[1])): int(r[2]) for r in cur.fetchall()}


def chat_common_tokens(conn: sqlite3.Connection, limit: int = 50) -> List[Tuple[int, str, int]]:
	cur = conn.execute(
		"SELECT id, token, count FROM chat_tokens ORDER BY count DESC LIMIT ?",
		(limit,),
	)
	return [(int(r[0]), str(r[1]), int(r[2])) for r in cur.fetchall()]


def chat_token_text(conn: sqlite3.Connection, token_id: int) -> Optional[str]:
	row = conn.execute("SELECT token FROM chat_tokens WHERE id = ?", (token_id,)).fetchone()
	return str(row[0]) if row else None

