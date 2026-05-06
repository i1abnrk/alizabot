from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .db import get_or_create_token_ids, init_schema
from .tokenizer import tokenize
from .utils import read_text_robust

CooccurrenceKey = Tuple[int, int, int]
CooccurrenceUpdate = Tuple[int, int, int, int]


class ContextBuilder:
    """Build and incrementally update token co-occurrence context."""

    def __init__(self, lowercase: bool = True, min_token_len: int = 1, max_distance: int = 5) -> None:
        self.lowercase = lowercase
        self.min_token_len = min_token_len
        self.max_distance = max_distance

    def initialize(self, conn: sqlite3.Connection) -> None:
        """Create required schema for base and incremental indexing."""
        init_schema(conn)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS indexed_files (
                path TEXT PRIMARY KEY,
                file_mtime REAL NOT NULL,
                last_indexed REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS file_cooccurrence (
                path TEXT NOT NULL,
                token_id INTEGER NOT NULL,
                neighbor_id INTEGER NOT NULL,
                distance INTEGER NOT NULL CHECK(distance BETWEEN 1 AND 5),
                count INTEGER NOT NULL,
                PRIMARY KEY (path, token_id, neighbor_id, distance),
                FOREIGN KEY(token_id) REFERENCES tokens(id) ON DELETE CASCADE,
                FOREIGN KEY(neighbor_id) REFERENCES tokens(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_file_co_path ON file_cooccurrence(path);
            """
        )
        conn.commit()

    def get_changed_files(self, conn: sqlite3.Connection, files: Iterable[Path]) -> List[Path]:
        """
        Return new or changed files.

        A file is considered changed when:
        - It is not present in indexed_files, or
        - Its current mtime is newer than the tracked file_mtime.
        """
        changed: List[Path] = []
        for file_path in files:
            file_mtime = file_path.stat().st_mtime
            row = conn.execute(
                "SELECT file_mtime FROM indexed_files WHERE path = ?",
                (str(file_path.resolve()),),
            ).fetchone()
            if row is None or file_mtime > float(row[0]):
                changed.append(file_path)
        return changed

    def update_files(self, conn: sqlite3.Connection, files: Iterable[Path]) -> int:
        """Incrementally index files and return count processed."""
        processed = 0
        for file_path in files:
            self._reindex_single_file(conn, file_path)
            processed += 1
        return processed

    def _reindex_single_file(self, conn: sqlite3.Connection, file_path: Path) -> None:
        """
        Re-index one file without double-counting.

        If the file was indexed before, previous per-file contributions are
        subtracted from the global cooccurrence table before new counts are added.
        """
        resolved = str(file_path.resolve())
        text = read_text_robust(file_path)
        tokens = tokenize(text, lowercase=self.lowercase, min_len=self.min_token_len)
        if not tokens:
            self._clear_file_contributions(conn, resolved)
            self._upsert_file_metadata(conn, resolved, file_path.stat().st_mtime)
            return

        token_ids = get_or_create_token_ids(conn, tokens)
        new_counts = self._build_file_updates(token_ids, tokens)
        old_counts = self._load_old_file_counts(conn, resolved)

        with conn:
            if old_counts:
                conn.executemany(
                    """
                    INSERT INTO cooccurrence(token_id, neighbor_id, distance, count)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(token_id, neighbor_id, distance)
                    DO UPDATE SET count = count + excluded.count
                    """,
                    ((t, n, d, -c) for (t, n, d), c in old_counts.items()),
                )

            if new_counts:
                conn.executemany(
                    """
                    INSERT INTO cooccurrence(token_id, neighbor_id, distance, count)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(token_id, neighbor_id, distance)
                    DO UPDATE SET count = count + excluded.count
                    """,
                    ((t, n, d, c) for (t, n, d), c in new_counts.items()),
                )

            conn.execute("DELETE FROM cooccurrence WHERE count <= 0")
            conn.execute("DELETE FROM file_cooccurrence WHERE path = ?", (resolved,))
            if new_counts:
                conn.executemany(
                    """
                    INSERT INTO file_cooccurrence(path, token_id, neighbor_id, distance, count)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    ((resolved, t, n, d, c) for (t, n, d), c in new_counts.items()),
                )

            conn.execute(
                """
                INSERT INTO indexed_files(path, file_mtime, last_indexed)
                VALUES (?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    file_mtime = excluded.file_mtime,
                    last_indexed = excluded.last_indexed
                """,
                (resolved, file_path.stat().st_mtime, time.time()),
            )

    def _load_old_file_counts(self, conn: sqlite3.Connection, file_path: str) -> Dict[CooccurrenceKey, int]:
        rows = conn.execute(
            """
            SELECT token_id, neighbor_id, distance, count
            FROM file_cooccurrence
            WHERE path = ?
            """,
            (file_path,),
        ).fetchall()
        return {(int(t), int(n), int(d)): int(c) for (t, n, d, c) in rows}

    def _clear_file_contributions(self, conn: sqlite3.Connection, file_path: str) -> None:
        old_counts = self._load_old_file_counts(conn, file_path)
        with conn:
            if old_counts:
                conn.executemany(
                    """
                    INSERT INTO cooccurrence(token_id, neighbor_id, distance, count)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(token_id, neighbor_id, distance)
                    DO UPDATE SET count = count + excluded.count
                    """,
                    ((t, n, d, -c) for (t, n, d), c in old_counts.items()),
                )
                conn.execute("DELETE FROM cooccurrence WHERE count <= 0")
            conn.execute("DELETE FROM file_cooccurrence WHERE path = ?", (file_path,))

    def _upsert_file_metadata(self, conn: sqlite3.Connection, file_path: str, file_mtime: float) -> None:
        with conn:
            conn.execute(
                """
                INSERT INTO indexed_files(path, file_mtime, last_indexed)
                VALUES (?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    file_mtime = excluded.file_mtime,
                    last_indexed = excluded.last_indexed
                """,
                (file_path, file_mtime, time.time()),
            )

    def _build_file_updates(self, token_ids: Dict[str, int], tokens: List[str]) -> Dict[CooccurrenceKey, int]:
        updates: Dict[CooccurrenceKey, int] = defaultdict(int)
        token_count = len(tokens)
        for i in range(token_count):
            token_id = token_ids.get(tokens[i])
            if token_id is None:
                continue
            for distance in range(1, self.max_distance + 1):
                j = i - distance
                if j < 0:
                    break
                neighbor_id = token_ids.get(tokens[j])
                if neighbor_id is None:
                    continue
                updates[(token_id, neighbor_id, distance)] += 1
        return updates
