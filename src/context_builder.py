from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .db import get_or_create_token_ids, init_schema
from .tokenizer import tokenize
from .utils import canonical_index_path, file_stat_fingerprint, read_text_robust

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
                file_size INTEGER,
                mtime_ns INTEGER,
                file_mtime REAL,
                last_indexed REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
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
        self._migrate_indexed_files_schema(conn)
        conn.commit()

    def migrate_path_keys_to_canonical(self, conn: sqlite3.Connection) -> None:
        """
        Rewrite indexed_files / file_cooccurrence path keys to match canonical_index_path.

        Older builds stored str(path.resolve()) (Windows drive letter casing, backslashes).
        Mismatched keys make every corpus file look \"new\" and duplicate co-occurrence rows.
        """
        rows = conn.execute("SELECT path FROM indexed_files").fetchall()
        for (old_key,) in rows:
            try:
                p = Path(old_key)
            except OSError:
                continue
            if not p.is_file():
                continue
            new_key = canonical_index_path(p)
            if new_key == old_key:
                continue
            with conn:
                conn.execute(
                    "UPDATE file_cooccurrence SET path = ? WHERE path = ?",
                    (new_key, old_key),
                )
                conn.execute(
                    "UPDATE indexed_files SET path = ? WHERE path = ?",
                    (new_key, old_key),
                )

    def _migrate_indexed_files_schema(self, conn: sqlite3.Connection) -> None:
        """Add fingerprint columns to older databases that only stored file_mtime."""
        cols = {row[1] for row in conn.execute("PRAGMA table_info(indexed_files)")}
        if "file_size" not in cols:
            conn.execute("ALTER TABLE indexed_files ADD COLUMN file_size INTEGER")
        if "mtime_ns" not in cols:
            conn.execute("ALTER TABLE indexed_files ADD COLUMN mtime_ns INTEGER")
        if "file_mtime" not in cols:
            conn.execute("ALTER TABLE indexed_files ADD COLUMN file_mtime REAL")

    def get_changed_files(self, conn: sqlite3.Connection, files: Iterable[Path]) -> List[Path]:
        """
        Return files that need (re)indexing.

        Uses (file_size, mtime_ns) as the fingerprint. Integer mtime avoids SQLite
        REAL rounding that could make files look perpetually newer than stored.

        Legacy rows (NULL file_size or mtime_ns) are compared with float mtime
        tolerance, then backfilled when the file is unchanged on disk.
        """
        changed: List[Path] = []
        for file_path in files:
            key = canonical_index_path(file_path)
            cur_size, cur_ns = file_stat_fingerprint(file_path)
            row = conn.execute(
                """
                SELECT file_size, mtime_ns, file_mtime
                FROM indexed_files
                WHERE path = ?
                """,
                (key,),
            ).fetchone()
            if row is None:
                changed.append(file_path)
                continue
            sz_stored, ns_stored, mtime_legacy = row[0], row[1], row[2]
            if sz_stored is not None and ns_stored is not None:
                if (sz_stored, ns_stored) != (cur_size, cur_ns):
                    changed.append(file_path)
                continue
            # Legacy row (missing fingerprint columns): match on mtime float + size when known
            st = file_path.stat()
            legacy_mtime = float(mtime_legacy) if mtime_legacy is not None else None
            size_ok = sz_stored is None or sz_stored == st.st_size
            if legacy_mtime is not None and abs(st.st_mtime - legacy_mtime) <= 1e-6 and size_ok:
                conn.execute(
                    """
                    UPDATE indexed_files
                    SET file_size = ?, mtime_ns = ?
                    WHERE path = ?
                    """,
                    (cur_size, cur_ns, key),
                )
                continue
            changed.append(file_path)
        return changed

    def list_indexed_paths(self, conn: sqlite3.Connection) -> List[str]:
        rows = conn.execute("SELECT path FROM indexed_files").fetchall()
        return [str(r[0]) for r in rows]

    def purge_stale_paths(self, conn: sqlite3.Connection, stale_paths: Iterable[str]) -> int:
        """Remove index data for paths no longer present on disk. Returns rows removed."""
        n = 0
        for path_key in stale_paths:
            self._clear_file_contributions(conn, path_key)
            with conn:
                conn.execute("DELETE FROM indexed_files WHERE path = ?", (path_key,))
            n += 1
        return n

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
        key = canonical_index_path(file_path)
        text = read_text_robust(file_path)
        tokens = tokenize(text, lowercase=self.lowercase, min_len=self.min_token_len)
        if not tokens:
            self._clear_file_contributions(conn, key)
            self._upsert_file_metadata(conn, key, file_path)
            return

        token_ids = get_or_create_token_ids(conn, tokens)
        new_counts = self._build_file_updates(token_ids, tokens)
        old_counts = self._load_old_file_counts(conn, key)

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
            conn.execute("DELETE FROM file_cooccurrence WHERE path = ?", (key,))
            if new_counts:
                conn.executemany(
                    """
                    INSERT INTO file_cooccurrence(path, token_id, neighbor_id, distance, count)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    ((key, t, n, d, c) for (t, n, d), c in new_counts.items()),
                )

            self._upsert_file_metadata_in_tx(conn, key, file_path)

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

    def _upsert_file_metadata(self, conn: sqlite3.Connection, path_key: str, file_path: Path) -> None:
        st = file_path.stat()
        with conn:
            self._upsert_file_metadata_in_tx(conn, path_key, file_path, st=st)

    def _upsert_file_metadata_in_tx(
        self, conn: sqlite3.Connection, path_key: str, file_path: Path, st: Optional[object] = None
    ) -> None:
        if st is None:
            st = file_path.stat()
        conn.execute(
            """
            INSERT INTO indexed_files(path, file_size, mtime_ns, file_mtime, last_indexed)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                file_size = excluded.file_size,
                mtime_ns = excluded.mtime_ns,
                file_mtime = excluded.file_mtime,
                last_indexed = excluded.last_indexed
            """,
            (path_key, st.st_size, st.st_mtime_ns, st.st_mtime, time.time()),
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
