"""
SQLite index orchestration: change detection, full co-occurrence rebuilds, and status.

Issue #2: the legacy ``file_cooccurrence`` table is dropped in favor of a lighter DB.
``indexed_files`` (path, file_size, mtime_ns, …) still records fingerprints so we only
recompute the global ``cooccurrence`` table when the corpus actually changes.

Manual verification (issue #1 regression checks)
-----------------------------------------------
1. Delete ``artifacts/index.sqlite`` (or your ``--db-path``).
2. Run the indexer CLI; expect: "Database not found -> performing full clean build".
3. Run the same command again immediately; expect: "No changes detected -> loading existing database".
4. Touch or open+save one corpus ``.txt`` file, run indexer; expect a fingerprint-driven
   full co-occurrence rebuild (not a per-file SQL merge).
5. Run with ``--force-rebuild``; expect a full rebuild every time regardless of fingerprints.

See also: ``ContextBuilder.get_changed_files`` uses (size, mtime_ns) fingerprints to avoid
SQLite REAL mtime rounding falsely marking every file as changed.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path
from typing import List, Optional, Set

from .context_builder import ContextBuilder
from .utils import canonical_index_path, ensure_parent_dir, iter_text_files, total_text_bytes

MANIFEST_KEY = "corpus_manifest_sha256"


def _compute_corpus_manifest_sha256(data_dir: Path, files: List[Path]) -> str:
    """
    SHA256 of sorted lines ``relative_path\\0size_bytes\\0mtime_ns`` (UTF-8).

    Relative paths are stable when the corpus root moves; sizes and nanosecond mtimes
    catch real edits without relying on the database file's own timestamp.
    """

    def sort_key(p: Path) -> str:
        rel = p.relative_to(data_dir).as_posix()
        return rel.casefold() if os.name == "nt" else rel

    lines: List[str] = []
    for path in sorted(files, key=sort_key):
        rel = path.relative_to(data_dir).as_posix()
        st = path.stat()
        lines.append(f"{rel}\0{st.st_size}\0{st.st_mtime_ns}")
    raw = "\n".join(lines)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_manifest(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute(
        "SELECT value FROM index_metadata WHERE key = ?",
        (MANIFEST_KEY,),
    ).fetchone()
    return str(row[0]) if row else None


def _write_manifest(conn: sqlite3.Connection, digest: str) -> None:
    conn.execute(
        """
        INSERT INTO index_metadata(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (MANIFEST_KEY, digest),
    )


def _clear_incremental_tables(conn: sqlite3.Connection) -> None:
    """Remove co-occurrence and file tracking so a rebuild starts from a blank slate."""
    conn.execute("DELETE FROM cooccurrence")
    conn.execute("DELETE FROM indexed_files")
    conn.execute("DELETE FROM index_metadata")
    conn.execute("DELETE FROM tokens")


class DatabaseManager:
    """Coordinate database freshness checks and incremental indexing."""

    def __init__(self, context_builder: ContextBuilder) -> None:
        self.context_builder = context_builder

    def get_connection(self, db_path: str | Path) -> sqlite3.Connection:
        path = Path(db_path)
        ensure_parent_dir(path)
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def drop_file_cooccurrence_table(self, conn: sqlite3.Connection) -> None:
        """
        Migration helper: remove the legacy per-file co-occurrence table if present.

        Dropping it shrinks the database; global ``cooccurrence`` is rebuilt from disk
        when fingerprints indicate a corpus change.
        """
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'file_cooccurrence'"
        ).fetchone()
        if row is not None:
            print("Dropping legacy file_cooccurrence table for size optimization...")
            conn.execute("DROP TABLE file_cooccurrence")

    def build_or_load(
        self,
        data_dir: str | Path,
        db_path: str | Path,
        *,
        force_rebuild: bool = False,
        stream_mode: bool = True,
    ) -> sqlite3.Connection:
        data_path = Path(data_dir)
        db_file = Path(db_path)
        db_existed_before = db_file.exists()

        conn = self.get_connection(db_file)
        self.context_builder.initialize(conn)
        self.context_builder.migrate_path_keys_to_canonical(conn)
        if stream_mode:
            self.drop_file_cooccurrence_table(conn)
        conn.commit()

        all_files: List[Path] = list(iter_text_files(data_path))
        if not all_files:
            print(f"No .txt files found in {data_path}. Loaded database without updates.")
            self._finalize_post_build(conn, data_path, db_file, processed_paths=[], full_rebuild=False)
            return conn

        if force_rebuild:
            print("Force rebuild requested -> clearing co-occurrence tables and rebuilding from scratch.")
            _clear_incremental_tables(conn)
            conn.commit()
            print(f"Indexing {len(all_files)} files (full rebuild)...")
            processed = self.context_builder.rebuild_corpus_cooccurrence(
                conn, all_files, stream_mode=stream_mode
            )
            conn.commit()
            manifest = _compute_corpus_manifest_sha256(data_path, all_files)
            _write_manifest(conn, manifest)
            conn.commit()
            print(f"Full rebuild complete. Processed {processed} files.")
            self._finalize_post_build(
                conn, data_path, db_file, processed_paths=all_files, full_rebuild=True
            )
            return conn

        if not db_existed_before:
            print("Database not found -> performing full clean build.")
            print(f"Indexing {len(all_files)} files...")
            processed = self.context_builder.rebuild_corpus_cooccurrence(
                conn, all_files, stream_mode=stream_mode
            )
            conn.commit()
            manifest = _compute_corpus_manifest_sha256(data_path, all_files)
            _write_manifest(conn, manifest)
            conn.commit()
            print(f"Initial build complete. Processed {processed} files.")
            self._finalize_post_build(
                conn, data_path, db_file, processed_paths=all_files, full_rebuild=True
            )
            return conn

        indexed_paths = set(self.context_builder.list_indexed_paths(conn))
        current_paths: Set[str] = {canonical_index_path(p) for p in all_files}
        stale_paths = sorted(indexed_paths - current_paths)
        removed_stale = 0
        if stale_paths:
            removed_stale = self.context_builder.purge_stale_paths(conn, stale_paths)
            conn.commit()
            print(f"Removed {removed_stale} missing file(s) from the index (deleted or moved on disk).")

        manifest = _compute_corpus_manifest_sha256(data_path, all_files)
        manifest_stored = _read_manifest(conn)
        changed_files = self.context_builder.get_changed_files(conn, all_files)

        if not changed_files and manifest == manifest_stored and removed_stale == 0:
            print("No changes detected -> loading existing database.")
            self._finalize_post_build(conn, data_path, db_file, processed_paths=[], full_rebuild=False)
            return conn

        if not changed_files and manifest != manifest_stored and removed_stale == 0:
            _write_manifest(conn, manifest)
            conn.commit()
            print("Index data matches files on disk; updated corpus manifest only.")
            self._finalize_post_build(conn, data_path, db_file, processed_paths=[], full_rebuild=False)
            return conn

        reason = f"{len(changed_files)} file(s) fingerprinted as changed" if changed_files else "index/corpus mismatch"
        if removed_stale:
            reason = f"{reason}; removed {removed_stale} stale path(s)" if changed_files else f"removed {removed_stale} stale path(s)"
        print(f"{reason} -> rebuilding global co-occurrence from corpus ({len(all_files)} file(s)).")
        processed = self.context_builder.rebuild_corpus_cooccurrence(
            conn, all_files, stream_mode=stream_mode
        )
        conn.commit()
        _write_manifest(conn, manifest)
        conn.commit()
        print(f"Rebuild complete. Refreshed metadata for {processed} file(s).")
        self._finalize_post_build(
            conn,
            data_path,
            db_file,
            processed_paths=all_files,
            full_rebuild=True,
        )
        return conn

    def _finalize_post_build(
        self,
        conn: sqlite3.Connection,
        data_path: Path,
        db_file: Path,
        *,
        processed_paths: List[Path],
        full_rebuild: bool,
    ) -> None:
        try:
            corpus_bytes = total_text_bytes(data_path)
            db_bytes = db_file.stat().st_size
        except OSError:
            return

        min_corpus_for_warn = 512 * 1024  # skip noisy ratio on toy corpora
        if corpus_bytes >= min_corpus_for_warn and db_bytes > corpus_bytes * 1.5:
            ratio = db_bytes / corpus_bytes
            print(
                f"Warning: database size ({db_bytes / (1024 * 1024):.1f} MB) is {ratio:.2f}x "
                f"the raw text size ({corpus_bytes / (1024 * 1024):.1f} MB). "
                "If this is unexpected, try ``--force-rebuild`` and then ``VACUUM``."
            )

        if full_rebuild and len(processed_paths) >= 50:
            print(
                f"Tip: after a large rebuild, run ``sqlite3 {db_file} 'VACUUM;'`` "
                f"to compact the database (current size: {db_bytes / (1024 * 1024):.1f} MB)."
            )


def manual_test_instructions() -> str:
    """Return the manual test steps documented in this module's docstring."""
    return __doc__ or ""
