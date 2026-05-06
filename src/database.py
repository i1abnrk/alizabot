from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List

from .context_builder import ContextBuilder
from .utils import ensure_parent_dir, iter_text_files, newest_text_mtime


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

    def should_rebuild(self, data_dir: str | Path, db_path: str | Path) -> bool:
        data_path = Path(data_dir)
        db_file = Path(db_path)

        if not db_file.exists():
            return True

        newest_text = newest_text_mtime(data_path)
        if newest_text is None:
            return False

        db_mtime = db_file.stat().st_mtime
        return newest_text > db_mtime

    def build_or_load(self, data_dir: str | Path, db_path: str | Path) -> sqlite3.Connection:
        data_path = Path(data_dir)
        db_file = Path(db_path)

        conn = self.get_connection(db_file)
        self.context_builder.initialize(conn)

        all_files: List[Path] = list(iter_text_files(data_path))
        if not all_files:
            print(f"No .txt files found in {data_path}. Loaded database without updates.")
            return conn

        changed_files = self.context_builder.get_changed_files(conn, all_files)
        if not changed_files and not self.should_rebuild(data_path, db_file):
            print("Database is up-to-date. Loading existing index.")
            return conn

        print(f"Found {len(changed_files)} new/changed files, updating index...")
        processed = self.context_builder.update_files(conn, changed_files)
        conn.commit()
        print(f"Incremental update complete. Processed {processed} files.")
        return conn
