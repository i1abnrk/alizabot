"""
Runtime paths and argparse defaults for the v0.1.2 live console.

Default DB path is ``artifacts/index.sqlite`` under the process working directory;
override with ``ALIZABOT_DB_PATH`` or ``--db`` (see ``src.console``).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DB_PATH = Path(os.getenv("ALIZABOT_DB_PATH", "artifacts/index.sqlite")).resolve()


def build_arg_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="AlizaBot v0.1.2 live console")
	p.add_argument(
		"--db",
		type=str,
		default=None,
		help="SQLite database path (overrides ALIZABOT_DB_PATH for this run)",
	)
	return p


def apply_db_cli_arg(db: str | None) -> None:
	"""Mutate ``DB_PATH`` when ``--db`` is passed."""
	global DB_PATH
	if db:
		DB_PATH = Path(db).expanduser().resolve()
