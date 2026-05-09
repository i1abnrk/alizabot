import argparse
import sys
from pathlib import Path

from .context_builder import ContextBuilder
from .database import DatabaseManager


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Build 5-distance token co-occurrence index (t-1..t-5).")
	parser.add_argument("--data-dir", required=True, help="Root directory containing .txt files (recursive).")
	parser.add_argument("--db-path", required=True, help="Path to SQLite database to create/update.")
	parser.add_argument(
		"--force-rebuild",
		action="store_true",
		help="Clear co-occurrence tables and rebuild the index from scratch (ignores fingerprints).",
	)
	parser.add_argument("--min-token-len", type=int, default=1, help="Drop tokens shorter than this length.")
	parser.add_argument("--no-lowercase", action="store_true", help="Disable lowercasing of tokens.")
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	data_dir = Path(args.data_dir)
	db_path = Path(args.db_path)
	if not data_dir.is_dir():
		print(f"Data directory not found: {args.data_dir}", file=sys.stderr)
		return 2

	builder = ContextBuilder(
		lowercase=not args.no_lowercase,
		min_token_len=args.min_token_len,
		max_distance=5,
	)
	manager = DatabaseManager(builder)
	conn = manager.build_or_load(data_dir=data_dir, db_path=db_path, force_rebuild=args.force_rebuild)
	conn.close()
	print(f"Database ready at {db_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

