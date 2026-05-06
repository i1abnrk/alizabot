import argparse
import os
import sys
from typing import List

from .corpus import iter_token_sequences
from .db import connect_db, init_schema
from .indexer import index_token_sequence


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Build 5-distance token co-occurrence index (t-1..t-5).")
	parser.add_argument("--data-dir", required=True, help="Root directory containing .txt files (recursive).")
	parser.add_argument("--db-path", required=True, help="Path to SQLite database to create/update.")
	parser.add_argument("--min-token-len", type=int, default=1, help="Drop tokens shorter than this length.")
	parser.add_argument("--no-lowercase", action="store_true", help="Disable lowercasing of tokens.")
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	if not os.path.isdir(args.data_dir):
		print(f"Data directory not found: {args.data_dir}", file=sys.stderr)
		return 2
	conn = connect_db(args.db_path)
	init_schema(conn)
	lowercase = not args.no_lowercase

	total_files = 0
	total_tokens = 0
	for path, tokens in iter_token_sequences(args.data_dir, lowercase=lowercase, min_len=args.min_token_len):
		total_files += 1
		total_tokens += len(tokens)
		index_token_sequence(conn, tokens)
	print(f"Indexed {total_tokens} tokens from {total_files} files into {args.db_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

