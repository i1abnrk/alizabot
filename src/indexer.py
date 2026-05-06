from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from .db import get_or_create_token_ids, upsert_cooccurrence_batch


def build_cooccurrence_updates(
	token_ids: Dict[str, int], tokens: List[str], max_distance: int = 5
) -> List[Tuple[int, int, int, int]]:
	updates: Dict[Tuple[int, int, int], int] = defaultdict(int)
	n = len(tokens)
	for i in range(n):
		token = tokens[i]
		token_id = token_ids.get(token)
		if token_id is None:
			continue
		for d in range(1, max_distance + 1):
			j = i - d
			if j < 0:
				break
			neighbor = tokens[j]
			neighbor_id = token_ids.get(neighbor)
			if neighbor_id is None:
				continue
			key = (token_id, neighbor_id, d)
			updates[key] += 1
	return [(k[0], k[1], k[2], v) for k, v in updates.items()]


def index_token_sequence(conn, tokens: List[str], batch_size: int = 100_000) -> None:
	token_ids = get_or_create_token_ids(conn, tokens)
	updates = build_cooccurrence_updates(token_ids, tokens, max_distance=5)
	# Chunk large batches to keep memory reasonable
	for i in range(0, len(updates), batch_size):
		upsert_cooccurrence_batch(conn, updates[i : i + batch_size])

