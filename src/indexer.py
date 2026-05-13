"""Corpus batch indexer + live v0.1.2 ``index_text`` (whitespace, distance 1–5)."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from . import config
from .db import (
	chat_token_id,
	chat_connection,
	get_or_create_chat_token,
	get_or_create_token_ids,
	update_chat_cooccurrence,
	upsert_cooccurrence_batch,
)


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
	for i in range(0, len(updates), batch_size):
		upsert_cooccurrence_batch(conn, updates[i : i + batch_size])


def index_text(text: str) -> None:
	"""Whitespace tokenization; update chat token counts and forward co-occurrence (d=1..5)."""
	tokens = text.split()
	if not tokens:
		return
	n = len(tokens)
	with chat_connection(config.DB_PATH) as conn:
		for t in tokens:
			get_or_create_chat_token(conn, t)
		for i in range(n):
			from_tok = tokens[i]
			from_id = chat_token_id(conn, from_tok)
			if from_id is None:
				continue
			max_d = min(5, n - 1 - i)
			for d in range(1, max_d + 1):
				to_tok = tokens[i + d]
				to_id = chat_token_id(conn, to_tok)
				if to_id is None:
					continue
				update_chat_cooccurrence(conn, from_id, to_id, d, increment=1)
