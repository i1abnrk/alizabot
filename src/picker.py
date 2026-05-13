"""
MVP reply line: live ``index_text`` then distance-weighted ``random.choices``.

The Java-era stack used ``WordPicker`` plus ``ChancePie`` roulette stages; this
``WordPicker`` is a small stub so a richer scorer can replace it later without
changing ``console.py``.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from . import config
from .db import (
	chat_common_tokens,
	chat_connection,
	chat_token_id,
	chat_token_text,
	chat_totals_by_from_and_distance,
	get_chat_cooccurrences,
)
from .indexer import index_text


class WordPicker:
	"""Placeholder for the classic WordPicker; swap in cost tables / decay later."""

	def __init__(self, rng: random.Random | None = None) -> None:
		self._rng = rng or random.Random()


def _context_token_ids(conn, window_tokens: Sequence[str]) -> List[int]:
	ids: List[int] = []
	for t in window_tokens:
		tid = chat_token_id(conn, t)
		if tid is not None:
			ids.append(tid)
	return ids


def _weighted_pick(
	rng: random.Random,
	candidates: Sequence[int],
	weights: Sequence[float],
) -> int | None:
	if not candidates or not weights:
		return None
	return rng.choices(candidates, weights=weights, k=1)[0]


def _score_candidates(
	conn,
	from_ids: Sequence[int],
	max_dist: int = 5,
) -> Tuple[List[int], List[float]]:
	rows = get_chat_cooccurrences(conn, from_ids, max_dist=max_dist)
	if not rows:
		return [], []
	totals = chat_totals_by_from_and_distance(conn, from_ids, max_dist=max_dist)
	agg: Dict[int, float] = {}
	for from_i, to_i, d, occ in rows:
		denom = totals.get((from_i, d), 0)
		if denom <= 0:
			continue
		score = (occ / denom) * (1.0 + 1.8**d)
		agg[to_i] = agg.get(to_i, 0.0) + score
	if not agg:
		return [], []
	pop = list(agg.keys())
	wts = [agg[i] for i in pop]
	return pop, wts


def _fallback_word(conn, rng: random.Random) -> str | None:
	rows = chat_common_tokens(conn, limit=50)
	if not rows:
		return None
	return rng.choice(rows)[1]


def generate_reply(user_input: str, *, rng: random.Random | None = None) -> str:
	rng = rng or random.Random()
	index_text(user_input)
	user_tokens = user_input.split()
	if not user_tokens:
		with chat_connection(config.DB_PATH) as conn:
			w = _fallback_word(conn, rng)
			return w or "…"

	target_len = max(6, min(25, len(user_tokens)))
	reply_tokens: List[str] = []

	with chat_connection(config.DB_PATH) as conn:
		while len(reply_tokens) < target_len:
			combined = user_tokens + reply_tokens
			k = min(3, max(1, len(combined)))
			window = combined[-k:]
			from_ids = _context_token_ids(conn, window)
			next_id: int | None = None
			if from_ids:
				pop, wts = _score_candidates(conn, from_ids)
				next_id = _weighted_pick(rng, pop, wts)
			if next_id is None:
				w = _fallback_word(conn, rng)
				if not w:
					break
				reply_tokens.append(w)
				continue
			word = chat_token_text(conn, next_id)
			if not word:
				break
			reply_tokens.append(word)

	return " ".join(reply_tokens) if reply_tokens else "…"
