"""Weighted roulette-wheel sampling and 5-distance co-occurrence inference.

``ChancePie`` supports a two-stage draw: a broad ``sample_n`` shortlist (~121),
then a final ``pick`` from that shortlist. ``WorkPicker`` scores next-token
candidates from the global ``cooccurrence`` table using the same directional
semantics as indexing (right token, left neighbor at distance *d*).
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from collections import defaultdict
import sqlite3
import random
import math
from bisect import bisect_left

# Sentinel used to left-pad short histories to a fixed 5-token window.
_CONTEXT_PAD = "<PAD>"


@dataclass
class WeightedToken:
    token: str
    weight: float
    token_id: int = 0


class ChancePie:
    """Weighted roulette wheel sampler (two-stage)."""

    def __init__(self, weighted_tokens: List[WeightedToken]) -> None:
        self._tokens: List[WeightedToken] = list(weighted_tokens)
        self._weights: List[float] = [max(0.0, float(t.weight)) for t in self._tokens]
        self._cumulative: List[float] = []

        running = 0.0
        for w in self._weights:
            running += w
            self._cumulative.append(running)

        self._total_weight: float = running

    def pick(self) -> Optional[str]:
        if not self._tokens or self._total_weight <= 0.0:
            return None

        needle = random.random() * self._total_weight
        idx = bisect_left(self._cumulative, needle)
        if idx >= len(self._tokens):
            idx = len(self._tokens) - 1
        return self._tokens[idx].token

    def sample_n(self, n: int) -> List[WeightedToken]:
        """Return up to n unique weighted tokens (used for first stage ~121)."""
        if n <= 0 or not self._tokens or self._total_weight <= 0.0:
            return []

        max_unique = sum(1 for w in self._weights if w > 0.0)
        target = min(n, max_unique)
        if target <= 0:
            return []

        chosen: List[WeightedToken] = []
        seen_ids: set = set()

        # Oversample with replacement, then dedupe by token_id (or text if id is 0).
        max_attempts = max(target * 6, len(self._tokens) * 2)
        attempts = 0

        while len(chosen) < target and attempts < max_attempts:
            attempts += 1
            needle = random.random() * self._total_weight
            idx = bisect_left(self._cumulative, needle)
            if idx >= len(self._tokens):
                idx = len(self._tokens) - 1

            tok = self._tokens[idx]
            if self._weights[idx] <= 0.0:
                continue

            uniqueness_key = ("id", tok.token_id) if tok.token_id != 0 else ("token", tok.token)
            if uniqueness_key in seen_ids:
                continue

            seen_ids.add(uniqueness_key)
            chosen.append(tok)

        return chosen


class WorkPicker:
    """5-distance weighted picker matching original AlizaGameAPI logic."""

    def __init__(self, db_path: str, k: float = 0.04, first_stage_size: int = 121) -> None:
        self.conn = sqlite3.connect(db_path)
        self.k = k
        self.base = 1.0 + k
        self.first_stage_size = first_stage_size

    def get_next_token(self, context: List[str]) -> Optional[str]:
        """Given a list of previous tokens, return the next token."""
        weighted = self._get_weighted_candidates(context)
        if not weighted:
            return None

        broad = ChancePie(weighted)
        shortlist = broad.sample_n(self.first_stage_size)
        if not shortlist:
            return None

        final = ChancePie(shortlist)
        return final.pick()

    def _pad_to_five(self, context: List[str]) -> List[str]:
        """Keep the last five tokens; pad on the left with ``<PAD>`` if needed."""
        tail = list(context[-5:]) if len(context) >= 5 else list(context)
        if len(tail) < 5:
            tail = [_CONTEXT_PAD] * (5 - len(tail)) + tail
        return tail

    def _get_weighted_candidates(self, context: List[str]) -> List[WeightedToken]:
        """Core logic: for each of the last 5 tokens, get candidates at that distance,
        compute (count / total_at_d) * (1 + k)^d, then combine scores."""
        window = self._pad_to_five(context)
        # Map context surface forms to ids (unknown tokens, including PAD, are skipped).
        rows = self.conn.execute(
            f"SELECT text, id FROM tokens WHERE text IN ({','.join('?' * len(window))})",
            window,
        ).fetchall()
        text_to_id: Dict[str, int] = {str(t): int(i) for t, i in rows}

        scores: Dict[int, float] = defaultdict(float)

        for slot, surface in enumerate(window):
            distance = 5 - slot
            neighbor_id = text_to_id.get(surface)
            if neighbor_id is None:
                continue

            (total_at_d,) = self.conn.execute(
                """
                SELECT COALESCE(SUM(count), 0)
                FROM cooccurrence
                WHERE neighbor_id = ? AND distance = ?
                """,
                (neighbor_id, distance),
            ).fetchone()
            total = float(total_at_d)
            if total <= 0.0:
                continue

            boost = math.pow(self.base, float(distance))
            cur = self.conn.execute(
                """
                SELECT token_id, count
                FROM cooccurrence
                WHERE neighbor_id = ? AND distance = ?
                """,
                (neighbor_id, distance),
            )
            for token_id, count in cur:
                contrib = (float(count) / total) * boost
                scores[int(token_id)] += contrib

        if not scores:
            return []

        ids = list(scores.keys())
        placeholders = ",".join("?" * len(ids))
        id_to_text = {
            int(i): str(t)
            for i, t in self.conn.execute(
                f"SELECT id, text FROM tokens WHERE id IN ({placeholders})",
                ids,
            )
        }

        # +1 prior on every candidate so nothing has zero mass after combining distances.
        prior = 1.0
        return [
            WeightedToken(
                token=id_to_text.get(tid, ""),
                weight=scores[tid] + prior,
                token_id=tid,
            )
            for tid in ids
            if tid in id_to_text
        ]


def test_picker() -> None:
    picker = WorkPicker("artifacts/index.sqlite")
    context = ["the", "sky", "is", "very", "blue"]
    for _ in range(10):
        token = picker.get_next_token(context)
        print(token)
        context.append(token)


if __name__ == "__main__":
    test_picker()
