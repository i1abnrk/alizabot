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
    """5-distance weighted picker.

    distance_mode controls the distance weighting formula:
        True  → (1+k) ** distance                 (bonus / close-is-strong)
        False → (1+k) ** (window_size - distance)   (inverted)
    """

    def __init__(self, db_path: str, k: float = 0.04, first_stage_size: int = 121,
                 distance_mode: bool = True, window_size: int = 5) -> None:
        self.conn = sqlite3.connect(db_path)
        self.k = k
        self.first_stage_size = first_stage_size
        self.distance_mode = distance_mode     # boolean control surface
        self.window_size = window_size         # context window size (BERT-style "stride")

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

    def generate_reply(self, text: str, *, rng: random.Random | None = None) -> str:
        """
        High-level convenience method that mimics the old generate_reply behavior.

        Generates a reply whose length is roughly 0.75x – 1.25x the number of
        tokens in the input query. This is the method the web UI and other
        higher-level callers should use instead of manually driving get_next_token.
        """
        rng = rng or random.Random()

        user_tokens = text.split()
        user_len = len(user_tokens)
        if user_len == 0:
            return "…"

        target_len = max(1, int(user_len * rng.uniform(0.75, 1.25)))

        reply_tokens: list[str] = []
        context = list(user_tokens)

        while len(reply_tokens) < target_len:
            next_token = self.get_next_token(context)
            if next_token is None or next_token == "<PAD>":
                break
            reply_tokens.append(next_token)
            context.append(next_token)

        return " ".join(reply_tokens) if reply_tokens else "…"

    def _pad_to_window(self, context: List[str]) -> List[str]:
        """Keep the last `window_size` tokens; pad on the left with ``<PAD>`` if needed."""
        n = self.window_size
        tail = list(context[-n:]) if len(context) >= n else list(context)
        if len(tail) < n:
            tail = [_CONTEXT_PAD] * (n - len(tail)) + tail
        return tail

    def _get_weighted_candidates(self, context: List[str]) -> List[WeightedToken]:
        """Core logic: for each of the last `window_size` tokens, get candidates at that distance,
        compute (count / total_at_d) * distance_weight, then combine scores.

        distance_weight = (1+k) ** distance      if distance_mode else
                          (1+k) ** (window_size - distance)
        """
        window = self._pad_to_window(context)
        # Map context surface forms to ids (unknown tokens, including PAD, are skipped).
        rows = self.conn.execute(
            f"SELECT text, id FROM tokens WHERE text IN ({','.join('?' * len(window))})",
            window,
        ).fetchall()
        text_to_id: Dict[str, int] = {str(t): int(i) for t, i in rows}

        scores: Dict[int, float] = defaultdict(float)

        for slot, surface in enumerate(window):
            distance = self.window_size - slot
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

            # distance_weight formula as specified:
            # (1+k) ** distance if distance_mode else (window_size - distance)
            if self.distance_mode:
                dist_weight = (1 + self.k) ** distance
            else:
                dist_weight = (1 + self.k) ** (self.window_size - distance)

            boost = dist_weight
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
