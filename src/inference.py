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


def bayes_weight(candidate: str, **kwargs) -> float:
    """
    Placeholder for the Bayesian component of candidate scoring.

    Currently returns 1.0 (neutral) until a real weight calculation is decided.

    Future usage example (as described):
        for each candidate in pie:
            pie.set_value( get_value(candidate) * bayes_weight(candidate) )
    """
    return 1.0


@dataclass
class WeightedToken:
    token: str
    weight: float
    token_id: int = 0


class ChancePie:
    """Weighted roulette wheel sampler (two-stage).

    Follows the spirit of the original AbstractChancePie:
    - Accepts weighted candidates
    - Normalizes internally to a probability distribution (weights sum to 1.0)
    - Selection is performed in the [0, 1) probability space
    """

    def __init__(self, weighted_tokens: List[WeightedToken]) -> None:
        self._tokens: List[WeightedToken] = list(weighted_tokens)
        raw_weights = [max(0.0, float(t.weight)) for t in self._tokens]

        total = sum(raw_weights)
        if total > 0.0:
            self._probs: List[float] = [w / total for w in raw_weights]
        else:
            self._probs = [0.0] * len(raw_weights)

        # Build cumulative distribution over the normalized probabilities (sums to ~1.0)
        self._cumulative: List[float] = []
        running = 0.0
        for p in self._probs:
            running += p
            self._cumulative.append(running)

        # Clamp last value to exactly 1.0
        if self._cumulative:
            self._cumulative[-1] = 1.0

    def pick(self) -> Optional[str]:
        """Alias for next() for backward compatibility."""
        return self.next()

    def next(self) -> Optional[str]:
        """Return one sample using the normalized probability distribution [0, 1)."""
        if not self._tokens or not any(p > 0 for p in self._probs):
            return None

        # Sample in the normalized [0, 1) probability space
        needle = random.random()
        idx = bisect_left(self._cumulative, needle)
        if idx >= len(self._tokens):
            idx = len(self._tokens) - 1
        return self._tokens[idx].token

    def sample_n(self, n: int) -> List[WeightedToken]:
        """Return up to n unique tokens by repeatedly calling next() (kept for compatibility/experiments)."""
        if n <= 0 or not self._tokens:
            return []

        chosen: List[WeightedToken] = []
        seen: set = set()
        attempts = 0
        max_attempts = max(n * 8, len(self._tokens) * 3)

        while len(chosen) < n and attempts < max_attempts:
            attempts += 1
            token_str = self.next()
            if token_str is None:
                break

            for wt in self._tokens:
                if wt.token == token_str:
                    key = ("id", wt.token_id) if wt.token_id != 0 else ("token", wt.token)
                    if key not in seen:
                        seen.add(key)
                        chosen.append(wt)
                    break

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
        """Given a list of previous tokens, return the next token.

        This implements the classic two-stage ChancePie selection:
        1. Instantiate a ChancePie over all weighted candidates.
        2. Call .next() first_stage_size times (with deduplication) to build a shortlist.
        3. Instantiate a second ChancePie over the shortlist and call .next() once.
        """
        candidates = self._get_weighted_candidates(context)
        if not candidates:
            return None

        # Stage 1: Broad sampling - instantiate ChancePie and drive it directly
        broad_pie = ChancePie(candidates)

        shortlist: List[WeightedToken] = []
        seen: set = set()
        attempts = 0
        max_attempts = max(self.first_stage_size * 8, len(candidates) * 3)

        while len(shortlist) < self.first_stage_size and attempts < max_attempts:
            attempts += 1
            token = broad_pie.next()
            if token is None:
                break

            # Find the corresponding WeightedToken for deduping
            # (we use token_id when available for uniqueness)
            for wt in candidates:
                if wt.token == token:
                    key = ("id", wt.token_id) if wt.token_id != 0 else ("token", wt.token)
                    if key not in seen:
                        seen.add(key)
                        shortlist.append(wt)
                    break

        if not shortlist:
            return None

        # Stage 2: Final selection from the shortlist
        final_pie = ChancePie(shortlist)
        return final_pie.next()

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
        """Compute the initial weighted candidate list for a ChancePie.

        For each context token at its distance we compute:
            base_fitness = count / total_at_distance
            final_weight = base_fitness * distance_weight * bayes_weight(candidate)

        The bayes_weight() function is currently a dummy that returns 1.0.
        When a real Bayesian weight calculation is ready, it will be applied here
        (following the pattern: pie.set_value(get_value(candidate) * bayes_weight(candidate)) ).

        The resulting list of WeightedToken is passed to ChancePie for
        two-stage selection.
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
                base_fitness = (float(count) / total) if total > 0.0 else 0.0
                bw = bayes_weight(surface)   # placeholder (currently returns 1.0)
                weight = base_fitness * boost * bw
                scores[int(token_id)] += weight

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
