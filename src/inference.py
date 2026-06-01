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
    """Custom in-memory data view representing token counts from the database.

    Keys are strictly **integer token_ids** internally (faster, less error-prone).
    Strings are only resolved at final output time.

    Core API (inspired by alizagameapi ChancePie):
      - put(token_id: int, count: float = 1.0)
      - get(arc: float = None) -> Optional[int]
      - theta(token_id: int) -> float     # cumulative arc at which this token starts
    """

    def __init__(self) -> None:
        self._counts: Dict[int, float] = {}
        self._total: float = 0.0
        self._cumulative: List[Tuple[int, float]] = []
        self._dirty = True

    def put(self, token_id: int, count: float = 1.0) -> None:
        """Add/increment count for a token_id."""
        if count <= 0:
            return
        self._counts[token_id] = self._counts.get(token_id, 0.0) + count
        self._total += count
        self._dirty = True

    def _rebuild(self) -> None:
        if not self._dirty:
            return
        if self._total <= 0:
            self._cumulative = []
            self._dirty = False
            return

        # Sort by token_id for deterministic order
        sorted_items = sorted(self._counts.items())

        self._cumulative = []
        running = 0.0
        for tid, cnt in sorted_items:
            if cnt > 0:
                running += cnt / self._total
                self._cumulative.append((tid, running))

        if self._cumulative:
            self._cumulative[-1] = (self._cumulative[-1][0], 1.0)

        self._dirty = False

    def get(self, arc: Optional[float] = None) -> Optional[int]:
        """Return token_id at the given normalized arc in [0, 1)."""
        self._rebuild()
        if not self._cumulative:
            return None

        if arc is None:
            arc = random.random()

        idx = bisect_left([c for _, c in self._cumulative], arc)
        if idx >= len(self._cumulative):
            idx = len(self._cumulative) - 1
        return self._cumulative[idx][0]

    def theta(self, token_id: int) -> float:
        """Return the cumulative arc (in [0,1)) at which this token_id would start being selected.

        Inverse of get(arc). Follows the logic from ChancePie3.java.
        """
        self._rebuild()
        if token_id not in self._counts or self._total <= 0:
            return 0.0

        cumulative_before = 0.0
        for tid, cum in self._cumulative:
            if tid == token_id:
                return cumulative_before
            cumulative_before = cum
        return 0.0

    # Aliases
    def next(self) -> Optional[int]:
        return self.get()

    def pick(self) -> Optional[int]:
        return self.get()

    @classmethod
    def from_context(
        cls,
        conn: sqlite3.Connection,
        context: List[str],
        window_size: int = 5,
        distance_mode: bool = True,
        k: float = 0.04,
    ) -> "ChancePie":
        """Builds a ChancePie directly from DB using integer token_ids.

        Distance weighting is applied at population time via weighted put().
        """
        n = window_size
        tail = list(context[-n:]) if len(context) >= n else list(context)
        if len(tail) < n:
            tail = [_CONTEXT_PAD] * (n - len(tail)) + tail

        window = tail

        # Resolve context strings to IDs
        rows = conn.execute(
            f"SELECT text, id FROM tokens WHERE text IN ({','.join('?' * len(window))})",
            window,
        ).fetchall()
        text_to_id: Dict[str, int] = {str(t): int(i) for t, i in rows}

        pie = cls()

        for slot, surface in enumerate(window):
            distance = n - slot
            neighbor_id = text_to_id.get(surface)
            if neighbor_id is None:
                continue

            (total_at_d,) = conn.execute(
                "SELECT COALESCE(SUM(count), 0) FROM cooccurrence "
                "WHERE neighbor_id = ? AND distance = ?",
                (neighbor_id, distance),
            ).fetchone()

            total = float(total_at_d)
            if total <= 0.0:
                continue

            if distance_mode:
                dist_weight = (1 + k) ** distance
            else:
                dist_weight = (1 + k) ** (n - distance)

            cur = conn.execute(
                "SELECT token_id, count FROM cooccurrence "
                "WHERE neighbor_id = ? AND distance = ?",
                (neighbor_id, distance),
            )

            for token_id, count in cur:
                base = (float(count) / total) if total > 0 else 0.0
                effective = base * dist_weight
                if effective > 0:
                    pie.put(int(token_id), effective)

        return pie


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
        self.distance_mode = distance_mode
        self.window_size = window_size

        # String <-> ID caches (populated on demand)
        self._text_to_id: Dict[str, int] = {}
        self._id_to_text: Dict[int, str] = {}

    def _ensure_token_id(self, text: str) -> Optional[int]:
        if text in self._text_to_id:
            return self._text_to_id[text]
        row = self.conn.execute("SELECT id FROM tokens WHERE text = ?", (text,)).fetchone()
        if row:
            tid = int(row[0])
            self._text_to_id[text] = tid
            self._id_to_text[tid] = text
            return tid
        return None

    def _token_text(self, token_id: int) -> str:
        if token_id in self._id_to_text:
            return self._id_to_text[token_id]
        row = self.conn.execute("SELECT text FROM tokens WHERE id = ?", (token_id,)).fetchone()
        text = str(row[0]) if row else ""
        self._id_to_text[token_id] = text
        self._text_to_id[text] = token_id
        return text

    def get_next_token(self, context: List[str]) -> Optional[str]:
        """Two-stage selection. Works with integer token_ids internally."""
        # Convert incoming strings to IDs as early as possible
        context_ids: List[int] = []
        for t in context:
            tid = self._ensure_token_id(t)
            if tid is not None:
                context_ids.append(tid)

        if not context_ids:
            return None

        broad_pie = ChancePie.from_context(
            self.conn,
            context,                    # from_context still accepts strings for convenience (it resolves internally)
            window_size=self.window_size,
            distance_mode=self.distance_mode,
            k=self.k,
        )

        # Stage 1: Broad sampling using repeated .get() (returns token_ids)
        shortlist: List[int] = []
        seen: set = set()
        attempts = 0
        max_attempts = max(self.first_stage_size * 10, 2000)

        while len(shortlist) < self.first_stage_size and attempts < max_attempts:
            attempts += 1
            tid = broad_pie.get()
            if tid is not None and tid != 0 and tid not in seen:   # 0 is not a real token
                seen.add(tid)
                shortlist.append(tid)

        if not shortlist:
            return None

        # Stage 2: Final pie from shortlist (equal weight for now)
        final_pie = ChancePie()
        for tid in shortlist:
            final_pie.put(tid, 1.0)

        chosen_id = final_pie.get()
        if chosen_id is None:
            return None

        return self._token_text(chosen_id)

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
        """Thin wrapper — delegates to ChancePie.from_context for the DB work.

        This method is kept for backward compatibility during the transition.
        The real DB + distance weighting logic now lives in ChancePie.
        """
        pie = ChancePie.from_context(
            self.conn,
            context,
            window_size=self.window_size,
            distance_mode=self.distance_mode,
            k=self.k,
        )
        # Return the internal list so existing code that expects List[WeightedToken] still works.
        # In the future we may return the pie itself.
        return list(pie._tokens)


def test_picker() -> None:
    picker = WorkPicker("artifacts/index.sqlite")
    context = ["the", "sky", "is", "very", "blue"]
    for _ in range(10):
        token = picker.get_next_token(context)
        print(token)
        context.append(token)


if __name__ == "__main__":
    test_picker()
