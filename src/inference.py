"""Weighted roulette-wheel sampling utilities for inference.

This module provides `ChancePie`, a small helper used to perform weighted
selection over candidate tokens. It is designed for two-stage selection:

1) Broad shortlist sampling (`sample_n`)
2) Final weighted pick (`pick`)
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import random
from typing import List, Optional


@dataclass
class WeightedToken:
    """A token candidate with an associated non-negative weight."""

    token: str
    weight: float
    token_id: int = 0


class ChancePie:
    """Roulette-wheel sampler over `WeightedToken` entries.

    The class precomputes cumulative weights so single picks can be done in
    O(log n) with binary search.
    """

    def __init__(self, weighted_tokens: List[WeightedToken]):
        """Initialize the sampler and precompute cumulative weights.

        Args:
            weighted_tokens: Candidate tokens to sample from.

        Notes:
            - Negative weights are clamped to 0.0.
            - If all effective weights are 0.0, sampling methods return no
              result gracefully.
        """
        self._tokens: List[WeightedToken] = list(weighted_tokens)
        self._weights: List[float] = [max(0.0, float(t.weight)) for t in self._tokens]
        self._cumulative: List[float] = []

        running = 0.0
        for w in self._weights:
            running += w
            self._cumulative.append(running)

        self._total_weight: float = running

    def __len__(self) -> int:
        """Return number of available token candidates."""
        return len(self._tokens)

    def __repr__(self) -> str:
        """Return a concise debug representation."""
        return (
            f"ChancePie(size={len(self)}, total_weight={self._total_weight:.6f})"
        )

    def pick(self) -> Optional[str]:
        """Pick one token string using weighted random selection.

        Returns:
            The selected token string, or `None` when there is no valid
            positive-weight token to select.
        """
        if not self._tokens or self._total_weight <= 0.0:
            return None

        needle = random.random() * self._total_weight
        idx = bisect_left(self._cumulative, needle)
        if idx >= len(self._tokens):
            idx = len(self._tokens) - 1
        return self._tokens[idx].token

    def sample_n(self, n: int = 121) -> List[WeightedToken]:
        """Sample up to `n` unique weighted tokens.

        This method is intended for first-stage broad selection in a two-stage
        pipeline. It oversamples with replacement and deduplicates until enough
        unique items are collected or attempts are exhausted.

        Args:
            n: Desired number of unique tokens (defaults to 121).

        Returns:
            A list of unique `WeightedToken` objects, size in [0, n].
        """
        if n <= 0 or not self._tokens or self._total_weight <= 0.0:
            return []

        max_unique = sum(1 for w in self._weights if w > 0.0)
        target = min(n, max_unique)
        if target <= 0:
            return []

        chosen: List[WeightedToken] = []
        seen_ids = set()

        # Oversampling budget: enough retries to recover from collisions while
        # remaining bounded for large candidate sets.
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

            # Prefer token_id when present; fall back to token text.
            uniqueness_key = ("id", tok.token_id) if tok.token_id != 0 else ("token", tok.token)
            if uniqueness_key in seen_ids:
                continue

            seen_ids.add(uniqueness_key)
            chosen.append(tok)

        return chosen

    def top_k(self, k: int) -> List[WeightedToken]:
        """Return the top-`k` tokens by descending weight.

        Helpful for debugging and inspection of candidate distributions.
        """
        if k <= 0:
            return []
        return sorted(self._tokens, key=lambda item: item.weight, reverse=True)[:k]
