"""Exact win odds for a lottery format -- pure combinatorics, no data needed.

These numbers are fixed by the game's rules (pool size, how many numbers you
pick, whether a separate bonus/"gold" ball must also match) and don't depend
on anything the rest of this package tests for. Even for a game confirmed
fair and unpredictable (as both Mega 6/45 and Power 5/35 came back in this
project's audits), the jackpot odds are still exactly this number -- no
number-picking strategy changes it, only the game format does.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb


def combination_count(pool_size: int, draw_size: int) -> int:
    """Number of equally likely ways to choose draw_size numbers from pool_size."""
    return comb(pool_size, draw_size)


@dataclass(frozen=True)
class JackpotOdds:
    game_name: str
    pool_size: int
    draw_size: int
    bonus_pool_size: int | None
    combinations: int

    @property
    def probability(self) -> float:
        return 1 / self.combinations

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.game_name}: 1 in {self.combinations:,} (p={self.probability:.4e})"


def jackpot_odds(
    game_name: str, pool_size: int, draw_size: int, bonus_pool_size: int | None = None
) -> JackpotOdds:
    """Top-prize odds for matching every main number (and the bonus ball, if any).

    `bonus_pool_size` is the range an extra "gold"/power ball is drawn from,
    when the format requires matching it too for the top prize -- pass the
    same value as `pool_size` if it's drawn from the identical range, a
    different int if it's a separate smaller/larger range, or leave it None
    for a format with no bonus-ball requirement.
    """
    combinations = combination_count(pool_size, draw_size)
    if bonus_pool_size:
        combinations *= bonus_pool_size
    return JackpotOdds(game_name, pool_size, draw_size, bonus_pool_size, combinations)


def multi_ticket_probability(single_ticket_probability: float, n_distinct_tickets: int) -> float:
    """P(at least one of n_distinct_tickets wins) for tickets covering DIFFERENT
    combinations, all played against the same single draw.

    This is exact (n * p), not the independent-trials approximation
    1-(1-p)^n: with distinct tickets there's exactly one winning combination
    per draw, so at most one of your tickets can ever match it -- the events
    are mutually exclusive, not independent, which is what makes "just buy
    more tickets" scale linearly rather than compound.
    """
    if not 0 <= single_ticket_probability <= 1:
        raise ValueError(f"single_ticket_probability must be in [0, 1], got {single_ticket_probability}")
    if n_distinct_tickets < 0:
        raise ValueError(f"n_distinct_tickets must be >= 0, got {n_distinct_tickets}")
    return min(1.0, single_ticket_probability * n_distinct_tickets)
