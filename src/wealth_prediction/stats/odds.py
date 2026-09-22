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


def main_match_distribution(pool_size: int, draw_size: int) -> dict[int, float]:
    """P(exactly k of your draw_size picks match the draw_size winning numbers), k=0..draw_size.

    Standard hypergeometric overlap between your ticket and the winning
    combination, both drawn without replacement from the same pool_size --
    this is the distribution every "match k of n" prize tier is built from,
    independent of any separate bonus/gold ball.
    """
    total = combination_count(pool_size, draw_size)
    return {
        k: comb(draw_size, k) * comb(pool_size - draw_size, draw_size - k) / total
        for k in range(draw_size + 1)
    }


@dataclass(frozen=True)
class PrizeTier:
    name: str
    main_matches: int
    gold_match: bool | None  # True=must match gold, False=must NOT match, None=gold irrelevant
    prize_vnd: float | None  # None for a jackpot with no fixed/published amount


@dataclass(frozen=True)
class PrizeTierResult:
    tier: PrizeTier
    probability: float

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        prize = f"{self.tier.prize_vnd:,.0f} VND" if self.tier.prize_vnd is not None else "jackpot (variable)"
        return f"{self.tier.name}: p={self.probability:.6e} (1 in {1 / self.probability:,.1f}), {prize}"


def prize_table_probabilities(
    pool_size: int, draw_size: int, gold_pool_size: int, tiers: list[PrizeTier]
) -> list[PrizeTierResult]:
    """Exact win probability for each tier of a "match k of n main numbers,
    optionally also match/miss a 1-in-gold_pool_size bonus ball" prize table.

    Each tier's main-number-match probability comes from
    `main_match_distribution`; the gold requirement (if any) is independent
    of which main numbers you picked, so the two multiply directly.
    """
    main_dist = main_match_distribution(pool_size, draw_size)
    results = []
    for tier in tiers:
        p_main = main_dist[tier.main_matches]
        if tier.gold_match is None:
            p = p_main
        elif tier.gold_match:
            p = p_main / gold_pool_size
        else:
            p = p_main * (gold_pool_size - 1) / gold_pool_size
        results.append(PrizeTierResult(tier, p))
    return results


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
