import pytest

from wealth_prediction.stats.odds import (
    PrizeTier,
    combination_count,
    jackpot_odds,
    main_match_distribution,
    multi_ticket_probability,
    prize_table_probabilities,
    score_ticket,
)


def test_combination_count_matches_known_mega_645_odds():
    # Vietlott's own published Mega 6/45 jackpot odds are 1 in 8,145,060.
    assert combination_count(45, 6) == 8_145_060


def test_jackpot_odds_no_bonus_ball():
    odds = jackpot_odds("Mega 6/45", pool_size=45, draw_size=6)
    assert odds.combinations == 8_145_060
    assert odds.probability == 1 / 8_145_060


def test_jackpot_odds_with_bonus_ball_multiplies_combinations():
    odds = jackpot_odds("Power 5/35 + gold", pool_size=35, draw_size=5, bonus_pool_size=35)
    assert odds.combinations == combination_count(35, 5) * 35
    assert odds.combinations == 11_362_120


def test_jackpot_odds_without_bonus_is_easier_than_with_bonus():
    no_bonus = jackpot_odds("5/35", pool_size=35, draw_size=5)
    with_bonus = jackpot_odds("5/35 + gold", pool_size=35, draw_size=5, bonus_pool_size=35)
    assert no_bonus.probability > with_bonus.probability


def test_multi_ticket_probability_scales_linearly():
    p = 1 / 1000
    assert multi_ticket_probability(p, 1) == p
    assert multi_ticket_probability(p, 5) == pytest.approx(5 * p)


def test_multi_ticket_probability_caps_at_one():
    assert multi_ticket_probability(0.3, 10) == 1.0


def test_multi_ticket_probability_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        multi_ticket_probability(1.5, 5)
    with pytest.raises(ValueError):
        multi_ticket_probability(0.1, -1)


def test_main_match_distribution_sums_to_one_and_matches_hand_calc():
    dist = main_match_distribution(35, 5)
    assert dist.keys() == set(range(6))
    assert sum(dist.values()) == pytest.approx(1.0)
    # Jackpot-tier main-match probability must equal 1 / C(35,5).
    assert dist[5] == pytest.approx(1 / combination_count(35, 5))
    # P(match >= 3) matches an independently hand-computed value.
    p_at_least_3 = dist[3] + dist[4] + dist[5]
    assert p_at_least_3 == pytest.approx(4501 / 324_632, rel=1e-9)


def test_prize_table_probabilities_gold_split_sums_to_main_match_probability():
    tiers = [
        PrizeTier("with gold", main_matches=4, gold_match=True, prize_vnd=5_000_000),
        PrizeTier("without gold", main_matches=4, gold_match=False, prize_vnd=500_000),
    ]
    results = prize_table_probabilities(pool_size=35, draw_size=5, gold_pool_size=12, tiers=tiers)
    dist = main_match_distribution(35, 5)
    total = sum(r.probability for r in results)
    assert total == pytest.approx(dist[4])


LOTTO_535_TIERS = [
    PrizeTier("Jackpot", 5, True, None),
    PrizeTier("Giai Nhat", 5, False, 10_000_000),
    PrizeTier("Giai Nhi", 4, True, 5_000_000),
    PrizeTier("Giai Ba", 4, False, 500_000),
    PrizeTier("Giai Tu", 3, True, 100_000),
    PrizeTier("Giai Nam", 3, False, 30_000),
    PrizeTier("Khuyen Khich", 2, True, 10_000),
    PrizeTier("Khuyen Khich", 1, True, 10_000),
    PrizeTier("Khuyen Khich", 0, True, 10_000),
]


def test_score_ticket_matches_exact_tier():
    tier = score_ticket(main_matches=4, gold_match=True, tiers=LOTTO_535_TIERS)
    assert tier.name == "Giai Nhi"


def test_score_ticket_returns_none_for_a_real_loss():
    # 2 main matches without gold isn't listed in the table -- a genuine non-win.
    assert score_ticket(main_matches=2, gold_match=False, tiers=LOTTO_535_TIERS) is None


def test_score_ticket_jackpot():
    tier = score_ticket(main_matches=5, gold_match=True, tiers=LOTTO_535_TIERS)
    assert tier.name == "Jackpot"
