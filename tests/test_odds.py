from wealth_prediction.stats.odds import combination_count, jackpot_odds


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
