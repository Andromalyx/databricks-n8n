from wealth_prediction.data.synthetic import simulate_biased_draws, simulate_fair_draws
from wealth_prediction.stats.uniformity import chi_square_uniformity_test, number_frequencies, per_number_significance


def test_number_frequencies_sum_matches_total_slots(game_config):
    df = simulate_fair_draws(50, game_config.pool_size, game_config.draw_size, seed=1)
    freq = number_frequencies(df, game_config)
    assert len(freq) == game_config.pool_size
    assert freq.sum() == 50 * game_config.draw_size


def test_chi_square_uniformity_fails_to_reject_fair_data(game_config):
    df = simulate_fair_draws(3000, game_config.pool_size, game_config.draw_size, seed=7)
    result = chi_square_uniformity_test(df, game_config)
    assert result.p_value > 0.01


def test_chi_square_uniformity_rejects_biased_data(game_config):
    df = simulate_biased_draws(3000, pool_size=game_config.pool_size, draw_size=game_config.draw_size, seed=7)
    result = chi_square_uniformity_test(df, game_config)
    assert result.significant


def test_per_number_significance_flags_columns_present(game_config):
    df = simulate_fair_draws(500, game_config.pool_size, game_config.draw_size, seed=1)
    out = per_number_significance(df, game_config)
    assert set(out.columns) >= {"number", "observed", "expected", "p_value", "p_adjusted", "flagged"}
    assert len(out) == game_config.pool_size
