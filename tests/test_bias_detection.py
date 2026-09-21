from wealth_prediction.data.synthetic import simulate_drifting_draws, simulate_fair_draws
from wealth_prediction.stats.bias_detection import (
    gap_hazard_curve,
    gap_hazard_trend_test,
    hot_cold_numbers,
    randomness_scorecard,
    split_half_drift_test,
)
from wealth_prediction.stats.uniformity import chi_square_uniformity_test


def test_hot_cold_numbers_shape_and_columns(game_config):
    df = simulate_fair_draws(300, game_config.pool_size, game_config.draw_size, seed=1)
    out = hot_cold_numbers(df, game_config)
    assert len(out) == game_config.pool_size
    assert "distinguishable_from_baseline" in out.columns


def test_split_half_drift_fails_to_reject_stable_fair_history(game_config):
    df = simulate_fair_draws(2000, game_config.pool_size, game_config.draw_size, seed=1)
    result = split_half_drift_test(df, game_config)
    assert result.p_value > 0.01


def test_split_half_drift_rejects_drifting_history(game_config):
    df = simulate_drifting_draws(2000, pool_size=game_config.pool_size, draw_size=game_config.draw_size, seed=1)
    result = split_half_drift_test(df, game_config)
    assert result.significant


def test_gap_hazard_curve_covers_all_panel_rows(game_config):
    df = simulate_fair_draws(300, game_config.pool_size, game_config.draw_size, seed=1)
    curve = gap_hazard_curve(df, game_config, max_gap=15)
    assert curve["n_observations"].sum() == 300 * game_config.pool_size
    assert (curve["hit_rate"] >= 0).all() and (curve["hit_rate"] <= 1).all()


def test_gap_hazard_trend_fails_to_reject_fair_data(game_config):
    # On a genuinely fair/memoryless process, gap length should have no
    # effect on appearance odds -- the "overdue number" idea should not hold.
    df = simulate_fair_draws(1000, game_config.pool_size, game_config.draw_size, seed=1)
    result = gap_hazard_trend_test(df, game_config)
    assert result.p_value > 0.01


def test_randomness_scorecard_aggregates_results(game_config):
    df = simulate_fair_draws(1000, game_config.pool_size, game_config.draw_size, seed=1)
    results = [chi_square_uniformity_test(df, game_config), split_half_drift_test(df, game_config)]
    scorecard = randomness_scorecard(results)
    assert scorecard["n_tests"] == 2
    assert "fisher_combined_p_value" in scorecard
