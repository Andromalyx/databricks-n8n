from wealth_prediction.data.synthetic import simulate_fair_draws
from wealth_prediction.stats.normality import draw_statistics, ks_test_vs_fair_null, run_normality_suite


def test_draw_statistics_columns(game_config):
    df = simulate_fair_draws(20, game_config.pool_size, game_config.draw_size, seed=1)
    stats_df = draw_statistics(df, game_config)
    assert set(stats_df.columns) == {"draw_id", "sum", "mean", "std", "range", "odd_count"}
    assert len(stats_df) == 20


def test_ks_test_vs_fair_null_fails_to_reject_when_data_is_actually_fair(game_config):
    df = simulate_fair_draws(500, game_config.pool_size, game_config.draw_size, seed=99)
    sums = draw_statistics(df, game_config)["sum"].to_numpy()
    result = ks_test_vs_fair_null(sums, game_config, n_sims=2000, seed=1)
    assert result.p_value > 0.01


def test_run_normality_suite_returns_all_expected_keys(game_config):
    df = simulate_fair_draws(200, game_config.pool_size, game_config.draw_size, seed=1)
    results = run_normality_suite(df, game_config, n_sims=500, seed=1)
    assert set(results.keys()) == {"draw_statistics", "shapiro", "dagostino_k2", "ks_gaussian", "ks_fair_null"}
