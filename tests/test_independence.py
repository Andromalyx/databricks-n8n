import numpy as np

from wealth_prediction.data.synthetic import simulate_fair_draws
from wealth_prediction.stats.independence import (
    gap_distribution_test,
    ljung_box_autocorrelation,
    pairwise_cooccurrence_test,
    runs_test,
)


def test_runs_test_fails_to_reject_random_sequence():
    rng = np.random.default_rng(0)
    seq = rng.integers(0, 2, size=500)
    result = runs_test(seq)
    assert result.p_value > 0.01


def test_runs_test_rejects_perfectly_alternating_sequence():
    seq = np.array([0, 1] * 100)
    result = runs_test(seq)
    assert result.significant


def test_ljung_box_fails_to_reject_white_noise():
    rng = np.random.default_rng(0)
    series = rng.normal(size=300)
    result = ljung_box_autocorrelation(series, lags=10)
    assert result.p_value > 0.01


def test_pairwise_cooccurrence_fails_to_reject_fair_data(game_config):
    df = simulate_fair_draws(2000, game_config.pool_size, game_config.draw_size, seed=5)
    result = pairwise_cooccurrence_test(df, game_config)
    assert result.p_value > 0.01


def test_gap_distribution_runs_without_error(game_config):
    df = simulate_fair_draws(500, game_config.pool_size, game_config.draw_size, seed=5)
    result = gap_distribution_test(df, game_config)
    assert 0.0 <= result.p_value <= 1.0
