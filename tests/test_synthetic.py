import numpy as np

from wealth_prediction.data.synthetic import simulate_biased_draws, simulate_drifting_draws, simulate_fair_draws


def test_simulate_fair_draws_shape(game_config):
    df = simulate_fair_draws(100, game_config.pool_size, game_config.draw_size, seed=1)
    assert len(df) == 100
    cols = [f"n{i + 1}" for i in range(game_config.draw_size)]
    assert list(df.columns) == ["draw_id", "draw_date", *cols]
    assert (df[cols].to_numpy() >= 1).all()
    assert (df[cols].to_numpy() <= game_config.pool_size).all()


def test_simulate_fair_draws_no_duplicates_within_row(game_config):
    df = simulate_fair_draws(200, game_config.pool_size, game_config.draw_size, seed=2)
    cols = [f"n{i + 1}" for i in range(game_config.draw_size)]
    for row in df[cols].to_numpy():
        assert len(set(row)) == game_config.draw_size


def test_simulate_fair_draws_reproducible_with_seed(game_config):
    df1 = simulate_fair_draws(50, game_config.pool_size, game_config.draw_size, seed=42)
    df2 = simulate_fair_draws(50, game_config.pool_size, game_config.draw_size, seed=42)
    cols = [f"n{i + 1}" for i in range(game_config.draw_size)]
    assert np.array_equal(df1[cols].to_numpy(), df2[cols].to_numpy())


def test_simulate_biased_draws_favors_low_numbers(game_config):
    df = simulate_biased_draws(2000, pool_size=game_config.pool_size, draw_size=game_config.draw_size, seed=3)
    cols = [f"n{i + 1}" for i in range(game_config.draw_size)]
    counts = df[cols].to_numpy().flatten()
    low_half = (counts <= game_config.pool_size // 2).sum()
    high_half = (counts > game_config.pool_size // 2).sum()
    assert low_half > high_half  # default bias weights favor low numbers


def test_simulate_drifting_draws_has_expected_length(game_config):
    df = simulate_drifting_draws(300, pool_size=game_config.pool_size, draw_size=game_config.draw_size, seed=4)
    assert len(df) == 300
    assert df["draw_id"].is_unique
