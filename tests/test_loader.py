import pandas as pd
import pytest

from wealth_prediction.data.loader import numbers_matrix, save_csv, to_long_format, validate_schema
from wealth_prediction.data.synthetic import simulate_fair_draws


def test_validate_schema_accepts_well_formed_data(game_config):
    df = simulate_fair_draws(20, game_config.pool_size, game_config.draw_size, seed=1)
    validate_schema(df, game_config)  # should not raise


def test_validate_schema_rejects_out_of_range_number(game_config):
    df = simulate_fair_draws(5, game_config.pool_size, game_config.draw_size, seed=1)
    df.loc[0, "n1"] = game_config.pool_size + 5
    with pytest.raises(ValueError, match="outside"):
        validate_schema(df, game_config)


def test_validate_schema_rejects_duplicate_numbers_in_a_draw(game_config):
    df = simulate_fair_draws(5, game_config.pool_size, game_config.draw_size, seed=1)
    df.loc[0, "n2"] = df.loc[0, "n1"]
    with pytest.raises(ValueError, match="Duplicate numbers"):
        validate_schema(df, game_config)


def test_validate_schema_rejects_missing_columns(game_config):
    df = simulate_fair_draws(5, game_config.pool_size, game_config.draw_size, seed=1).drop(columns=["n1"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df, game_config)


def test_save_and_load_csv_roundtrip(tmp_path, game_config):
    from wealth_prediction.data.loader import load_csv

    df = simulate_fair_draws(30, game_config.pool_size, game_config.draw_size, seed=1)
    path = tmp_path / "draws.csv"
    save_csv(df, path, game_config)
    loaded = load_csv(path, game_config)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), loaded.reset_index(drop=True), check_dtype=False)


def test_numbers_matrix_shape(game_config):
    df = simulate_fair_draws(15, game_config.pool_size, game_config.draw_size, seed=1)
    matrix = numbers_matrix(df, game_config)
    assert matrix.shape == (15, game_config.draw_size)


def test_to_long_format_row_count(game_config):
    df = simulate_fair_draws(15, game_config.pool_size, game_config.draw_size, seed=1)
    long_df = to_long_format(df, game_config)
    assert len(long_df) == 15 * game_config.draw_size
