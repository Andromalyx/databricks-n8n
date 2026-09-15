from wealth_prediction.config import ModelingConfig
from wealth_prediction.data.synthetic import simulate_fair_draws
from wealth_prediction.modeling.features import FEATURE_COLUMNS, build_panel
from wealth_prediction.modeling.predictor import evaluate_model_vs_baseline


def test_build_panel_shape_and_no_nans(game_config):
    df = simulate_fair_draws(100, game_config.pool_size, game_config.draw_size, seed=1)
    panel = build_panel(df, game_config, history_window=10)
    assert len(panel) == 100 * game_config.pool_size
    assert not panel[FEATURE_COLUMNS].isna().any().any()
    assert set(panel["appeared"].unique()) <= {0, 1}


def test_evaluate_model_vs_baseline_on_fair_data_shows_no_reliable_edge(game_config):
    # A fair process should not let the model beat the baseline; this is the
    # sanity check that the pipeline doesn't spuriously claim an "edge".
    df = simulate_fair_draws(150, game_config.pool_size, game_config.draw_size, seed=1)
    modeling_cfg = ModelingConfig(history_window=10, test_fraction=0.3)
    result = evaluate_model_vs_baseline(df, game_config, modeling_cfg, alpha=0.01)
    assert result.n_test_samples > 0
    assert not result.model_beats_baseline
