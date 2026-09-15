import numpy as np

from wealth_prediction.config import ModelingConfig
from wealth_prediction.data.synthetic import simulate_fair_draws
from wealth_prediction.modeling.features import FEATURE_COLUMNS, build_panel, next_draw_features
from wealth_prediction.modeling.predictor import (
    backtest_bundle,
    evaluate_model_vs_baseline,
    predict_next_draw_scores,
    recommend_number_bundle,
    summarize_bundle_backtest,
)


def test_build_panel_shape_and_no_nans(game_config):
    df = simulate_fair_draws(100, game_config.pool_size, game_config.draw_size, seed=1)
    panel = build_panel(df, game_config, history_window=10)
    assert len(panel) == 100 * game_config.pool_size
    assert not panel[FEATURE_COLUMNS].isna().any().any()
    assert set(panel["appeared"].unique()) <= {0, 1}


def test_build_panel_markov_and_cooccurrence_scores_are_probability_like(game_config):
    df = simulate_fair_draws(80, game_config.pool_size, game_config.draw_size, seed=1)
    panel = build_panel(df, game_config, history_window=10)
    # Each draw's markov_score / cooccurrence_score is Laplace-normalized over
    # the pool, so it should sum to ~1 across all 45 numbers within a draw.
    per_draw_sums = panel.groupby("draw_id")[["markov_score", "cooccurrence_score"]].sum()
    assert np.allclose(per_draw_sums["markov_score"], 1.0, atol=1e-6)
    assert np.allclose(per_draw_sums["cooccurrence_score"], 1.0, atol=1e-6)


def test_next_draw_features_shape_and_no_nans(game_config):
    df = simulate_fair_draws(60, game_config.pool_size, game_config.draw_size, seed=1)
    features = next_draw_features(df, game_config, history_window=10)
    assert len(features) == game_config.pool_size
    assert not features[FEATURE_COLUMNS].isna().any().any()
    assert np.isclose(features["markov_score"].sum(), 1.0, atol=1e-6)
    assert np.isclose(features["cooccurrence_score"].sum(), 1.0, atol=1e-6)


def test_evaluate_model_vs_baseline_on_fair_data_shows_no_reliable_edge(game_config):
    # A fair process should not let the model beat the baseline; this is the
    # sanity check that the pipeline doesn't spuriously claim an "edge".
    df = simulate_fair_draws(150, game_config.pool_size, game_config.draw_size, seed=1)
    modeling_cfg = ModelingConfig(history_window=10, test_fraction=0.3)
    result = evaluate_model_vs_baseline(df, game_config, modeling_cfg, alpha=0.01)
    assert result.n_test_samples > 0
    assert not result.model_beats_baseline


def test_predict_next_draw_scores_covers_every_number_and_sums_reasonably(game_config):
    df = simulate_fair_draws(150, game_config.pool_size, game_config.draw_size, seed=1)
    modeling_cfg = ModelingConfig(history_window=10, test_fraction=0.3)
    scores = predict_next_draw_scores(df, game_config, modeling_cfg)
    assert len(scores) == game_config.pool_size
    assert set(scores["number"]) == set(range(1, game_config.pool_size + 1))
    assert (scores["predicted_probability"] >= 0).all()
    assert (scores["lift_over_baseline"] > 0).all()


def test_recommend_number_bundle_returns_requested_size(game_config):
    df = simulate_fair_draws(150, game_config.pool_size, game_config.draw_size, seed=1)
    modeling_cfg = ModelingConfig(history_window=10, test_fraction=0.3)
    bundle = recommend_number_bundle(df, game_config, modeling_cfg, bundle_size=8)
    assert len(bundle) == 8
    # Sorted descending by predicted probability.
    assert bundle["predicted_probability"].is_monotonic_decreasing


def test_backtest_bundle_shape_and_correct_count_bounds(game_config):
    df = simulate_fair_draws(150, game_config.pool_size, game_config.draw_size, seed=1)
    modeling_cfg = ModelingConfig(history_window=10, test_fraction=0.3)
    results = backtest_bundle(df, game_config, modeling_cfg, bundle_size=6)
    assert len(results) > 0
    assert set(results.columns) >= {
        "draw_id", "predicted_numbers", "predicted_probabilities", "actual_numbers", "hits", "n_correct",
    }
    assert (results["n_correct"] >= 0).all()
    assert (results["n_correct"] <= game_config.draw_size).all()
    assert (results["predicted_numbers"].str.len() == 6).all()
    # n_correct must equal the length of hits, and every hit must be in both lists.
    for _, row in results.iterrows():
        assert row["n_correct"] == len(row["hits"])
        assert set(row["hits"]) <= set(row["predicted_numbers"]) & set(row["actual_numbers"])


def test_summarize_bundle_backtest_on_fair_data_does_not_beat_chance(game_config):
    # On a genuinely fair process, the bundle's hit rate should track the
    # exact hypergeometric chance level, not exceed it.
    df = simulate_fair_draws(400, game_config.pool_size, game_config.draw_size, seed=1)
    modeling_cfg = ModelingConfig(history_window=10, test_fraction=0.3)
    results = backtest_bundle(df, game_config, modeling_cfg, bundle_size=6)
    summary = summarize_bundle_backtest(results, game_config, bundle_size=6, alpha=0.01)
    assert summary["n_test_draws"] == len(results)
    assert summary["total_possible"] == len(results) * 6
    assert not summary["beats_chance"]
