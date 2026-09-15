"""Feature engineering for the "can a model beat the baseline" test.

Builds a long panel with one row per (draw, number) and a binary label
`appeared`. Features are strictly backward-looking (computed from draws
*before* the current one) so the walk-forward evaluation in predictor.py
can't leak future information -- the single most common way this kind of
analysis silently cheats itself into a fake "edge".

`markov_score` and `cooccurrence_score` are inspired by the vietvudanh/
vietlott-data project's Markov-chain and pair-frequency strategies -- real,
reasonable techniques to test for temporal/pairwise structure. Unlike that
project's backtester (which scores against a miscalibrated prize table and
lets a couple of rare payouts dominate the ROI), here they're just two more
inputs to the same leakage-safe, proper-scoring-rule evaluation as every
other feature: if the lottery is fair, they carry no more signal than
`trailing_freq`, and `predictor.evaluate_model_vs_baseline` will say so.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from wealth_prediction.config import GameConfig
from wealth_prediction.data.loader import numbers_matrix, to_long_format


def _laplace_normalize(raw: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Smooth a vector of nonnegative counts into a distribution over pool_size numbers."""
    return (raw + alpha) / (raw.sum() + alpha * len(raw))


def _markov_cooccurrence_matrices(
    df: pd.DataFrame, game: GameConfig, alpha: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Two (n_draws + 1, pool_size) score matrices, row t using only draws strictly before t.

    Row t (t < n_draws): `markov_score` = smoothed P(number | numbers seen in
    draw t-1), from a running first-order transition matrix; `cooccurrence_score`
    = smoothed relative frequency of co-occurring with draw t-1's numbers,
    from a running pairwise co-occurrence matrix. Both matrices are updated
    with draw t's own numbers only *after* row t is computed, so nothing here
    can see its own label.

    Row n_draws is the one-step-ahead state for the next, not-yet-observed
    draw -- exactly what `next_draw_features` needs.
    """
    matrix = numbers_matrix(df, game) - 1  # 0-indexed for direct array indexing
    n_draws, pool_size = len(matrix), game.pool_size
    transition_counts = np.zeros((pool_size, pool_size))
    cooccurrence_counts = np.zeros((pool_size, pool_size))
    prev_numbers = None

    markov_scores = np.empty((n_draws + 1, pool_size))
    cooc_scores = np.empty((n_draws + 1, pool_size))
    uniform = np.full(pool_size, 1.0 / pool_size)

    for t in range(n_draws + 1):
        if prev_numbers is None:
            markov_scores[t] = uniform
            cooc_scores[t] = uniform
        else:
            markov_scores[t] = _laplace_normalize(transition_counts[prev_numbers].sum(axis=0), alpha)
            cooc_scores[t] = _laplace_normalize(cooccurrence_counts[prev_numbers].sum(axis=0), alpha)

        if t < n_draws:
            current_numbers = matrix[t]
            if prev_numbers is not None:
                for a in prev_numbers:
                    transition_counts[a, current_numbers] += 1
            for i, j in itertools.combinations(current_numbers, 2):
                cooccurrence_counts[i, j] += 1
                cooccurrence_counts[j, i] += 1
            prev_numbers = current_numbers

    return markov_scores, cooc_scores


def _markov_cooccurrence_frame(df: pd.DataFrame, game: GameConfig, alpha: float = 0.5) -> pd.DataFrame:
    """Long-format (draw_id, number, markov_score, cooccurrence_score) for every observed draw."""
    draw_ids = np.sort(df["draw_id"].unique())
    n_draws = len(draw_ids)
    numbers = np.arange(1, game.pool_size + 1)
    markov_scores, cooc_scores = _markov_cooccurrence_matrices(df, game, alpha=alpha)

    def _to_long(scores: np.ndarray, col_name: str) -> pd.DataFrame:
        wide = pd.DataFrame(scores[:n_draws], index=draw_ids, columns=numbers)
        return wide.reset_index(names="draw_id").melt(id_vars="draw_id", var_name="number", value_name=col_name)

    markov_long = _to_long(markov_scores, "markov_score")
    cooc_long = _to_long(cooc_scores, "cooccurrence_score")
    return markov_long.merge(cooc_long, on=["draw_id", "number"])


def build_panel(df: pd.DataFrame, game: GameConfig, history_window: int = 20) -> pd.DataFrame:
    """One row per (draw_id, number) with label `appeared` and lagged features.

    Features per row (all computed using only draws strictly before draw_id):
      - trailing_freq: appearance rate over the last `history_window` draws
      - overall_freq: appearance rate over all prior draws
      - gap_since_last: draws since this number last appeared (capped at draw index)
      - draw_index: position in history, 0-based (lets a model learn any global drift)
      - markov_score: P(number | numbers in the immediately preceding draw), first-order
      - cooccurrence_score: historical co-occurrence strength with the preceding draw's numbers
    """
    long_df = to_long_format(df, game)
    appeared = long_df.assign(appeared=1)

    draw_ids = np.sort(df["draw_id"].unique())
    numbers = np.arange(1, game.pool_size + 1)

    # Full (draw_id x number) grid so absent numbers get appeared=0 rather than being missing.
    grid = pd.MultiIndex.from_product([draw_ids, numbers], names=["draw_id", "number"]).to_frame(index=False)
    panel = grid.merge(appeared[["draw_id", "number", "appeared"]], on=["draw_id", "number"], how="left")
    panel["appeared"] = panel["appeared"].fillna(0).astype(int)
    panel = panel.sort_values(["number", "draw_id"]).reset_index(drop=True)

    grouped = panel.groupby("number")["appeared"]
    # shift(1) before any rolling/cumulative stat: a draw's own outcome must never feed its own features.
    prior = grouped.shift(1)
    panel["trailing_freq"] = prior.groupby(panel["number"]).transform(
        lambda s: s.rolling(history_window, min_periods=1).mean()
    )
    panel["overall_freq"] = prior.groupby(panel["number"]).transform(
        lambda s: s.expanding(min_periods=1).mean()
    )
    # Each number's very first draw has no prior history: fall back to the
    # theoretical baseline rate rather than leaving/propagating a NaN.
    baseline_rate = game.draw_size / game.pool_size
    panel["trailing_freq"] = panel["trailing_freq"].fillna(baseline_rate)
    panel["overall_freq"] = panel["overall_freq"].fillna(baseline_rate)

    def _gap_since_last(s: pd.Series) -> pd.Series:
        idx = np.arange(len(s))
        last_hit = np.where(s.to_numpy() == 1, idx, np.nan)
        last_hit = pd.Series(last_hit).ffill().shift(1)
        return pd.Series(idx, index=s.index) - last_hit.to_numpy()

    panel["gap_since_last"] = panel.groupby("number")["appeared"].transform(_gap_since_last)
    panel["gap_since_last"] = panel["gap_since_last"].fillna(panel["gap_since_last"].max())

    draw_index_map = {d: i for i, d in enumerate(draw_ids)}
    panel["draw_index"] = panel["draw_id"].map(draw_index_map)

    panel = panel.merge(_markov_cooccurrence_frame(df, game), on=["draw_id", "number"], how="left")

    return panel.sort_values(["draw_id", "number"]).reset_index(drop=True)


def next_draw_features(df: pd.DataFrame, game: GameConfig, history_window: int = 20) -> pd.DataFrame:
    """One feature row per number for the upcoming, not-yet-drawn draw.

    Built from the *complete* observed history (there's no held-out label to
    protect against leaking, because the label doesn't exist yet) -- this is
    what `predictor.predict_next_draw_scores` scores with a model trained on
    `build_panel`'s historical rows.
    """
    n_draws = df["draw_id"].nunique()
    matrix = numbers_matrix(df, game)

    appeared = np.zeros((n_draws, game.pool_size), dtype=int)
    for t in range(n_draws):
        appeared[t, matrix[t] - 1] = 1

    overall_freq = appeared.mean(axis=0)
    trailing_freq = appeared[-history_window:].mean(axis=0)

    gap_since_last = np.full(game.pool_size, float(n_draws))
    for k in range(game.pool_size):
        hits = np.where(appeared[:, k] == 1)[0]
        if len(hits):
            gap_since_last[k] = (n_draws - 1) - hits[-1]

    markov_scores, cooc_scores = _markov_cooccurrence_matrices(df, game)

    return pd.DataFrame(
        {
            "number": np.arange(1, game.pool_size + 1),
            "trailing_freq": trailing_freq,
            "overall_freq": overall_freq,
            "gap_since_last": gap_since_last,
            "draw_index": n_draws,  # position right after the last observed draw
            "markov_score": markov_scores[-1],
            "cooccurrence_score": cooc_scores[-1],
        }
    )


FEATURE_COLUMNS = [
    "trailing_freq",
    "overall_freq",
    "gap_since_last",
    "draw_index",
    "markov_score",
    "cooccurrence_score",
]
