"""Feature engineering for the "can a model beat the baseline" test.

Builds a long panel with one row per (draw, number) and a binary label
`appeared`. Features are strictly backward-looking (computed from draws
*before* the current one) so the walk-forward evaluation in predictor.py
can't leak future information -- the single most common way this kind of
analysis silently cheats itself into a fake "edge".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wealth_prediction.config import GameConfig
from wealth_prediction.data.loader import to_long_format


def build_panel(df: pd.DataFrame, game: GameConfig, history_window: int = 20) -> pd.DataFrame:
    """One row per (draw_id, number) with label `appeared` and lagged features.

    Features per row (all computed using only draws strictly before draw_id):
      - trailing_freq: appearance rate over the last `history_window` draws
      - overall_freq: appearance rate over all prior draws
      - gap_since_last: draws since this number last appeared (capped at draw index)
      - draw_index: position in history, 0-based (lets a model learn any global drift)
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

    return panel.sort_values(["draw_id", "number"]).reset_index(drop=True)


FEATURE_COLUMNS = ["trailing_freq", "overall_freq", "gap_since_last", "draw_index"]
