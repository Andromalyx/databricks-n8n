"""Load, validate, and reshape Vietlott draw history.

Canonical schema (one row per draw):
    draw_id   : int, sequential
    draw_date : date
    n1..n6    : int in [1, pool_size], sorted ascending, no duplicates per row

All downstream stats/modeling code assumes this shape, so `load_csv` is the
single validation choke point -- garbage data fails loudly here rather than
producing silently-wrong p-values three modules later.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wealth_prediction.config import GameConfig

logger = logging.getLogger(__name__)


def number_columns(draw_size: int) -> list[str]:
    return [f"n{i + 1}" for i in range(draw_size)]


def validate_schema(df: pd.DataFrame, game: GameConfig) -> None:
    """Raise ValueError with a specific, actionable message on the first problem found."""
    required = {"draw_id", "draw_date", *number_columns(game.draw_size)}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Draw history is empty.")

    cols = number_columns(game.draw_size)
    numbers = df[cols].to_numpy()

    if not np.issubdtype(numbers.dtype, np.integer):
        # Coercion failures (e.g. blank cells) show up as floats/NaN or object dtype.
        if pd.isna(numbers).any():
            bad_rows = df.index[pd.isna(df[cols]).any(axis=1)].tolist()
            raise ValueError(f"Non-numeric or missing draw numbers in rows: {bad_rows[:10]}")

    out_of_range = (numbers < 1) | (numbers > game.pool_size)
    if out_of_range.any():
        bad_rows = df.index[out_of_range.any(axis=1)].tolist()
        raise ValueError(
            f"Numbers outside [1, {game.pool_size}] in rows: {bad_rows[:10]}"
        )

    dup_mask = np.array([len(set(row)) != len(row) for row in numbers])
    if dup_mask.any():
        bad_rows = df.index[dup_mask].tolist()
        raise ValueError(f"Duplicate numbers within a single draw in rows: {bad_rows[:10]}")

    if df["draw_id"].duplicated().any():
        raise ValueError("Duplicate draw_id values found.")


def load_csv(path: str | Path, game: GameConfig) -> pd.DataFrame:
    """Load and validate a draw-history CSV, sorted chronologically by draw_id."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Populate it via the scraper "
            "(python -m wealth_prediction.data.scraper) or synthetic data for testing."
        )
    df = pd.read_csv(path, parse_dates=["draw_date"])
    validate_schema(df, game)
    df = df.sort_values("draw_id").reset_index(drop=True)
    logger.info("Loaded %d draws from %s", len(df), path)
    return df


def save_csv(df: pd.DataFrame, path: str | Path, game: GameConfig) -> None:
    validate_schema(df, game)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d draws to %s", len(df), path)


def numbers_matrix(df: pd.DataFrame, game: GameConfig) -> np.ndarray:
    """Return the (n_draws, draw_size) integer matrix of drawn numbers."""
    return df[number_columns(game.draw_size)].to_numpy(dtype=int)


def to_long_format(df: pd.DataFrame, game: GameConfig) -> pd.DataFrame:
    """Melt n1..n6 into one (draw_id, draw_date, number) row per drawn ball.

    Convenient for frequency counts and groupby-based feature engineering.
    """
    cols = number_columns(game.draw_size)
    long_df = df.melt(
        id_vars=["draw_id", "draw_date"], value_vars=cols, var_name="slot", value_name="number"
    )
    return long_df.sort_values(["draw_id", "number"]).reset_index(drop=True)
