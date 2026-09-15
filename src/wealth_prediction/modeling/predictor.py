"""Honest test of "is there an exploitable pattern": model vs. random baseline.

If Vietlott 6/45 is fair, NO feature set can out-predict the constant
baseline P(number appears) = draw_size/pool_size out-of-sample, because that
constant already is the true generating probability. So this module isn't
trying to build a numbers-picking system -- it's a statistical instrument:
train a real classifier on real features, evaluate it walk-forward
(chronological split, never shuffled), and formally test whether its
predictions beat the trivial baseline by more than sampling noise would
produce. A confirmed "no" is itself the useful, expected result for a fair
lottery; a reproducible, significant "yes" would be the actual finding worth
investigating further.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from wealth_prediction.config import GameConfig, ModelingConfig
from wealth_prediction.modeling.features import FEATURE_COLUMNS, build_panel

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    model_log_loss: float
    baseline_log_loss: float
    model_brier: float
    baseline_brier: float
    n_test_samples: int
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    alpha: float

    @property
    def model_beats_baseline(self) -> bool:
        """True only if the model is BOTH numerically better AND the
        improvement is statistically significant -- guards against reporting
        noise as an "edge".
        """
        return self.model_log_loss < self.baseline_log_loss and self.wilcoxon_p_value < self.alpha

    def summary(self) -> str:
        verdict = "MODEL BEATS BASELINE (investigate further)" if self.model_beats_baseline else "no exploitable edge detected"
        return (
            f"log-loss: model={self.model_log_loss:.4f} vs baseline={self.baseline_log_loss:.4f}; "
            f"Brier: model={self.model_brier:.4f} vs baseline={self.baseline_brier:.4f}; "
            f"Wilcoxon p={self.wilcoxon_p_value:.4g} (n={self.n_test_samples}) -> {verdict}"
        )


def _chronological_split(panel: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    draw_ids = np.sort(panel["draw_id"].unique())
    split_idx = int(len(draw_ids) * (1 - test_fraction))
    if split_idx < 1 or split_idx >= len(draw_ids):
        raise ValueError(f"test_fraction={test_fraction} leaves no data on one side of {len(draw_ids)} draws.")
    cutoff = draw_ids[split_idx]
    return panel[panel["draw_id"] < cutoff], panel[panel["draw_id"] >= cutoff]


def evaluate_model_vs_baseline(
    df: pd.DataFrame,
    game: GameConfig,
    modeling: ModelingConfig,
    model=None,
    alpha: float = 0.05,
) -> EvaluationResult:
    """Train `model` (default: GradientBoostingClassifier) on a chronological
    train split, evaluate log-loss/Brier on the held-out future draws against
    the constant baseline, and Wilcoxon-test whether the per-sample log-loss
    improvement is significant.
    """
    if len(df) < 30:
        raise ValueError(f"Need at least 30 draws for a meaningful walk-forward split, got {len(df)}.")

    panel = build_panel(df, game, history_window=modeling.history_window)
    train, test = _chronological_split(panel, modeling.test_fraction)

    model = model or GradientBoostingClassifier(random_state=0)
    model.fit(train[FEATURE_COLUMNS], train["appeared"])

    y_test = test["appeared"].to_numpy()
    p_model = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    p_model = np.clip(p_model, 1e-6, 1 - 1e-6)  # avoid -inf log-loss on a perfectly confident wrong call

    baseline_rate = game.draw_size / game.pool_size
    p_baseline = np.full_like(p_model, baseline_rate)

    model_ll = log_loss(y_test, p_model, labels=[0, 1])
    baseline_ll = log_loss(y_test, p_baseline, labels=[0, 1])
    model_brier = brier_score_loss(y_test, p_model)
    baseline_brier = brier_score_loss(y_test, p_baseline)

    per_sample_model_ll = -(y_test * np.log(p_model) + (1 - y_test) * np.log(1 - p_model))
    per_sample_baseline_ll = -(y_test * np.log(p_baseline) + (1 - y_test) * np.log(1 - p_baseline))
    diff = per_sample_baseline_ll - per_sample_model_ll  # positive => model better on that sample
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(diff)
    except ValueError:
        # All-zero differences (identical predictions): no evidence of any edge.
        wilcoxon_stat, wilcoxon_p = 0.0, 1.0

    result = EvaluationResult(
        model_log_loss=model_ll,
        baseline_log_loss=baseline_ll,
        model_brier=model_brier,
        baseline_brier=baseline_brier,
        n_test_samples=len(y_test),
        wilcoxon_statistic=wilcoxon_stat,
        wilcoxon_p_value=wilcoxon_p,
        alpha=alpha,
    )
    logger.info(result.summary())
    return result


def default_models() -> dict[str, object]:
    """A couple of reasonable, cheap baselines to compare -- gradient boosting
    for nonlinear structure, logistic regression as a simpler linear check.
    """
    return {
        "gradient_boosting": GradientBoostingClassifier(random_state=0),
        "logistic_regression": LogisticRegression(max_iter=1000),
    }
