"""Typed config loading for the wealth_prediction package.

Everything downstream (null-distribution simulation, chi-square degrees of
freedom, feature windows) depends on `GameConfig`, so we load it once from
YAML into dataclasses instead of passing loose dicts around.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


@dataclass(frozen=True)
class GameConfig:
    name: str = "Vietlott Mega 6/45"
    pool_size: int = 45
    draw_size: int = 6
    ticket_price_vnd: int = 10_000


@dataclass(frozen=True)
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    draws_file: str = "draws.csv"

    @property
    def raw_path(self) -> Path:
        return Path(self.raw_dir) / self.draws_file

    @property
    def processed_path(self) -> Path:
        return Path(self.processed_dir) / self.draws_file


@dataclass(frozen=True)
class ScraperConfig:
    base_url: str = "https://api.vietlott.vn/v5/desktop/vi/issearch/index/power645"
    page_size: int = 50
    max_pages: int = 20
    timeout_s: float = 15.0
    max_retries: int = 3
    retry_backoff_s: float = 2.0


@dataclass(frozen=True)
class AnalysisConfig:
    monte_carlo_sims: int = 20_000
    significance_level: float = 0.05
    random_seed: int = 42
    split_half_drift_test: bool = True


@dataclass(frozen=True)
class ModelingConfig:
    history_window: int = 20
    test_fraction: float = 0.2


@dataclass(frozen=True)
class Config:
    game: GameConfig = field(default_factory=GameConfig)
    data: DataConfig = field(default_factory=DataConfig)
    scraper: ScraperConfig = field(default_factory=ScraperConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)


def _section(raw: dict[str, Any], key: str, cls: type) -> Any:
    return cls(**raw.get(key, {})) if isinstance(raw.get(key), dict) else cls()


def load_config(path: str | Path | None = None) -> Config:
    """Load config.yaml, falling back to defaults for any missing section.

    A missing file is not an error: every dataclass above has sane defaults,
    so the pipeline is runnable out of the box (e.g. in tests or a fresh clone).
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        logger.warning("Config file %s not found; using built-in defaults.", path)
        return Config()

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    return Config(
        game=_section(raw, "game", GameConfig),
        data=_section(raw, "data", DataConfig),
        scraper=_section(raw, "scraper", ScraperConfig),
        analysis=_section(raw, "analysis", AnalysisConfig),
        modeling=_section(raw, "modeling", ModelingConfig),
    )
