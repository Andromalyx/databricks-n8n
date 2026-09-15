"""Best-effort scraper for Vietlott Mega 6/45 draw history.

Vietlott does not publish a documented, stable public API. `ScraperConfig.base_url`
points at the commonly reverse-engineered "desktop search" endpoint used by several
open-source Vietlott trackers, but the exact JSON field names below (`RECORD_KEYS`)
may drift without notice. If `fetch_all` starts raising `ScraperParseError` or
returning zero rows:
  1. Hit the base_url in a browser/curl, inspect the JSON shape, and update
     `RECORD_KEYS` / `_parse_record` to match.
  2. Or skip the scraper entirely: export results manually to CSV and load them
     via `wealth_prediction.data.loader.load_csv` -- that's the schema every
     other module actually depends on, and it doesn't care how the CSV was made.

Run standalone:
    python -m wealth_prediction.data.scraper --out data/raw/draws.csv --pages 20
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from wealth_prediction.config import Config, load_config
from wealth_prediction.data.loader import save_csv

logger = logging.getLogger(__name__)

# Field names as observed on the reverse-engineered endpoint at time of writing.
# `_parse_record` isolates all of this guesswork so it's a one-place fix if it drifts.
RECORD_KEYS = {
    "draw_id": "issueIndex",
    "draw_date": "issueDate",
    "numbers": "firstPrizeNumbers",  # e.g. "01,05,12,23,34,45"
}


class ScraperError(RuntimeError):
    """Network/HTTP failure talking to the Vietlott endpoint."""


class ScraperParseError(RuntimeError):
    """Response received but didn't match the expected shape -- endpoint likely changed."""


def _session_with_retries(config) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=config.max_retries,
        backoff_factor=config.retry_backoff_s,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class VietlottScraper:
    def __init__(self, config: Config | None = None):
        self.config = (config or load_config()).scraper
        self.game = (config or load_config()).game
        self._session = _session_with_retries(self.config)

    def fetch_page(self, page: int) -> dict:
        params = {"pageSize": self.config.page_size, "pageIndex": page}
        try:
            resp = self._session.get(self.config.base_url, params=params, timeout=self.config.timeout_s)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise ScraperError(f"Request to {self.config.base_url} (page {page}) failed: {exc}") from exc
        except ValueError as exc:  # json decode error
            raise ScraperParseError(f"Non-JSON response on page {page}: {exc}") from exc

    def _parse_record(self, record: dict) -> dict:
        try:
            raw_numbers = record[RECORD_KEYS["numbers"]]
            numbers = [int(n) for n in str(raw_numbers).replace(" ", "").split(",")]
            if len(numbers) != self.game.draw_size:
                raise ScraperParseError(
                    f"Expected {self.game.draw_size} numbers, got {numbers} in record {record}"
                )
            draw_date = pd.to_datetime(record[RECORD_KEYS["draw_date"]]).date()
            row = {"draw_id": int(record[RECORD_KEYS["draw_id"]]), "draw_date": draw_date}
            row.update({f"n{i + 1}": n for i, n in enumerate(sorted(numbers))})
            return row
        except (KeyError, TypeError, ValueError) as exc:
            raise ScraperParseError(f"Unexpected record shape: {record}") from exc

    def fetch_all(self, max_pages: int | None = None) -> pd.DataFrame:
        max_pages = max_pages or self.config.max_pages
        rows: list[dict] = []
        for page in range(max_pages):
            logger.info("Fetching page %d/%d", page + 1, max_pages)
            payload = self.fetch_page(page)
            records = payload.get("data") or payload.get("items") or []
            if not records:
                logger.info("No more records at page %d; stopping.", page)
                break
            rows.extend(self._parse_record(r) for r in records)
            time.sleep(0.2)  # be polite to a free public endpoint
        if not rows:
            raise ScraperParseError(
                "Scraper returned zero rows -- the endpoint or response shape likely "
                "changed. See module docstring for how to fix RECORD_KEYS, or fall back "
                "to a manually exported CSV via wealth_prediction.data.loader.load_csv."
            )
        df = pd.DataFrame(rows).drop_duplicates(subset="draw_id").sort_values("draw_id")
        return df.reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="Output CSV path (default: config data.raw_path)")
    parser.add_argument("--pages", type=int, default=None, help="Max pages to fetch")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    scraper = VietlottScraper(config)
    df = scraper.fetch_all(max_pages=args.pages)
    out_path = args.out or config.data.raw_path
    save_csv(df, out_path, config.game)
    print(f"Wrote {len(df)} draws to {out_path} (as of {datetime.now().isoformat(timespec='seconds')})")


if __name__ == "__main__":
    main()
