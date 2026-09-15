# wealth-prediction

A statistical audit toolkit for Vietlott Mega 6/45 lottery draws: does the
historical result set look like a fair, independent, uniform-random process,
or is there detectable structure (drift, bias, an exploitable pattern)?

## Read this first: what this project can and can't do

A regulated 6/45 draw picks 6 numbers uniformly at random from 45, without
replacement, independently each time. If it's actually fair, **no model, no
matter how sophisticated, can predict future numbers better than chance** —
that's not a limitation of this codebase, it's what "fair" mathematically
means. So this project is not a numbers-picking system. It's an audit:

1. **Randomness/normality testing** — do observed draws match what a fair
   process would produce (uniformity, independence, distribution shape)?
2. **Bias/drift detection** — is there a hot/cold number, or a change
   partway through history, that a fair process wouldn't produce?
3. **An honest ML evaluation** — train a real classifier on real features,
   evaluate it walk-forward (chronological holdout, never shuffled), and
   formally test whether it beats the trivial constant-probability baseline
   by more than sampling noise. If the lottery is fair, the expected and
   correct outcome is "no" — that's a successful result, not a failed model.

A rejected test is a prompt to investigate (more data, a data-collection
artifact, look for a specific mechanism) — not proof of fraud. With ~10
tests at alpha=0.05 you expect about one false positive by chance; the
report's Fisher-combined omnibus p-value is the number that actually matters,
not any single row in the table.

## Project layout

```
.
├── config/config.yaml                 # game params, scraper, analysis, modeling knobs
├── src/wealth_prediction/
│   ├── config.py                      # typed config loading
│   ├── data/
│   │   ├── synthetic.py               # fair/biased/drifting draw generators (also the Monte Carlo null engine)
│   │   ├── scraper.py                 # best-effort Vietlott history scraper
│   │   └── loader.py                  # CSV load/save + schema validation
│   ├── stats/
│   │   ├── uniformity.py              # chi-square goodness-of-fit per number
│   │   ├── normality.py               # Shapiro/D'Agostino/KS + Monte Carlo fair-null KS test
│   │   ├── independence.py            # runs test, Ljung-Box, pairwise co-occurrence, gap test
│   │   └── bias_detection.py          # hot/cold numbers, split-half drift, revenue correlation, scorecard
│   ├── modeling/
│   │   ├── features.py                # leakage-safe (backward-looking only) feature panel
│   │   └── predictor.py               # model vs. random-baseline walk-forward evaluation
│   ├── reporting/report.py            # figures + markdown report generation
│   └── pipeline.py                    # CLI orchestrator
├── scripts/run_analysis.py            # thin CLI wrapper
└── tests/                             # pytest suite (fair data should pass, biased/drifting should be flagged)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # installs the `wealth-prediction` CLI entry point
```

## Getting data

Three sources, chosen with `--source`:

- `synthetic` (default) — simulate a fair 6/45 process. Use this to see the
  pipeline run end-to-end with zero setup.
- `synthetic-biased` — simulate a deliberately biased process, to sanity-check
  that the detectors actually have power (they should flag this as non-fair).
- `csv` — load real history from `data/raw/draws.csv` (or `--data <path>`).
  Required columns: `draw_id, draw_date, n1..n6` (see
  `wealth_prediction.data.loader.validate_schema`).

`data/raw/draws.csv` is checked into this repo as the canonical dataset:
1,365+ real Mega 6/45 draws (2017-10-25 onward), converted from
[vietvudanh/vietlott-data](https://github.com/vietvudanh/vietlott-data)'s
`data/power645.jsonl` (MIT licensed; that project's crawler runs from a
Vietnam-based host since Vietlott blocks other regions). Refresh it by
re-pulling that repo and re-running the same `date/id/result` →
`draw_id/draw_date/n1..n6` conversion, rather than the scraper below —
Vietlott's own site is blocked by the same regional restriction from most
hosting providers, including this project's sandboxed dev environment.

To attempt a direct scrape instead:

```bash
# Best-effort scraper (Vietlott has no documented public API — verify the
# JSON field mapping in scraper.py against a live response first; see its
# module docstring for what to fix if it starts failing).
python -m wealth_prediction.data.scraper --out data/raw/draws.csv --pages 20

# Or just drop in a manually exported CSV matching the schema above.
```

## Running the audit

```bash
python -m wealth_prediction.pipeline --source synthetic --n-draws 500
python -m wealth_prediction.pipeline --source csv --data data/raw/draws.csv
python -m wealth_prediction.pipeline --source synthetic-biased --n-draws 500  # sanity check
```

Outputs land in `reports/`: `report.md` (test table + omnibus verdict + ML
summary), `figures/*.png`, and per-number CSVs (`hot_cold_numbers.csv`,
`per_number_significance.csv`).

Flags: `--out-dir`, `--config <path>`, `--no-model` (skip the ML evaluation,
which is the slowest step).

## Tests

```bash
pytest
```

The suite checks both directions: tests should **fail to reject** fairness on
`simulate_fair_draws` output, and **reject** it on `simulate_biased_draws` /
`simulate_drifting_draws` output — i.e., the detectors have to prove they can
actually detect something before we trust them on real data.

## Extending

- **Revenue/jackpot correlation**: `stats.bias_detection.revenue_correlation_test`
  takes an external DataFrame (draw_id + a sales/jackpot column) and tests
  whether per-draw sum correlates with it — the closest thing here to testing
  an "optimize revenue, avoid the winner" hypothesis. Vietlott doesn't publish
  sales data alongside results, so you'll need to supply your own series.
- **Different game**: everything derives from `config/config.yaml`'s
  `pool_size`/`draw_size` — point it at Power 6/55 or another lottery and the
  combinatorics (pairwise co-occurrence rate, gap geometric parameter, uniform
  expected counts) adjust automatically.
