#!/usr/bin/env python
"""Thin CLI wrapper: `python scripts/run_analysis.py [args]` == `python -m wealth_prediction.pipeline [args]`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wealth_prediction.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
