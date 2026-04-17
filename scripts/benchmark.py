from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# benchmark runs quiet — no metric/trace console spew. Override via env to test.
# Must be set BEFORE importing src.pipeline so observability sees the values.
os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")

from scripts.gaming_csv_to_db import (  # noqa: E402
    DEFAULT_CSV_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_TABLE_NAME,
    csv_to_sqlite,
)
from src.pipeline import AnalyticsPipeline  # noqa: E402


def _ensure_gaming_db() -> Path:
    """Ensure gaming mental health DB exists; create from CSV if missing."""
    if not DEFAULT_DB_PATH.exists():
        csv_to_sqlite(DEFAULT_CSV_PATH, DEFAULT_DB_PATH, DEFAULT_TABLE_NAME, if_exists="replace")
    return DEFAULT_DB_PATH


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, max(0, int(round((p / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="Number of full prompt-set repetitions.")
    args = parser.parse_args()

    db_path = _ensure_gaming_db()
    root = Path(__file__).resolve().parents[1]
    prompts_path = root / "tests" / "public_prompts.json"

    pipeline = AnalyticsPipeline(db_path=db_path)
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))

    totals: list[float] = []
    success = 0
    count = 0

    for _ in range(args.runs):
        for prompt in prompts:
            result = pipeline.run(prompt)
            totals.append(result.timings["total_ms"])
            success += int(result.status == "success")
            count += 1

    summary = {
        "runs": args.runs,
        "samples": count,
        "success_rate": round(success / count, 4) if count else 0.0,
        "avg_ms": round(statistics.fmean(totals), 2) if totals else 0.0,
        "p50_ms": round(percentile(totals, 50), 2),
        "p95_ms": round(percentile(totals, 95), 2),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
