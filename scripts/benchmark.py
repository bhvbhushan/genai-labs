from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
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
    idx = min(len(sorted_vals) - 1, max(0, round((p / 100.0) * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of full prompt-set repetitions."
    )
    args = parser.parse_args()

    # Benchmark stays quiet by default so stdout is a clean JSON blob.
    os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
    os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")

    db_path = _ensure_gaming_db()
    root = Path(__file__).resolve().parents[1]
    prompts_path = root / "tests" / "public_prompts.json"

    pipeline = AnalyticsPipeline(db_path=db_path)
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))

    totals: list[float] = []
    total_tokens: list[float] = []
    llm_calls: list[float] = []
    status_counter: Counter[str] = Counter()
    success = 0
    count = 0

    for _ in range(args.runs):
        for prompt in prompts:
            result = pipeline.run(prompt)
            totals.append(result.timings["total_ms"])
            stats = result.total_llm_stats or {}
            total_tokens.append(float(stats.get("total_tokens", 0) or 0))
            llm_calls.append(float(stats.get("llm_calls", 0) or 0))
            status_counter[result.status] += 1
            success += int(result.status == "success")
            count += 1

    cache = pipeline._response_cache
    cache_stats = {
        "hits": cache.stats.hits,
        "misses": cache.stats.misses,
        "hit_rate": round(cache.stats.hit_rate, 4),
    }

    summary = {
        "runs": args.runs,
        "samples": count,
        "success_rate": round(success / count, 4) if count else 0.0,
        "avg_ms": round(statistics.fmean(totals), 2) if totals else 0.0,
        "p50_ms": round(percentile(totals, 50), 2),
        "p95_ms": round(percentile(totals, 95), 2),
        "avg_total_tokens": round(statistics.fmean(total_tokens), 2) if total_tokens else 0.0,
        "p50_total_tokens": round(percentile(total_tokens, 50), 2),
        "p95_total_tokens": round(percentile(total_tokens, 95), 2),
        "avg_llm_calls": round(statistics.fmean(llm_calls), 4) if llm_calls else 0.0,
        "status_breakdown": dict(status_counter),
        "cache_stats": cache_stats,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
