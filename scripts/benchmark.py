import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# OTel runs by default. Console exporters write to .observability/*.jsonl so
# stdout stays a clean JSON blob. Opt-out remains available for tests via
# OTEL_METRICS_EXPORTER=none / OTEL_TRACES_EXPORTER=none at invocation time.

from scripts.gaming_csv_to_db import (  # noqa: E402
    DEFAULT_CSV_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_TABLE_NAME,
    csv_to_sqlite,
)
from src.observability import shutdown_observability  # noqa: E402
from src.pipeline import AnalyticsPipeline  # noqa: E402
from src.types import PipelineOutput  # noqa: E402


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


def _is_cache_hit(result: PipelineOutput) -> bool:
    """A cache hit carries a ``{"cache_hit": True}`` marker in the first
    intermediate output of its SQL-generation stage (see
    ``src.response_cache._rewrite_for_hit``). Detecting via the marker keeps
    us independent of timing (which is zeroed on hits anyway).
    """
    gen = result.sql_generation
    if gen is None or not gen.intermediate_outputs:
        return False
    first = gen.intermediate_outputs[0]
    return bool(first.get("cache_hit"))


def _bucket_stats(
    totals: list[float],
    total_tokens: list[float],
    llm_calls: list[float],
) -> dict[str, Any]:
    """Aggregate latency/token/call percentiles for a single bucket of samples."""
    return {
        "count": len(totals),
        "avg_ms": round(statistics.fmean(totals), 2) if totals else 0.0,
        "p50_ms": round(percentile(totals, 50), 2),
        "p95_ms": round(percentile(totals, 95), 2),
        "avg_total_tokens": round(statistics.fmean(total_tokens), 2) if total_tokens else 0.0,
        "p50_total_tokens": round(percentile(total_tokens, 50), 2),
        "p95_total_tokens": round(percentile(total_tokens, 95), 2),
        "avg_llm_calls": round(statistics.fmean(llm_calls), 4) if llm_calls else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of full prompt-set repetitions."
    )
    args = parser.parse_args()

    db_path = _ensure_gaming_db()
    root = Path(__file__).resolve().parents[1]
    prompts_path = root / "tests" / "public_prompts.json"

    pipeline = AnalyticsPipeline(db_path=db_path)
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))

    # Combined samples (every request, hits + misses).
    totals: list[float] = []
    total_tokens: list[float] = []
    llm_calls: list[float] = []
    # Cache-hit samples only.
    hit_totals: list[float] = []
    hit_tokens: list[float] = []
    hit_calls: list[float] = []
    # Cache-miss samples only.
    miss_totals: list[float] = []
    miss_tokens: list[float] = []
    miss_calls: list[float] = []

    status_counter: Counter[str] = Counter()
    success = 0
    count = 0

    for _ in range(args.runs):
        for prompt in prompts:
            result = pipeline.run(prompt)
            ms = result.timings["total_ms"]
            stats = result.total_llm_stats or {}
            tokens = float(stats.get("total_tokens", 0) or 0)
            calls = float(stats.get("llm_calls", 0) or 0)

            totals.append(ms)
            total_tokens.append(tokens)
            llm_calls.append(calls)

            if _is_cache_hit(result):
                hit_totals.append(ms)
                hit_tokens.append(tokens)
                hit_calls.append(calls)
            else:
                miss_totals.append(ms)
                miss_tokens.append(tokens)
                miss_calls.append(calls)

            status_counter[result.status] += 1
            success += int(result.status == "success")
            count += 1

    cache = pipeline._response_cache
    cache_stats = {
        "hits": cache.stats.hits,
        "misses": cache.stats.misses,
        "hit_rate": round(cache.stats.hit_rate, 4),
    }

    obs_dir = root / ".observability"
    observability_files = {
        "metrics": str(obs_dir / "metrics.jsonl"),
        "traces": str(obs_dir / "traces.jsonl"),
    }

    summary: dict[str, Any] = {
        "runs": args.runs,
        "samples": count,
        "success_rate": round(success / count, 4) if count else 0.0,
        "status_breakdown": dict(status_counter),
        "cache_stats": cache_stats,
        "observability_files": observability_files,
        "combined": _bucket_stats(totals, total_tokens, llm_calls),
    }
    if miss_totals:
        summary["cache_misses_only"] = _bucket_stats(miss_totals, miss_tokens, miss_calls)
    if hit_totals:
        summary["cache_hits_only"] = _bucket_stats(hit_totals, hit_tokens, hit_calls)

    # Final flush so the ~5s metric reader window is captured before exit.
    shutdown_observability()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
