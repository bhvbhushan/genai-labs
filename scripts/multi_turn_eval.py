"""Multi-turn conversation evaluation for the analytics pipeline.

Runs a fixed set of scripted multi-turn scenarios through
:class:`src.pipeline.AnalyticsPipeline` and reports per-turn latency,
token cost, status, and classifier intent. Each scenario is a sequence
of prompts sharing the same ``conversation_id``; the first turn has no
history and short-circuits the classifier, subsequent turns should be
routed through :class:`src.followup.FollowupClassifier`.

Output is a single JSON blob on stdout so the grading harness can
parse it. OTel exporters run by default (same convention as
``scripts/benchmark.py``); override via the standard
``OTEL_METRICS_EXPORTER=none`` / ``OTEL_TRACES_EXPORTER=none`` env
vars.

Usage
-----
::

    python scripts/multi_turn_eval.py
    python scripts/multi_turn_eval.py --runs 2   # repeat every scenario
"""

import argparse
import json
import statistics
import sys
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gaming_csv_to_db import (  # noqa: E402
    DEFAULT_CSV_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_TABLE_NAME,
    csv_to_sqlite,
)
from src.observability import shutdown_observability  # noqa: E402
from src.pipeline import AnalyticsPipeline  # noqa: E402
from src.types import PipelineOutput  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Scenarios
# ─────────────────────────────────────────────────────────────────────────────
#
# Each scenario is a list of prompts sent in order under one conversation_id.
# The classifier is expected to route turns 2+ through one of NEW_QUERY /
# FOLLOWUP_NEW_SQL / FOLLOWUP_REINTERPRET. We capture the actual intent it
# picked so a reviewer can see the routing behavior against realistic prompts.
SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "demographics_drill_down",
        "description": (
            "T1 establishes baseline; T2 narrows the scope (expect "
            "FOLLOWUP_NEW_SQL); T3 asks about the prior result "
            "(expect FOLLOWUP_REINTERPRET)."
        ),
        "turns": [
            "What is the average addiction level by gender?",
            "what about males specifically?",
            "which group had the highest value in the first result?",
        ],
    },
    {
        "name": "metric_swap",
        "description": (
            "T1 asks for a metric; T2 swaps the metric (expect "
            "FOLLOWUP_NEW_SQL); T3 restricts by age range on the new "
            "metric (expect FOLLOWUP_NEW_SQL)."
        ),
        "turns": [
            "Show average addiction level across age groups.",
            "now show anxiety score instead of addiction level.",
            "restrict that to respondents aged 18 to 25.",
        ],
    },
    {
        "name": "unrelated_followup",
        "description": (
            "T1 is demographic; T2 pivots to an unrelated question "
            "(expect NEW_QUERY — classifier should recognize no link)."
        ),
        "turns": [
            "How many respondents have addiction_level >= 5?",
            "What is the average sleep_hours across the dataset?",
        ],
    },
    {
        "name": "reinterpret_after_empty_rows",
        "description": (
            "T1 returns zero rows for a filter; T2 asks about the "
            "prior result — classifier should auto-downgrade "
            "REINTERPRET to FOLLOWUP_NEW_SQL since there's nothing "
            "to reinterpret."
        ),
        "turns": [
            "Show respondents with addiction_level > 100.",
            "what's the highest value in those results?",
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_gaming_db() -> Path:
    """Create the gaming DB from CSV if it isn't on disk."""
    if not DEFAULT_DB_PATH.exists():
        csv_to_sqlite(
            DEFAULT_CSV_PATH,
            DEFAULT_DB_PATH,
            DEFAULT_TABLE_NAME,
            if_exists="replace",
        )
    return DEFAULT_DB_PATH


def _turn_record(
    turn_idx: int,
    question: str,
    out: PipelineOutput,
    intent: str | None,
) -> dict[str, Any]:
    """One row of the per-turn table printed in JSON."""
    stats = out.total_llm_stats or {}
    return {
        "turn": turn_idx,
        "question": question,
        "status": out.status,
        "intent": intent,
        "sql": out.sql,
        "rows": len(out.rows),
        "ms": round(out.timings.get("total_ms", 0.0), 1),
        "tokens": int(stats.get("total_tokens", 0) or 0),
        "llm_calls": int(stats.get("llm_calls", 0) or 0),
    }


def _run_scenario(
    pipeline: AnalyticsPipeline,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    """Run one scenario end-to-end, returning per-turn records + summary."""
    convo_id = f"eval-{scenario['name']}-{uuid.uuid4().hex[:8]}"
    turn_records: list[dict[str, Any]] = []
    for i, question in enumerate(scenario["turns"], start=1):
        out = pipeline.run(question, conversation_id=convo_id)
        # After the call, read the latest Turn to capture the intent the
        # classifier picked (first turn has no history so intent is NEW_QUERY).
        history = pipeline._conversation_store.get_history(convo_id)
        recorded_intent = history[-1].intent if history else None
        turn_records.append(_turn_record(i, question, out, recorded_intent))

    total_ms = sum(t["ms"] for t in turn_records)
    total_tokens = sum(t["tokens"] for t in turn_records)
    total_calls = sum(t["llm_calls"] for t in turn_records)
    return {
        "name": scenario["name"],
        "description": scenario["description"],
        "turn_count": len(turn_records),
        "total_ms": round(total_ms, 1),
        "total_tokens": total_tokens,
        "total_llm_calls": total_calls,
        "turns": turn_records,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run scripted multi-turn conversations through the pipeline.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to repeat every scenario (default: 1).",
    )
    args = parser.parse_args()

    db_path = _ensure_gaming_db()
    pipeline = AnalyticsPipeline(db_path=db_path)

    all_scenario_results: list[dict[str, Any]] = []
    for _ in range(args.runs):
        for scenario in SCENARIOS:
            all_scenario_results.append(_run_scenario(pipeline, scenario))

    all_ms = [s["total_ms"] for s in all_scenario_results]
    all_tokens = [s["total_tokens"] for s in all_scenario_results]
    all_calls = [s["total_llm_calls"] for s in all_scenario_results]
    per_intent: dict[str, int] = {}
    for s in all_scenario_results:
        for t in s["turns"]:
            key = t["intent"] or "NEW_QUERY"
            per_intent[key] = per_intent.get(key, 0) + 1

    obs_dir = PROJECT_ROOT / ".observability"
    summary = {
        "runs": args.runs,
        "scenarios": len(all_scenario_results),
        "total_turns": sum(s["turn_count"] for s in all_scenario_results),
        "avg_scenario_ms": round(statistics.fmean(all_ms), 1) if all_ms else 0.0,
        "avg_scenario_tokens": (round(statistics.fmean(all_tokens), 1) if all_tokens else 0.0),
        "avg_scenario_llm_calls": (round(statistics.fmean(all_calls), 4) if all_calls else 0.0),
        "intent_breakdown": per_intent,
        "observability_files": {
            "metrics": str(obs_dir / "metrics.jsonl"),
            "traces": str(obs_dir / "traces.jsonl"),
        },
        "scenario_results": all_scenario_results,
    }

    # Ensure the final OTel export window lands before exit.
    shutdown_observability()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
