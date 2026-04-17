"""Tests for src.pipeline — status derivation, executor, and orchestration."""

import logging
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.pipeline import (  # noqa: E402
    AnalyticsPipeline,
    SQLiteExecutor,
    _aggregate_llm_stats,
    _derive_status,
)
from src.schema import SchemaCatalog  # noqa: E402
from src.types import (  # noqa: E402
    AnswerGenerationOutput,
    SQLExecutionOutput,
    SQLGenerationOutput,
    SQLValidationOutput,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_gen(sql: str | None = "SELECT 1", error: str | None = None) -> SQLGenerationOutput:
    return SQLGenerationOutput(
        sql=sql,
        timing_ms=1.0,
        llm_stats={
            "llm_calls": 1,
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
            "model": "test-model",
        },
        intermediate_outputs=[],
        error=error,
    )


def _make_val(
    is_valid: bool = True,
    validated_sql: str | None = "SELECT 1 LIMIT 1000",
    error: str | None = None,
) -> SQLValidationOutput:
    return SQLValidationOutput(
        is_valid=is_valid,
        validated_sql=validated_sql if is_valid else None,
        error=error,
        timing_ms=0.5,
    )


def _make_exec(
    rows: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> SQLExecutionOutput:
    rows = rows if rows is not None else [{"n": 1}]
    return SQLExecutionOutput(rows=rows, row_count=len(rows), timing_ms=2.0, error=error)


def _make_ans(answer: str = "The answer is 1.") -> AnswerGenerationOutput:
    return AnswerGenerationOutput(
        answer=answer,
        timing_ms=4.0,
        llm_stats={
            "llm_calls": 1,
            "prompt_tokens": 10,
            "completion_tokens": 6,
            "total_tokens": 16,
            "model": "test-model",
        },
        intermediate_outputs=[],
        error=None,
    )


def _seed_db(db_path: Path, n_rows: int = 5) -> None:
    """Create a tiny sqlite DB with table ``t(v INTEGER)`` populated with ``n_rows``."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE t (v INTEGER)")
        conn.executemany("INSERT INTO t (v) VALUES (?)", [(i,) for i in range(n_rows)])
        conn.commit()
    finally:
        conn.close()


def _make_pipeline(
    gen: SQLGenerationOutput | None = None,
    val: SQLValidationOutput | None = None,
    exec_: SQLExecutionOutput | None = None,
    ans: AnswerGenerationOutput | None = None,
) -> AnalyticsPipeline:
    """Build an AnalyticsPipeline with fully mocked components (no I/O)."""
    llm = MagicMock()
    llm.generate_sql.return_value = gen or _make_gen()
    llm.generate_answer.return_value = ans or _make_ans()

    validator = MagicMock()
    validator.validate.return_value = val or _make_val()

    executor = MagicMock()
    executor.run.return_value = exec_ or _make_exec()

    schema = SchemaCatalog(table="t", columns=())
    return AnalyticsPipeline(
        schema=schema,
        llm_client=llm,
        validator=validator,
        executor=executor,
    )


# ─────────────────────────────────────────────────────────────────────────────
# _derive_status
# ─────────────────────────────────────────────────────────────────────────────


class DeriveStatusTests(unittest.TestCase):
    def test_success(self) -> None:
        self.assertEqual(
            _derive_status(_make_gen(), _make_val(), _make_exec()),
            "success",
        )

    def test_error_when_gen_sql_none_and_gen_error(self) -> None:
        gen = _make_gen(sql=None, error="timeout")
        # val/exec are irrelevant but must be valid instances.
        val = _make_val(is_valid=False, error="No SQL provided")
        exec_ = _make_exec(rows=[])
        self.assertEqual(_derive_status(gen, val, exec_), "error")

    def test_unanswerable_when_gen_sql_none_and_no_error(self) -> None:
        gen = _make_gen(sql=None, error=None)
        val = _make_val(is_valid=False, error="No SQL provided")
        exec_ = _make_exec(rows=[])
        self.assertEqual(_derive_status(gen, val, exec_), "unanswerable")

    def test_invalid_sql(self) -> None:
        gen = _make_gen()
        val = _make_val(is_valid=False, error="Non-SELECT")
        exec_ = _make_exec(rows=[])
        self.assertEqual(_derive_status(gen, val, exec_), "invalid_sql")

    def test_error_when_exec_fails(self) -> None:
        gen = _make_gen()
        val = _make_val()
        exec_ = _make_exec(rows=[], error="no such table")
        self.assertEqual(_derive_status(gen, val, exec_), "error")


# ─────────────────────────────────────────────────────────────────────────────
# _aggregate_llm_stats
# ─────────────────────────────────────────────────────────────────────────────


class AggregateStatsTests(unittest.TestCase):
    def test_sums_counts_and_picks_first_model(self) -> None:
        a = {
            "llm_calls": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "model": "model-a",
        }
        b = {
            "llm_calls": 2,
            "prompt_tokens": 20,
            "completion_tokens": 7,
            "total_tokens": 27,
            "model": "model-b",
        }
        merged = _aggregate_llm_stats(a, b)
        self.assertEqual(merged["llm_calls"], 3)
        self.assertEqual(merged["prompt_tokens"], 30)
        self.assertEqual(merged["completion_tokens"], 12)
        self.assertEqual(merged["total_tokens"], 42)
        self.assertEqual(merged["model"], "model-a")

    def test_handles_empty_dicts(self) -> None:
        merged = _aggregate_llm_stats({}, {})
        self.assertEqual(merged["llm_calls"], 0)
        self.assertEqual(merged["prompt_tokens"], 0)
        self.assertEqual(merged["completion_tokens"], 0)
        self.assertEqual(merged["total_tokens"], 0)
        self.assertEqual(merged["model"], "unknown")

    def test_picks_second_model_when_first_empty(self) -> None:
        a = {"model": ""}
        b = {"model": "fallback-model"}
        merged = _aggregate_llm_stats(a, b)
        self.assertEqual(merged["model"], "fallback-model")


# ─────────────────────────────────────────────────────────────────────────────
# SQLiteExecutor
# ─────────────────────────────────────────────────────────────────────────────


class SQLiteExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.db_path = Path(self._tmpdir.name) / "t.sqlite"

    def test_happy_path(self) -> None:
        _seed_db(self.db_path, n_rows=5)
        out = SQLiteExecutor(self.db_path).run("SELECT v FROM t ORDER BY v")
        self.assertIsNone(out.error)
        self.assertEqual(out.row_count, 5)
        self.assertEqual(out.rows, [{"v": i} for i in range(5)])
        self.assertGreaterEqual(out.timing_ms, 0.0)

    def test_read_only_rejects_insert(self) -> None:
        _seed_db(self.db_path, n_rows=2)
        out = SQLiteExecutor(self.db_path).run("INSERT INTO t (v) VALUES (99)")
        self.assertIsNotNone(out.error)
        self.assertEqual(out.rows, [])
        self.assertEqual(out.row_count, 0)

    def test_none_sql(self) -> None:
        _seed_db(self.db_path)
        out = SQLiteExecutor(self.db_path).run(None)
        self.assertEqual(out.rows, [])
        self.assertEqual(out.row_count, 0)
        self.assertIsNone(out.error)
        self.assertGreaterEqual(out.timing_ms, 0.0)

    def test_bad_sql(self) -> None:
        _seed_db(self.db_path)
        out = SQLiteExecutor(self.db_path).run("SELECT * FROM nonexistent")
        self.assertIsNotNone(out.error)
        self.assertEqual(out.rows, [])

    def test_row_cap(self) -> None:
        _seed_db(self.db_path, n_rows=200)
        out = SQLiteExecutor(self.db_path, row_cap=10).run("SELECT v FROM t ORDER BY v")
        self.assertIsNone(out.error)
        self.assertEqual(out.row_count, 10)


# ─────────────────────────────────────────────────────────────────────────────
# AnalyticsPipeline orchestration — mocked components, no I/O
# ─────────────────────────────────────────────────────────────────────────────


class AnalyticsPipelineTests(unittest.TestCase):
    def test_success_end_to_end(self) -> None:
        pipe = _make_pipeline()
        result = pipe.run("What is the count?")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.question, "What is the count?")
        self.assertEqual(result.sql, "SELECT 1 LIMIT 1000")
        self.assertEqual(result.rows, [{"n": 1}])
        self.assertEqual(result.answer, "The answer is 1.")

    def test_gen_error_surfaces_as_error(self) -> None:
        pipe = _make_pipeline(gen=_make_gen(sql=None, error="timeout"))
        result = pipe.run("q")
        self.assertEqual(result.status, "error")

    def test_gen_none_no_error_is_unanswerable(self) -> None:
        pipe = _make_pipeline(gen=_make_gen(sql=None, error=None))
        result = pipe.run("q")
        self.assertEqual(result.status, "unanswerable")

    def test_validator_rejects_returns_invalid_sql(self) -> None:
        pipe = _make_pipeline(val=_make_val(is_valid=False, error="Non-SELECT"))
        result = pipe.run("q")
        self.assertEqual(result.status, "invalid_sql")
        self.assertIsNone(result.sql)

    def test_executor_error_is_error(self) -> None:
        pipe = _make_pipeline(exec_=_make_exec(rows=[], error="db is locked"))
        result = pipe.run("q")
        self.assertEqual(result.status, "error")

    def test_request_id_propagates_to_logs(self) -> None:
        pipe = _make_pipeline()
        with self.assertLogs("src.pipeline", level="INFO") as captured:
            pipe.run("q", request_id="test-rid-123")
        # ``log_event`` sets request_id via the ``extra=`` kwarg, which lands
        # on the LogRecord as an attribute (not in the message string).
        request_ids = [getattr(r, "request_id", None) for r in captured.records]
        self.assertIn("test-rid-123", request_ids)

    def test_timings_keys_and_nonnegative(self) -> None:
        pipe = _make_pipeline()
        result = pipe.run("q")
        expected = {
            "sql_generation_ms",
            "sql_validation_ms",
            "sql_execution_ms",
            "answer_generation_ms",
            "total_ms",
        }
        self.assertEqual(set(result.timings.keys()), expected)
        for v in result.timings.values():
            self.assertGreaterEqual(v, 0.0)

    def test_total_llm_stats_aggregates(self) -> None:
        pipe = _make_pipeline()
        result = pipe.run("q")
        # Both stages contribute 1 call, 8 and 16 total tokens respectively.
        self.assertEqual(result.total_llm_stats["llm_calls"], 2)
        self.assertEqual(result.total_llm_stats["total_tokens"], 24)
        self.assertEqual(result.total_llm_stats["model"], "test-model")

    def test_auto_request_id_when_missing(self) -> None:
        pipe = _make_pipeline()
        result = pipe.run("q")
        self.assertIsNotNone(result.request_id)
        # uuid4().hex[:16] → 16 hex chars.
        assert result.request_id is not None
        self.assertEqual(len(result.request_id), 16)


# Silence the root logger's JsonFormatter noise during mocked tests so
# unittest output stays readable. The src.pipeline logger still emits, and
# ``assertLogs`` on that logger captures the records directly.
logging.getLogger().setLevel(logging.WARNING)


if __name__ == "__main__":
    unittest.main()
