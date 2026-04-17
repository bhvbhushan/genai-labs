"""Tests for src.pipeline — status derivation, executor, and orchestration."""

import logging
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.conversation import ConversationStore, Turn  # noqa: E402
from src.followup import FollowupResponse  # noqa: E402
from src.pipeline import (  # noqa: E402
    AnalyticsPipeline,
    SQLiteExecutor,
    _aggregate_llm_stats,
    _derive_status,
)
from src.response_cache import ResponseCache  # noqa: E402
from src.result_validator import ResultValidator  # noqa: E402
from src.schema import ColumnInfo, SchemaCatalog  # noqa: E402
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
    conversation_store: ConversationStore | None = None,
    followup_classifier: Any = None,
    response_cache: ResponseCache | None = None,
) -> AnalyticsPipeline:
    """Build an AnalyticsPipeline with fully mocked components (no I/O)."""
    llm = MagicMock()
    llm.model_name = "test-model"
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
        conversation_store=conversation_store,
        followup_classifier=followup_classifier,
        response_cache=response_cache,
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


# ─────────────────────────────────────────────────────────────────────────────
# Multi-turn: conversation_id + followup classifier integration
# ─────────────────────────────────────────────────────────────────────────────


class MultiTurnTests(unittest.TestCase):
    def test_no_conversation_id_skips_classifier_and_store(self) -> None:
        store = ConversationStore()
        classifier = MagicMock()
        pipe = _make_pipeline(
            conversation_store=store,
            followup_classifier=classifier,
        )
        result = pipe.run("q")
        self.assertEqual(result.status, "success")
        self.assertFalse(classifier.classify_and_rewrite.called)
        self.assertEqual(len(store), 0)

    def test_first_turn_with_conversation_id_skips_classifier_and_appends(self) -> None:
        store = ConversationStore()
        classifier = MagicMock()
        pipe = _make_pipeline(
            conversation_store=store,
            followup_classifier=classifier,
        )
        result = pipe.run("q1", conversation_id="c1")
        self.assertEqual(result.status, "success")
        # Empty history → classifier never invoked.
        self.assertFalse(classifier.classify_and_rewrite.called)
        self.assertIn("c1", store)
        history = store.get_history("c1")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].question, "q1")
        self.assertEqual(history[0].intent, "NEW_QUERY")

    def test_second_turn_invokes_classifier_with_history(self) -> None:
        store = ConversationStore()
        classifier = MagicMock()
        classifier.classify_and_rewrite.return_value = FollowupResponse(
            intent="NEW_QUERY",
            rewritten_question="q2",
            reuses_prior_rows=False,
        )
        pipe = _make_pipeline(
            conversation_store=store,
            followup_classifier=classifier,
        )
        pipe.run("q1", conversation_id="c1")
        pipe.run("q2", conversation_id="c1")

        self.assertTrue(classifier.classify_and_rewrite.called)
        call_args = classifier.classify_and_rewrite.call_args
        self.assertEqual(call_args.args[0], "q2")
        history_passed: list[Turn] = call_args.args[1]
        self.assertEqual(len(history_passed), 1)
        self.assertEqual(history_passed[0].question, "q1")
        # Two turns stored.
        self.assertEqual(len(store.get_history("c1")), 2)

    def test_followup_new_sql_uses_rewritten_question(self) -> None:
        store = ConversationStore()
        classifier = MagicMock()
        classifier.classify_and_rewrite.return_value = FollowupResponse(
            intent="FOLLOWUP_NEW_SQL",
            rewritten_question="addiction level distribution for males",
            reuses_prior_rows=False,
        )
        pipe = _make_pipeline(
            conversation_store=store,
            followup_classifier=classifier,
        )
        # Seed with prior turn so history is non-empty.
        pipe.run("addiction by gender", conversation_id="c1")

        llm_mock = cast("MagicMock", pipe._llm)
        llm_mock.generate_sql.reset_mock()
        llm_mock.generate_answer.reset_mock()

        result = pipe.run("what about males?", conversation_id="c1")
        self.assertEqual(result.status, "success")
        # generate_sql was called with the rewritten question.
        args = llm_mock.generate_sql.call_args
        self.assertEqual(args.args[0], "addiction level distribution for males")
        # Stored turn preserves the RAW user question + rewritten.
        history = store.get_history("c1")
        self.assertEqual(history[-1].question, "what about males?")
        self.assertEqual(
            history[-1].rewritten_question,
            "addiction level distribution for males",
        )
        self.assertEqual(history[-1].intent, "FOLLOWUP_NEW_SQL")

    def test_reinterpret_skips_sql_and_reuses_prior_rows(self) -> None:
        store = ConversationStore()
        classifier = MagicMock()
        classifier.classify_and_rewrite.return_value = FollowupResponse(
            intent="FOLLOWUP_REINTERPRET",
            rewritten_question="which row is the highest?",
            reuses_prior_rows=True,
        )
        prior_exec = _make_exec(rows=[{"v": 1}, {"v": 9}, {"v": 4}])
        pipe = _make_pipeline(
            conversation_store=store,
            followup_classifier=classifier,
            exec_=prior_exec,
        )
        pipe.run("show values", conversation_id="c1")

        llm_mock = cast("MagicMock", pipe._llm)
        llm_mock.generate_sql.reset_mock()
        llm_mock.generate_answer.reset_mock()
        # Configure answer for the reinterpret call.
        llm_mock.generate_answer.return_value = _make_ans(answer="The highest is 9.")

        result = pipe.run("which is highest?", conversation_id="c1")

        self.assertEqual(result.status, "success")
        self.assertFalse(llm_mock.generate_sql.called)
        self.assertTrue(llm_mock.generate_answer.called)
        ga_args = llm_mock.generate_answer.call_args
        # Prior SQL + rows were forwarded.
        self.assertEqual(ga_args.args[0], "which is highest?")
        self.assertEqual(ga_args.args[1], "SELECT 1 LIMIT 1000")
        self.assertEqual(ga_args.args[2], [{"v": 1}, {"v": 9}, {"v": 4}])
        self.assertEqual(result.rows, [{"v": 1}, {"v": 9}, {"v": 4}])
        self.assertEqual(result.answer, "The highest is 9.")
        # All five timings present; skipped stages are zero.
        for key in (
            "sql_generation_ms",
            "sql_validation_ms",
            "sql_execution_ms",
            "answer_generation_ms",
            "total_ms",
        ):
            self.assertIn(key, result.timings)
        self.assertEqual(result.timings["sql_generation_ms"], 0.0)
        self.assertEqual(result.timings["sql_validation_ms"], 0.0)
        self.assertEqual(result.timings["sql_execution_ms"], 0.0)

    def test_new_query_intent_preserves_original_question(self) -> None:
        store = ConversationStore()
        classifier = MagicMock()
        classifier.classify_and_rewrite.return_value = FollowupResponse(
            intent="NEW_QUERY",
            rewritten_question="unrelated",
            reuses_prior_rows=False,
        )
        pipe = _make_pipeline(
            conversation_store=store,
            followup_classifier=classifier,
        )
        pipe.run("q1", conversation_id="c1")

        llm_mock = cast("MagicMock", pipe._llm)
        llm_mock.generate_sql.reset_mock()

        pipe.run("completely different question", conversation_id="c1")
        # NEW_QUERY does NOT rewrite: generate_sql sees the original.
        args = llm_mock.generate_sql.call_args
        self.assertEqual(args.args[0], "completely different question")

        history = store.get_history("c1")
        self.assertEqual(history[-1].intent, "NEW_QUERY")
        # NEW_QUERY stores no rewritten_question.
        self.assertIsNone(history[-1].rewritten_question)


# ─────────────────────────────────────────────────────────────────────────────
# Response cache integration
# ─────────────────────────────────────────────────────────────────────────────


class ResponseCacheIntegrationTests(unittest.TestCase):
    def test_first_call_runs_pipeline_then_caches(self) -> None:
        cache = ResponseCache()
        pipe = _make_pipeline(response_cache=cache)
        result = pipe.run("same question")
        self.assertEqual(result.status, "success")
        # Full pipeline ran once.
        llm_mock = cast("MagicMock", pipe._llm)
        self.assertEqual(llm_mock.generate_sql.call_count, 1)
        # And the result is now cached.
        self.assertIn("same question", cache)

    def test_second_call_is_a_cache_hit(self) -> None:
        cache = ResponseCache()
        pipe = _make_pipeline(response_cache=cache)
        pipe.run("same question")
        llm_mock = cast("MagicMock", pipe._llm)
        llm_mock.generate_sql.reset_mock()
        llm_mock.generate_answer.reset_mock()

        result = pipe.run("same question")
        self.assertEqual(result.status, "success")
        # No LLM work on the second call.
        self.assertFalse(llm_mock.generate_sql.called)
        self.assertFalse(llm_mock.generate_answer.called)
        # Zero'd stats on a cache hit.
        self.assertEqual(result.total_llm_stats["total_tokens"], 0)
        # And the cache_hit marker is visible.
        self.assertEqual(
            result.sql_generation.intermediate_outputs[0],
            {"cache_hit": True},
        )

    def test_multi_turn_bypasses_cache(self) -> None:
        cache = ResponseCache()
        store = ConversationStore()
        classifier = MagicMock()
        classifier.classify_and_rewrite.return_value = FollowupResponse(
            intent="NEW_QUERY",
            rewritten_question="q",
            reuses_prior_rows=False,
        )
        pipe = _make_pipeline(
            response_cache=cache,
            conversation_store=store,
            followup_classifier=classifier,
        )
        pipe.run("q", conversation_id="c1")
        pipe.run("q", conversation_id="c1")
        # Cache was not populated — conversation paths always go full pipeline.
        self.assertNotIn("q", cache)
        llm_mock = cast("MagicMock", pipe._llm)
        # Both turns hit generate_sql.
        self.assertEqual(llm_mock.generate_sql.call_count, 2)

    def test_failed_status_is_not_cached(self) -> None:
        cache = ResponseCache()
        # Validator fails → invalid_sql status.
        pipe = _make_pipeline(
            val=_make_val(is_valid=False, error="Non-SELECT"),
            response_cache=cache,
        )
        result1 = pipe.run("bad q")
        self.assertEqual(result1.status, "invalid_sql")
        self.assertNotIn("bad q", cache)
        llm_mock = cast("MagicMock", pipe._llm)
        # Second call re-runs the pipeline.
        pipe.run("bad q")
        self.assertEqual(llm_mock.generate_sql.call_count, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Result validator integration
# ─────────────────────────────────────────────────────────────────────────────


def _schema_with_age_range() -> SchemaCatalog:
    return SchemaCatalog(
        table="t",
        columns=(
            ColumnInfo(
                name="age",
                sql_type="INTEGER",
                kind="numeric",
                min_value=13.0,
                max_value=59.0,
            ),
        ),
    )


class ResultValidatorIntegrationTests(unittest.TestCase):
    def test_warnings_are_logged_but_status_still_success(self) -> None:
        schema = _schema_with_age_range()
        llm = MagicMock()
        llm.model_name = "test-model"
        llm.generate_sql.return_value = _make_gen(sql="SELECT age FROM t WHERE 1=1")
        llm.generate_answer.return_value = _make_ans()
        validator = MagicMock()
        validator.validate.return_value = _make_val(validated_sql="SELECT age FROM t WHERE 1=1")
        executor = MagicMock()
        # Age 200 is outside the declared max 59.
        executor.run.return_value = _make_exec(rows=[{"age": 200}])

        pipe = AnalyticsPipeline(
            schema=schema,
            llm_client=llm,
            validator=validator,
            executor=executor,
            result_validator=ResultValidator(schema),
        )
        with self.assertLogs("src.pipeline", level="INFO") as captured:
            result = pipe.run("out of range query")

        # Pipeline still returned success — warnings are non-fatal.
        self.assertEqual(result.status, "success")
        # And the warning landed in the logs.
        messages = [r.getMessage() for r in captured.records]
        self.assertIn("result_validation_warning", messages)

    def test_no_warnings_when_rows_in_range(self) -> None:
        schema = _schema_with_age_range()
        llm = MagicMock()
        llm.model_name = "test-model"
        llm.generate_sql.return_value = _make_gen(sql="SELECT age FROM t WHERE 1=1")
        llm.generate_answer.return_value = _make_ans()
        validator = MagicMock()
        validator.validate.return_value = _make_val(validated_sql="SELECT age FROM t WHERE 1=1")
        executor = MagicMock()
        executor.run.return_value = _make_exec(rows=[{"age": 25}])

        pipe = AnalyticsPipeline(
            schema=schema,
            llm_client=llm,
            validator=validator,
            executor=executor,
            result_validator=ResultValidator(schema),
        )
        with self.assertLogs("src.pipeline", level="INFO") as captured:
            result = pipe.run("in-range query")

        self.assertEqual(result.status, "success")
        messages = [r.getMessage() for r in captured.records]
        self.assertNotIn("result_validation_warning", messages)


# Silence the root logger's JsonFormatter noise during mocked tests so
# unittest output stays readable. The src.pipeline logger still emits, and
# ``assertLogs`` on that logger captures the records directly.
logging.getLogger().setLevel(logging.WARNING)


if __name__ == "__main__":
    unittest.main()
