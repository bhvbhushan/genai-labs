"""AnalyticsPipeline — thin orchestrator wiring Lanes A+B+C together.

The pipeline is ~four stages of orchestration and zero business logic. Each
stage is a :class:`src.observability.Timer`-wrapped call into the component
that owns the work:

1. ``generate_sql``      → :class:`src.llm_client.OpenRouterLLMClient`
2. ``validate``          → :class:`src.validator.SQLValidator`
3. ``execute``           → :class:`SQLiteExecutor`
4. ``generate_answer``   → :class:`src.llm_client.OpenRouterLLMClient`

Status derivation is a single decision table in :func:`_derive_status` so
the mapping from (gen, val, exec) stage outputs to ``PipelineOutput.status``
lives in exactly one place. ``total_llm_stats`` aggregates across the two
LLM stages so the evaluation harness sees the real token cost of the whole
request.

Multi-turn: when ``run`` is called with a ``conversation_id`` that has prior
history, the :class:`src.followup.FollowupClassifier` classifies the new
question against the last 4 turns and routes it. NEW_QUERY preserves the
default flow (pass-through). FOLLOWUP_NEW_SQL substitutes a self-contained
rewrite for the generation stage. FOLLOWUP_REINTERPRET skips SQL gen+exec
and calls ``generate_answer`` on the prior turn's cached rows. Without a
``conversation_id`` the pipeline behaves exactly as before Lane E.
"""

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from src.config import Settings, get_settings
from src.conversation import ConversationStore, Turn
from src.followup import FollowupClassifier, FollowupResponse
from src.llm_client import OpenRouterLLMClient
from src.observability import (
    Timer,
    configure_observability,
    get_logger,
    get_tracer,
    log_event,
    pipeline_requests_total,
    response_cache_hits_total,
    response_cache_misses_total,
    stage_duration_ms,
)
from src.response_cache import ResponseCache
from src.schema import SchemaCatalog
from src.types import (
    AnswerGenerationOutput,
    PipelineOutput,
    SQLExecutionOutput,
    SQLGenerationOutput,
    SQLValidationOutput,
)
from src.validator import SQLValidator

_logger = get_logger(__name__)

# Progress-handler check interval (VM operations between abort checks).
_PROGRESS_HANDLER_N = 100_000

# History window passed to the followup classifier.
_FOLLOWUP_HISTORY_WINDOW = 4


class SQLiteExecutor:
    """Read-only SQLite executor with row cap + statement deadline.

    The connection is opened via the SQLite URI form ``file:path?mode=ro`` so
    writes fail at the database level even before our AST validator catches
    them. A ``progress_handler`` checks a monotonic deadline every
    ~100k VM operations and aborts the running statement if exceeded.
    """

    def __init__(self, db_path: Path, *, row_cap: int = 100, timeout_s: float = 10.0) -> None:
        self._db_path = db_path
        self._row_cap = row_cap
        self._timeout_s = timeout_s

    def run(self, sql: str | None) -> SQLExecutionOutput:
        """Execute ``sql`` read-only; return rows + error inside the output dataclass."""
        start = time.perf_counter()
        if sql is None:
            return SQLExecutionOutput(
                rows=[],
                row_count=0,
                timing_ms=(time.perf_counter() - start) * 1000.0,
                error=None,
            )

        rows: list[dict[str, Any]] = []
        error: str | None = None
        try:
            uri = f"file:{self._db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=self._timeout_s)
            try:
                conn.row_factory = sqlite3.Row
                deadline = time.perf_counter() + self._timeout_s

                def _abort() -> int:
                    # Return non-zero to abort the currently executing statement.
                    return 1 if time.perf_counter() > deadline else 0

                conn.set_progress_handler(_abort, _PROGRESS_HANDLER_N)
                cur = conn.cursor()
                cur.execute(sql)
                raw = cur.fetchmany(self._row_cap)
                rows = [dict(r) for r in raw]
            finally:
                conn.close()
        except Exception as exc:
            error = str(exc)
            rows = []

        return SQLExecutionOutput(
            rows=rows,
            row_count=len(rows),
            timing_ms=(time.perf_counter() - start) * 1000.0,
            error=error,
        )


class AnalyticsPipeline:
    """Orchestrates SQL generation → validation → execution → answer generation."""

    def __init__(
        self,
        db_path: Path | None = None,
        *,
        settings: Settings | None = None,
        schema: SchemaCatalog | None = None,
        llm_client: OpenRouterLLMClient | None = None,
        executor: SQLiteExecutor | None = None,
        validator: SQLValidator | None = None,
        conversation_store: ConversationStore | None = None,
        followup_classifier: FollowupClassifier | None = None,
        response_cache: ResponseCache | None = None,
    ) -> None:
        configure_observability()
        self._settings = settings or get_settings()
        # Constructor arg wins over settings so tests/benchmark can point at
        # alternate databases without mutating the global settings singleton.
        resolved_db_path = db_path if db_path is not None else self._settings.db_path
        self._schema = schema or SchemaCatalog.from_db(
            resolved_db_path,
            self._settings.table_name,
        )
        self._llm = llm_client or OpenRouterLLMClient(
            api_key=self._settings.openrouter_api_key,
            model=self._settings.model,
            schema=self._schema,
            timeout_s=self._settings.llm_timeout_s,
            retries=self._settings.llm_retries,
            retry_base_s=self._settings.llm_retry_base_s,
            max_rows_to_llm=self._settings.max_rows_to_llm,
        )
        self._executor = executor or SQLiteExecutor(
            resolved_db_path,
            row_cap=self._settings.max_rows_return,
            timeout_s=self._settings.sql_timeout_s,
        )
        self._validator = validator or SQLValidator(
            self._schema,
            row_limit=self._settings.sql_row_limit,
        )
        self._conversation_store = (
            conversation_store if conversation_store is not None else ConversationStore()
        )
        self._followup_classifier = (
            followup_classifier
            if followup_classifier is not None
            else FollowupClassifier(self._llm)
        )
        self._response_cache = response_cache if response_cache is not None else ResponseCache()
        log_event(
            _logger,
            "pipeline_initialized",
            table=self._settings.table_name,
            columns=len(self._schema.columns),
        )

    def run(
        self,
        question: str,
        request_id: str | None = None,
        conversation_id: str | None = None,
    ) -> PipelineOutput:
        """Execute the four stages and return a fully-populated ``PipelineOutput``.

        When ``conversation_id`` is provided and has prior history, the
        followup classifier may rewrite the question (FOLLOWUP_NEW_SQL) or
        skip the SQL path entirely (FOLLOWUP_REINTERPRET). On success the
        turn is appended to the conversation store.
        """
        rid = request_id or uuid.uuid4().hex[:16]
        log_event(
            _logger,
            "pipeline_start",
            request_id=rid,
            question=question,
            conversation_id=conversation_id,
        )

        # Response cache: single-turn only. Multi-turn goes through the
        # conversation-aware path so follow-up classification still sees the
        # full history. Lookup happens BEFORE any LLM work.
        if conversation_id is None:
            cached = self._response_cache.get(question)
            if cached is not None:
                cached.request_id = rid
                log_event(
                    _logger,
                    "response_cache_hit",
                    request_id=rid,
                    question=question,
                )
                if response_cache_hits_total is not None:
                    response_cache_hits_total.add(1)
                if pipeline_requests_total is not None:
                    pipeline_requests_total.add(1, {"status": "success", "cache": "hit"})
                return cached
            if response_cache_misses_total is not None:
                response_cache_misses_total.add(1)

        # Multi-turn: classify + maybe rewrite against recent history.
        effective_question = question
        followup: FollowupResponse | None = None
        if conversation_id and conversation_id in self._conversation_store:
            history = self._conversation_store.last_turns(conversation_id, _FOLLOWUP_HISTORY_WINDOW)
            if history:
                followup = self._followup_classifier.classify_and_rewrite(
                    question, history, request_id=rid
                )
                log_event(
                    _logger,
                    "followup_classified",
                    request_id=rid,
                    intent=followup.intent,
                    reuses_prior_rows=followup.reuses_prior_rows,
                )
                if followup.intent == "FOLLOWUP_REINTERPRET":
                    output = self._reinterpret_prior(question, history[-1], rid, followup)
                    self._record_turn(conversation_id, question, followup, output)
                    return output
                if followup.intent == "FOLLOWUP_NEW_SQL":
                    effective_question = followup.rewritten_question
                # NEW_QUERY falls through with the original question.

        output = self._run_pipeline(effective_question, question, rid)

        if conversation_id:
            self._record_turn(conversation_id, question, followup, output)
        elif output.status == "success":
            # Cache only successful single-turn results. Errors / unanswerables
            # must be retried so they get a fresh chance to succeed.
            self._response_cache.put(question, output)

        return output

    # ------------------------------------------------------------------
    # Internal: standard pipeline (Lane D flow preserved byte-for-byte)
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        effective_question: str,
        original_question: str,
        rid: str,
    ) -> PipelineOutput:
        """Execute stages 1-4 against ``effective_question``.

        ``original_question`` is reported in the ``PipelineOutput.question``
        field so the caller sees exactly what they asked (not the rewrite);
        ``effective_question`` is what the LLM / answer stage consumes.
        """
        tracer = get_tracer()
        pipeline_start = time.perf_counter()

        with tracer.start_as_current_span("pipeline.run", attributes={"request_id": rid}):
            # Stage 1: SQL generation.
            with Timer("sql_generation"):
                sql_gen = self._llm.generate_sql(effective_question, request_id=rid)
            if stage_duration_ms is not None:
                stage_duration_ms.record(sql_gen.timing_ms, {"stage": "sql_generation"})

            # Stage 2: SQL validation.
            with Timer("sql_validation"):
                sql_val = self._validator.validate(sql_gen.sql, request_id=rid)
            if stage_duration_ms is not None:
                stage_duration_ms.record(sql_val.timing_ms, {"stage": "sql_validation"})

            # Stage 3: Execution (only when validation passed).
            sql_for_exec = sql_val.validated_sql if sql_val.is_valid else None
            with Timer("sql_execution"):
                sql_exec = self._executor.run(sql_for_exec)
            if stage_duration_ms is not None:
                stage_duration_ms.record(sql_exec.timing_ms, {"stage": "sql_execution"})

            # Stage 4: Answer generation.
            with Timer("answer_generation"):
                ans = self._llm.generate_answer(
                    effective_question,
                    sql_for_exec,
                    sql_exec.rows,
                    request_id=rid,
                )
            if stage_duration_ms is not None:
                stage_duration_ms.record(ans.timing_ms, {"stage": "answer_generation"})

        status = _derive_status(sql_gen, sql_val, sql_exec)

        total_ms = (time.perf_counter() - pipeline_start) * 1000.0
        timings: dict[str, float] = {
            "sql_generation_ms": sql_gen.timing_ms,
            "sql_validation_ms": sql_val.timing_ms,
            "sql_execution_ms": sql_exec.timing_ms,
            "answer_generation_ms": ans.timing_ms,
            "total_ms": total_ms,
        }

        total_llm_stats = _aggregate_llm_stats(sql_gen.llm_stats, ans.llm_stats)

        if pipeline_requests_total is not None:
            pipeline_requests_total.add(1, {"status": status})

        log_event(
            _logger,
            "pipeline_complete",
            request_id=rid,
            stage="pipeline",
            duration_ms=total_ms,
            status=status,
            tokens=total_llm_stats["total_tokens"],
        )

        return PipelineOutput(
            status=status,
            question=original_question,
            request_id=rid,
            sql_generation=sql_gen,
            sql_validation=sql_val,
            sql_execution=sql_exec,
            answer_generation=ans,
            sql=sql_val.validated_sql if sql_val.is_valid else None,
            rows=sql_exec.rows,
            answer=ans.answer,
            timings=timings,
            total_llm_stats=total_llm_stats,
        )

    # ------------------------------------------------------------------
    # Internal: reinterpret — skip SQL gen+exec, answer over cached rows
    # ------------------------------------------------------------------

    def _reinterpret_prior(
        self,
        question: str,
        prior_turn: Turn,
        rid: str,
        followup: FollowupResponse,
    ) -> PipelineOutput:
        """Answer ``question`` over the prior turn's cached rows.

        Skips SQL generation / validation / execution — these stages report
        zero timings + zero stats. The answer LLM call is the only cost.
        Status is ``"success"`` when the answer stage has no error.
        """
        tracer = get_tracer()
        pipeline_start = time.perf_counter()
        model = self._llm.model_name
        prior_rows: list[dict[str, Any]] = list(prior_turn.rows)

        with tracer.start_as_current_span(
            "pipeline.run.reinterpret",
            attributes={"request_id": rid},
        ):
            with Timer("answer_generation"):
                ans = self._llm.generate_answer(
                    question,
                    prior_turn.sql,
                    prior_rows,
                    request_id=rid,
                )
            if stage_duration_ms is not None:
                stage_duration_ms.record(ans.timing_ms, {"stage": "answer_generation"})

        total_ms = (time.perf_counter() - pipeline_start) * 1000.0

        # Skipped stages: zero timing, zero stats, no error. We record that
        # the SQL-gen stage was bypassed via intermediate_outputs so the
        # evaluation harness can see what happened.
        sql_gen = SQLGenerationOutput(
            sql=prior_turn.sql,
            timing_ms=0.0,
            llm_stats=_zero_llm_stats(model),
            intermediate_outputs=[
                {
                    "reinterpret": True,
                    "reuses_prior_rows": followup.reuses_prior_rows,
                }
            ],
            error=None,
        )
        sql_val = SQLValidationOutput(
            is_valid=True,
            validated_sql=prior_turn.sql,
            error=None,
            timing_ms=0.0,
        )
        sql_exec = SQLExecutionOutput(
            rows=prior_rows,
            row_count=len(prior_rows),
            timing_ms=0.0,
            error=None,
        )

        status = "error" if ans.error is not None else "success"
        timings: dict[str, float] = {
            "sql_generation_ms": 0.0,
            "sql_validation_ms": 0.0,
            "sql_execution_ms": 0.0,
            "answer_generation_ms": ans.timing_ms,
            "total_ms": total_ms,
        }
        total_llm_stats = _aggregate_llm_stats(
            _zero_llm_stats(model),
            ans.llm_stats,
        )

        if pipeline_requests_total is not None:
            pipeline_requests_total.add(1, {"status": status})

        log_event(
            _logger,
            "pipeline_complete",
            request_id=rid,
            stage="pipeline",
            duration_ms=total_ms,
            status=status,
            tokens=total_llm_stats["total_tokens"],
            reinterpret=True,
        )

        return PipelineOutput(
            status=status,
            question=question,
            request_id=rid,
            sql_generation=sql_gen,
            sql_validation=sql_val,
            sql_execution=sql_exec,
            answer_generation=ans,
            sql=prior_turn.sql,
            rows=prior_rows,
            answer=ans.answer,
            timings=timings,
            total_llm_stats=total_llm_stats,
        )

    # ------------------------------------------------------------------
    # Internal: conversation store append
    # ------------------------------------------------------------------

    def _record_turn(
        self,
        conversation_id: str,
        raw_question: str,
        followup: FollowupResponse | None,
        output: PipelineOutput,
    ) -> None:
        """Append a ``Turn`` capturing this request to the conversation store."""
        intent = followup.intent if followup is not None else "NEW_QUERY"
        rewritten = (
            followup.rewritten_question
            if followup is not None and followup.intent != "NEW_QUERY"
            else None
        )
        turn = Turn(
            question=raw_question,
            rewritten_question=rewritten,
            intent=intent,
            sql=output.sql,
            rows=tuple(output.rows),
            answer=output.answer,
        )
        self._conversation_store.append(conversation_id, turn)


def _derive_status(
    gen: SQLGenerationOutput,
    val: SQLValidationOutput,
    exec_: SQLExecutionOutput,
) -> str:
    """Map stage outputs to the terminal pipeline status (first match wins).

    ======================================================  ==================
    Condition                                               ``status``
    ======================================================  ==================
    ``gen.sql is None and gen.error is not None``           ``"error"``
    ``gen.sql is None``                                     ``"unanswerable"``
    ``val.is_valid is False``                               ``"invalid_sql"``
    ``exec_.error is not None``                             ``"error"``
    otherwise                                               ``"success"``
    ======================================================  ==================
    """
    if gen.sql is None and gen.error is not None:
        return "error"
    if gen.sql is None:
        return "unanswerable"
    if not val.is_valid:
        return "invalid_sql"
    if exec_.error is not None:
        return "error"
    return "success"


def _aggregate_llm_stats(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Sum token/call counts across stages; pick the first non-empty model."""
    return {
        "llm_calls": int(a.get("llm_calls", 0)) + int(b.get("llm_calls", 0)),
        "prompt_tokens": int(a.get("prompt_tokens", 0)) + int(b.get("prompt_tokens", 0)),
        "completion_tokens": int(a.get("completion_tokens", 0))
        + int(b.get("completion_tokens", 0)),
        "total_tokens": int(a.get("total_tokens", 0)) + int(b.get("total_tokens", 0)),
        "model": str(a.get("model") or b.get("model") or "unknown"),
    }


def _zero_llm_stats(model: str) -> dict[str, Any]:
    """Canonical zero-cost ``llm_stats`` dict for stages that didn't call the LLM."""
    return {
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "model": model,
    }


__all__ = [
    "AnalyticsPipeline",
    "AnswerGenerationOutput",
    "SQLExecutionOutput",
    "SQLGenerationOutput",
    "SQLValidationOutput",
    "SQLiteExecutor",
]
