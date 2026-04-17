"""Production OpenRouter client: token counting, JSON mode, retry, short-circuit.

This module provides :class:`OpenRouterLLMClient`, the single transport layer
between the pipeline and the OpenRouter SDK. Responsibilities:

- Build the SQL-generation system prompt ONCE at construction time so every
  request hits OpenRouter's automatic prompt cache on a byte-stable system
  message.
- Request structured JSON output on the SQL-generation stage and validate it
  with a pydantic model; on parse/validation failure, fall back to plain-text
  SELECT extraction and increment ``llm_json_fallback_total``.
- Retry transient errors (network / 5xx / 429) with jittered backoff. Auth
  failures (``Unauthorized`` / ``API key``) are not retried.
- Extract token usage from ``res.usage``; missing usage is recorded as zeros
  and bumps ``llm_usage_missing_total`` so metrics stay honest.
- Deterministically short-circuit the answer-generation stage for 1x1 scalar
  result sets — no LLM round-trip required.
- Format answer-stage rows as CSV (header + data) to reduce token count vs
  the baseline JSON serialisation.
"""

import csv
import io
import json
import random
import time
from dataclasses import dataclass
from typing import Any

from openrouter import OpenRouter
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from src.config import get_settings
from src.observability import (
    get_logger,
    llm_calls_total,
    llm_json_fallback_total,
    llm_short_circuit_total,
    llm_tokens_total,
    llm_usage_missing_total,
    log_event,
)
from src.prompts import (
    render_answer_system,
    render_answer_user,
    render_sql_system,
    render_sql_user,
)
from src.schema import SchemaCatalog
from src.types import AnswerGenerationOutput, SQLGenerationOutput

_logger = get_logger(__name__)

_UNANSWERABLE_MESSAGE = "I cannot answer this question with the available schema."
_NO_ROWS_MESSAGE = "The query returned no matching rows."
_UNPARSEABLE_REASON = "LLM output could not be parsed as SQL"

_AUTH_MARKERS: tuple[str, ...] = ("unauthorized", "api key", "invalid_api_key")


@dataclass(frozen=True)
class ChatResult:
    """Normalized payload returned from :meth:`OpenRouterLLMClient._chat`."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    usage_missing: bool


class SQLGenerationResponse(BaseModel):
    """Validated shape of the JSON-mode SQL-generation response.

    Intentionally does not enforce that ``sql`` starts with SELECT: the
    trust boundary lives in :class:`src.validator.SQLValidator` (sqlglot AST).
    Forwarding DDL/DML here lets the validator reject with a specific error
    so the pipeline emits ``status="invalid_sql"`` instead of swallowing the
    SQL into ``unanswerable``.
    """

    model_config = ConfigDict(extra="ignore")

    can_answer: bool
    sql: str | None = None
    reason: str | None = None

    @field_validator("sql")
    @classmethod
    def strip_sql(cls, v: str | None) -> str | None:
        """Strip surrounding whitespace; leave statement-kind policy to the AST validator."""
        if v is None:
            return None
        return v.strip()


def _format_scalar(value: Any) -> str:
    """Format a single-cell scalar for the deterministic short-circuit answer."""
    if value is None:
        return "None"
    # bool is a subclass of int, so check it first.
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return str(value)
    if isinstance(value, str):
        return value
    return str(value)


def _is_auth_error(exc: BaseException) -> bool:
    """Return True if ``exc`` looks like an OpenRouter auth failure."""
    text = str(exc).lower()
    return any(marker in text for marker in _AUTH_MARKERS)


def _rows_to_csv(rows: list[dict[str, Any]]) -> str:
    """Render ``rows`` as CSV with a header row. Empty list → empty string."""
    if not rows:
        return ""
    fieldnames = list(rows[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue().rstrip("\n")


def _zero_stats(model: str) -> dict[str, Any]:
    """Return the canonical zero-token ``llm_stats`` dict."""
    return {
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "model": model,
    }


def _stats_from_result(result: ChatResult, model: str) -> dict[str, Any]:
    """Build the canonical ``llm_stats`` dict from a successful chat result."""
    return {
        "llm_calls": 1,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "model": model,
    }


class OpenRouterLLMClient:
    """Production OpenRouter client: token counting, JSON mode, retry, short-circuit."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        schema: SchemaCatalog,
        timeout_s: float = 30.0,
        retries: int = 1,
        retry_base_s: float = 0.3,
        max_rows_to_llm: int = 30,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be a non-empty string")
        self._api_key = api_key
        self._model = model
        self._schema = schema
        self._timeout_s = timeout_s
        self._retries = retries
        self._retry_base_s = retry_base_s
        self._max_rows_to_llm = max_rows_to_llm
        # Build the stable system prompt ONCE so it is byte-identical across
        # every request — this is what OpenRouter's automatic prompt cache
        # keys on.
        sql_system_text = render_sql_system(schema.to_prompt())
        self._sql_system_message: dict[str, str] = {
            "role": "system",
            "content": sql_system_text,
        }
        self._answer_system_message: dict[str, str] = {
            "role": "system",
            "content": render_answer_system(),
        }
        self._client = OpenRouter(
            api_key=api_key,
            timeout_ms=int(timeout_s * 1000),
        )

    @property
    def model_name(self) -> str:
        """Return the model name (for stats.model field)."""
        return self._model

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        stage: str,
        request_id: str | None,
    ) -> ChatResult:
        """Send one chat completion with retry + metrics + structured logs."""
        kwargs: dict[str, Any] = {
            "messages": messages,
            "model": self._model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            # Reasoning models (gpt-5-nano, o-series) silently consume the
            # entire token budget on reasoning and emit empty content when
            # effort is not pinned. "minimal" is the cheapest reasoning-mode
            # setting that still produces answer tokens; non-reasoning models
            # ignore the field.
            "reasoning": {"effort": "minimal"},
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        attempts = self._retries + 1
        last_exc: BaseException | None = None
        start = time.perf_counter()
        for attempt in range(attempts):
            try:
                res = self._client.chat.send(**kwargs)
                return self._finalize_chat(
                    res,
                    stage=stage,
                    request_id=request_id,
                    duration_ms=(time.perf_counter() - start) * 1000.0,
                )
            except Exception as exc:
                last_exc = exc
                if _is_auth_error(exc):
                    if llm_calls_total is not None:
                        llm_calls_total.add(1, {"stage": stage, "outcome": "error"})
                    raise
                # Transient error path.
                if attempt < attempts - 1:
                    if llm_calls_total is not None:
                        llm_calls_total.add(1, {"stage": stage, "outcome": "retry"})
                    # Jittered constant backoff per plan: base * U(0.7, 1.3).
                    delay = self._retry_base_s * random.uniform(0.7, 1.3)
                    time.sleep(delay)
                    continue
                # No retries left.
                if llm_calls_total is not None:
                    llm_calls_total.add(1, {"stage": stage, "outcome": "error"})
                raise

        # Defensive: the loop above always returns or raises, but mypy needs
        # a terminal statement. Re-raise the last exception.
        assert last_exc is not None
        raise last_exc

    def _finalize_chat(
        self,
        res: Any,
        *,
        stage: str,
        request_id: str | None,
        duration_ms: float,
    ) -> ChatResult:
        """Extract content + usage from a successful SDK response."""
        choices = getattr(res, "choices", None) or []
        content: str | None = None
        if choices:
            message = getattr(choices[0], "message", None)
            raw_content = getattr(message, "content", None)
            if isinstance(raw_content, str):
                content = raw_content
        if not content:
            raise RuntimeError("OpenRouter response has no content")

        usage = getattr(res, "usage", None)
        raw_prompt = _safe_int(getattr(usage, "prompt_tokens", None))
        raw_completion = _safe_int(getattr(usage, "completion_tokens", None))
        raw_total = _safe_int(getattr(usage, "total_tokens", None))
        prompt_tokens: int
        completion_tokens: int
        total_tokens: int
        if usage is None or raw_prompt is None or raw_completion is None or raw_total is None:
            usage_missing = True
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            if llm_usage_missing_total is not None:
                llm_usage_missing_total.add(1, {"stage": stage})
        else:
            usage_missing = False
            prompt_tokens = raw_prompt
            completion_tokens = raw_completion
            total_tokens = raw_total

        # Token + call metrics.
        if llm_tokens_total is not None:
            llm_tokens_total.add(
                prompt_tokens,
                {"stage": stage, "kind": "prompt"},
            )
            llm_tokens_total.add(
                completion_tokens,
                {"stage": stage, "kind": "completion"},
            )
        if llm_calls_total is not None:
            llm_calls_total.add(1, {"stage": stage, "outcome": "success"})

        log_event(
            _logger,
            "llm_call_complete",
            request_id=request_id,
            stage=stage,
            duration_ms=duration_ms,
            tokens=total_tokens,
            model=self._model,
        )

        return ChatResult(
            content=content.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            usage_missing=usage_missing,
        )

    # ------------------------------------------------------------------
    # Public: SQL generation
    # ------------------------------------------------------------------

    def generate_sql(
        self,
        question: str,
        *,
        request_id: str | None = None,
    ) -> SQLGenerationOutput:
        """Generate a SQLite SELECT for ``question``. Never raises."""
        start = time.perf_counter()
        messages: list[dict[str, str]] = [
            self._sql_system_message,
            {"role": "user", "content": render_sql_user(question)},
        ]
        try:
            result = self._chat(
                messages,
                temperature=0.0,
                # Reasoning models (e.g., gpt-5-nano) burn tokens on hidden
                # reasoning before emitting content; even with effort=minimal
                # we need headroom so the JSON body survives.
                max_tokens=800,
                json_mode=True,
                stage="sql_generation",
                request_id=request_id,
            )
        except Exception as exc:
            return SQLGenerationOutput(
                sql=None,
                timing_ms=(time.perf_counter() - start) * 1000.0,
                llm_stats=_zero_stats(self._model),
                intermediate_outputs=[],
                error=str(exc),
            )

        parsed = self._parse_sql_response(result.content, request_id=request_id)
        timing_ms = (time.perf_counter() - start) * 1000.0
        stats = _stats_from_result(result, self._model)

        if not parsed.can_answer or parsed.sql is None:
            return SQLGenerationOutput(
                sql=None,
                timing_ms=timing_ms,
                llm_stats=stats,
                intermediate_outputs=[
                    {"can_answer": False, "reason": parsed.reason},
                ],
                error=None,
            )

        return SQLGenerationOutput(
            sql=parsed.sql,
            timing_ms=timing_ms,
            llm_stats=stats,
            intermediate_outputs=[
                {"can_answer": True, "reason": parsed.reason},
            ],
            error=None,
        )

    def _parse_sql_response(
        self,
        content: str,
        *,
        request_id: str | None,
    ) -> SQLGenerationResponse:
        """Validate JSON-mode content or fall back to plain-text extraction."""
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return self._fallback_sql_extraction(content, request_id=request_id)

        try:
            return SQLGenerationResponse.model_validate(payload)
        except ValidationError:
            return self._fallback_sql_extraction(content, request_id=request_id)

    def _fallback_sql_extraction(
        self,
        content: str,
        *,
        request_id: str | None,
    ) -> SQLGenerationResponse:
        """Extract a SELECT by case-insensitive search when JSON parsing fails."""
        if llm_json_fallback_total is not None:
            llm_json_fallback_total.add(1, {"stage": "sql_generation"})
        _logger.warning(
            "llm_json_fallback",
            extra={
                "stage": "sql_generation",
                "request_id": request_id,
            },
        )
        lower = content.lower()
        idx = lower.find("select ")
        if idx < 0:
            return SQLGenerationResponse(
                can_answer=False,
                sql=None,
                reason=_UNPARSEABLE_REASON,
            )
        extracted = content[idx:].strip()
        return SQLGenerationResponse(
            can_answer=True,
            sql=extracted,
            reason=None,
        )

    # ------------------------------------------------------------------
    # Public: answer generation
    # ------------------------------------------------------------------

    def generate_answer(
        self,
        question: str,
        sql: str | None,
        rows: list[dict[str, Any]],
        *,
        request_id: str | None = None,
    ) -> AnswerGenerationOutput:
        """Produce a natural-language answer. Never raises."""
        # 1. Unanswerable upstream.
        if sql is None:
            return AnswerGenerationOutput(
                answer=_UNANSWERABLE_MESSAGE,
                timing_ms=0.0,
                llm_stats=_zero_stats(self._model),
                intermediate_outputs=[],
                error=None,
            )

        # 2. Empty rows — canonical message, no LLM call.
        if not rows:
            return AnswerGenerationOutput(
                answer=_NO_ROWS_MESSAGE,
                timing_ms=0.0,
                llm_stats=_zero_stats(self._model),
                intermediate_outputs=[],
                error=None,
            )

        # 3. Deterministic short-circuit for 1x1 scalar.
        if len(rows) == 1 and len(rows[0]) == 1:
            value = next(iter(rows[0].values()))
            answer = f"The answer is {_format_scalar(value)}."
            if llm_short_circuit_total is not None:
                llm_short_circuit_total.add(1, {"stage": "answer_generation"})
            log_event(
                _logger,
                "llm_short_circuit",
                request_id=request_id,
                stage="answer_generation",
            )
            return AnswerGenerationOutput(
                answer=answer,
                timing_ms=0.0,
                llm_stats=_zero_stats(self._model),
                intermediate_outputs=[{"short_circuit": True}],
                error=None,
            )

        # 4. Full LLM call.
        start = time.perf_counter()
        rows_for_llm = rows[: self._max_rows_to_llm]
        rows_csv = _rows_to_csv(rows_for_llm)
        messages: list[dict[str, str]] = [
            self._answer_system_message,
            {
                "role": "user",
                "content": render_answer_user(question, sql, rows_csv),
            },
        ]
        try:
            result = self._chat(
                messages,
                temperature=0.2,
                # Same headroom as SQL gen — reasoning-model overhead applies
                # here too for the answer-generation call.
                max_tokens=800,
                json_mode=False,
                stage="answer_generation",
                request_id=request_id,
            )
        except Exception as exc:
            return AnswerGenerationOutput(
                answer=f"Error generating answer: {exc}",
                timing_ms=(time.perf_counter() - start) * 1000.0,
                llm_stats=_zero_stats(self._model),
                intermediate_outputs=[],
                error=str(exc),
            )

        return AnswerGenerationOutput(
            answer=result.content,
            timing_ms=(time.perf_counter() - start) * 1000.0,
            llm_stats=_stats_from_result(result, self._model),
            intermediate_outputs=[],
            error=None,
        )


def _safe_int(value: Any) -> int | None:
    """Coerce an SDK-provided usage value to int, or return None."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_default_llm_client(schema: SchemaCatalog) -> OpenRouterLLMClient:
    """Convenience factory: reads ``Settings`` via ``get_settings()``."""
    settings = get_settings()
    return OpenRouterLLMClient(
        api_key=settings.openrouter_api_key,
        model=settings.model,
        schema=schema,
        timeout_s=settings.llm_timeout_s,
        retries=settings.llm_retries,
        retry_base_s=settings.llm_retry_base_s,
        max_rows_to_llm=settings.max_rows_to_llm,
    )
