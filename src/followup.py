"""Follow-up classifier + rewriter — single LLM round-trip.

Given the user's question and the last few turns of conversation history, the
classifier decides between three intents and (for follow-ups) rewrites the
question so it is self-contained:

- ``NEW_QUERY``: question unrelated to prior turns, OR history is empty
  (short-circuit with zero LLM calls).
- ``FOLLOWUP_NEW_SQL``: the user is following up but wants different rows —
  the rewriter produces a self-contained question; the pipeline runs fully.
- ``FOLLOWUP_REINTERPRET``: the user is asking about the *already-computed*
  rows ("explain the highest") — the pipeline skips SQL gen+exec and calls
  ``generate_answer`` on the prior turn's cached rows. Automatically
  downgraded to ``FOLLOWUP_NEW_SQL`` if prior rows / SQL are absent.

One LLM call does both classification and rewriting (single JSON response),
so we pay one round-trip for both signals.

Falls back to ``NEW_QUERY`` with the original question on any parse failure;
increments the same ``llm_json_fallback_total`` counter used by ``llm_client``
so metrics show JSON drift across stages uniformly.
"""

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, ValidationError

from src import observability as _obs
from src.conversation import FollowupIntent, Turn, summarize_rows
from src.observability import (
    get_logger,
    log_event,
)

if TYPE_CHECKING:
    from src.llm_client import OpenRouterLLMClient

_logger = get_logger(__name__)

# Stable, byte-invariant system prompt. Kept as a module constant so every
# request hits the provider's automatic prompt cache.
_FOLLOWUP_SYSTEM_PROMPT = """You are a conversation rewriter for a SQL analytics assistant.

Given a user's new question and the most recent turns of their conversation,
classify the question into one of three intents and, for follow-ups, rewrite
the question so it stands on its own without the history.

Intents:

- NEW_QUERY: The new question is unrelated to the prior turns. No rewrite
  needed; set rewritten_question to the original question verbatim.

- FOLLOWUP_NEW_SQL: The new question is a follow-up that needs different SQL
  (e.g. "what about males?", "sort by age instead"). Produce a self-contained
  rewrite that captures the user's intent without needing the prior turns.
  Set reuses_prior_rows to false.

- FOLLOWUP_REINTERPRET: The new question asks about the ALREADY-COMPUTED rows
  from the most recent turn (e.g. "explain the highest value", "which row has
  the most?"). Do NOT require new SQL; the answer should come from the prior
  turn's cached rows. Set reuses_prior_rows to true and make
  rewritten_question a self-contained phrasing.

Return ONLY a JSON object with this exact shape:

{"intent": "NEW_QUERY" | "FOLLOWUP_NEW_SQL" | "FOLLOWUP_REINTERPRET",
 "rewritten_question": string,
 "reuses_prior_rows": boolean}

No prose, no markdown, no code fences.
"""


class FollowupResponse(BaseModel):
    """Validated shape of the classifier JSON response."""

    model_config = ConfigDict(extra="ignore")

    intent: FollowupIntent
    rewritten_question: str
    reuses_prior_rows: bool = False


def _render_followup_user(question: str, history: list[Turn]) -> str:
    """Render the user message for the classifier — byte-stable for a given input.

    Compact by design: the caller trims history to the last 4 turns.
    """
    lines: list[str] = []
    for i, turn in enumerate(history, 1):
        lines.append(f"Turn {i}: Q: {turn.question}")
        if turn.sql:
            lines.append(f"        SQL: {turn.sql}")
        lines.append(f"        Rows: {summarize_rows(turn.rows)}")
        lines.append(f"        Answer: {turn.answer}")
    lines.append("")
    lines.append(f"New question: {question}")
    return "\n".join(lines)


class FollowupClassifier:
    """LLM-backed classify + rewrite. Single round-trip per call.

    Reaches into ``OpenRouterLLMClient._chat`` for transport — the classifier
    legitimately needs the low-level chat call (JSON mode, deterministic
    temperature) and sits in the same package.
    """

    def __init__(self, llm_client: "OpenRouterLLMClient") -> None:
        self._llm = llm_client

    def classify_and_rewrite(
        self,
        question: str,
        history: list[Turn],
        *,
        request_id: str | None = None,
    ) -> FollowupResponse:
        """Classify the new question against recent history. Never raises.

        Empty history short-circuits to ``NEW_QUERY`` with no LLM call. Any
        parse / validation failure also falls back to ``NEW_QUERY`` so the
        pipeline continues to make forward progress.
        """
        if not history:
            return FollowupResponse(
                intent="NEW_QUERY",
                rewritten_question=question,
                reuses_prior_rows=False,
            )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _FOLLOWUP_SYSTEM_PROMPT},
            {"role": "user", "content": _render_followup_user(question, history)},
        ]

        try:
            result = self._llm._chat(
                messages,
                temperature=0.0,
                max_tokens=400,
                json_mode=True,
                stage="followup_classification",
                request_id=request_id,
            )
        except Exception as exc:
            log_event(
                _logger,
                "followup_llm_error",
                request_id=request_id,
                stage="followup_classification",
                error=str(exc),
            )
            return FollowupResponse(
                intent="NEW_QUERY",
                rewritten_question=question,
                reuses_prior_rows=False,
            )

        parsed = self._parse_response(result.content, question=question, request_id=request_id)

        # Safety downgrade: REINTERPRET requires a prior turn with both SQL
        # and rows. If either is missing, fall back to NEW_SQL so the
        # pipeline produces a fresh answer rather than echoing an empty set.
        if parsed.intent == "FOLLOWUP_REINTERPRET":
            last = history[-1]
            if last.sql is None or not last.rows:
                log_event(
                    _logger,
                    "followup_reinterpret_downgraded",
                    request_id=request_id,
                    stage="followup_classification",
                    reason="prior_rows_missing",
                )
                return parsed.model_copy(
                    update={"intent": "FOLLOWUP_NEW_SQL", "reuses_prior_rows": False},
                )

        return parsed

    def _parse_response(
        self,
        content: str,
        *,
        question: str,
        request_id: str | None,
    ) -> FollowupResponse:
        """Parse the JSON body or fall back to ``NEW_QUERY`` on any error."""
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return self._fallback(question=question, request_id=request_id, reason="json_decode")

        try:
            return FollowupResponse.model_validate(payload)
        except ValidationError:
            return self._fallback(question=question, request_id=request_id, reason="validation")

    def _fallback(
        self,
        *,
        question: str,
        request_id: str | None,
        reason: str,
    ) -> FollowupResponse:
        """Bump the shared JSON-fallback counter, warn, and return NEW_QUERY."""
        if _obs.llm_json_fallback_total is not None:
            _obs.llm_json_fallback_total.add(1, {"stage": "followup_classification"})
        _logger.warning(
            "llm_json_fallback",
            extra={
                "stage": "followup_classification",
                "request_id": request_id,
                "reason": reason,
            },
        )
        return FollowupResponse(
            intent="NEW_QUERY",
            rewritten_question=question,
            reuses_prior_rows=False,
        )


__all__ = [
    "_FOLLOWUP_SYSTEM_PROMPT",
    "FollowupClassifier",
    "FollowupResponse",
    "_render_followup_user",
]
