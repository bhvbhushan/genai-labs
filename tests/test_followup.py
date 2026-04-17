"""Tests for src.followup — FollowupClassifier (mocked LLM _chat)."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.conversation import Turn  # noqa: E402
from src.followup import (  # noqa: E402
    FollowupClassifier,
    FollowupResponse,
    _render_followup_user,
)
from src.llm_client import ChatResult  # noqa: E402


def _make_llm_with_content(content: str) -> MagicMock:
    """Build a MagicMock OpenRouterLLMClient whose _chat returns ``content``."""
    llm = MagicMock()
    llm._chat = MagicMock(
        return_value=ChatResult(
            content=content,
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8,
            usage_missing=False,
        )
    )
    return llm


def _turn(
    q: str = "Q",
    *,
    sql: str | None = None,
    rows: tuple[dict[str, object], ...] = (),
    answer: str = "A",
) -> Turn:
    return Turn(question=q, sql=sql, rows=rows, answer=answer)


class EmptyHistoryTests(unittest.TestCase):
    def test_short_circuits_without_llm_call(self) -> None:
        llm = _make_llm_with_content("{}")
        classifier = FollowupClassifier(llm)
        result = classifier.classify_and_rewrite("hello", history=[])
        self.assertEqual(result.intent, "NEW_QUERY")
        self.assertEqual(result.rewritten_question, "hello")
        self.assertFalse(result.reuses_prior_rows)
        self.assertFalse(llm._chat.called)


class LLMInvocationTests(unittest.TestCase):
    def test_llm_called_with_json_mode_and_zero_temp(self) -> None:
        llm = _make_llm_with_content(
            '{"intent": "NEW_QUERY", "rewritten_question": "x", "reuses_prior_rows": false}'
        )
        classifier = FollowupClassifier(llm)
        history = [_turn("prev", sql="SELECT 1", rows=({"n": 1},), answer="1")]
        classifier.classify_and_rewrite("new q", history=history, request_id="rid-1")

        self.assertTrue(llm._chat.called)
        kwargs = llm._chat.call_args.kwargs
        self.assertTrue(kwargs["json_mode"])
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertEqual(kwargs["stage"], "followup_classification")
        self.assertEqual(kwargs["request_id"], "rid-1")
        # messages positional or kwarg:
        messages = llm._chat.call_args.args[0]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("new q", messages[1]["content"])

    def test_followup_new_sql_passthrough(self) -> None:
        llm = _make_llm_with_content(
            '{"intent": "FOLLOWUP_NEW_SQL",'
            ' "rewritten_question": "addiction by males",'
            ' "reuses_prior_rows": false}'
        )
        classifier = FollowupClassifier(llm)
        history = [_turn("addiction by gender", sql="SELECT g, AVG(a) FROM t", rows=({"g": "m"},))]
        result = classifier.classify_and_rewrite("what about males?", history=history)
        self.assertEqual(result.intent, "FOLLOWUP_NEW_SQL")
        self.assertEqual(result.rewritten_question, "addiction by males")
        self.assertFalse(result.reuses_prior_rows)

    def test_reinterpret_with_prior_rows_passes_through(self) -> None:
        llm = _make_llm_with_content(
            '{"intent": "FOLLOWUP_REINTERPRET",'
            ' "rewritten_question": "which row is highest",'
            ' "reuses_prior_rows": true}'
        )
        classifier = FollowupClassifier(llm)
        history = [_turn("prev", sql="SELECT v FROM t", rows=({"v": 1}, {"v": 2}))]
        result = classifier.classify_and_rewrite("highest one?", history=history)
        self.assertEqual(result.intent, "FOLLOWUP_REINTERPRET")
        self.assertTrue(result.reuses_prior_rows)

    def test_reinterpret_without_prior_rows_downgrades(self) -> None:
        llm = _make_llm_with_content(
            '{"intent": "FOLLOWUP_REINTERPRET",'
            ' "rewritten_question": "highest",'
            ' "reuses_prior_rows": true}'
        )
        classifier = FollowupClassifier(llm)
        history = [_turn("prev", sql="SELECT v FROM t", rows=())]
        result = classifier.classify_and_rewrite("highest one?", history=history)
        self.assertEqual(result.intent, "FOLLOWUP_NEW_SQL")
        self.assertFalse(result.reuses_prior_rows)

    def test_reinterpret_without_prior_sql_downgrades(self) -> None:
        llm = _make_llm_with_content(
            '{"intent": "FOLLOWUP_REINTERPRET",'
            ' "rewritten_question": "highest",'
            ' "reuses_prior_rows": true}'
        )
        classifier = FollowupClassifier(llm)
        history = [_turn("prev", sql=None, rows=())]
        result = classifier.classify_and_rewrite("highest one?", history=history)
        self.assertEqual(result.intent, "FOLLOWUP_NEW_SQL")


class ParseFailureTests(unittest.TestCase):
    def test_bad_json_falls_back_to_new_query(self) -> None:
        llm = _make_llm_with_content("not-json-at-all")
        classifier = FollowupClassifier(llm)
        history = [_turn("prev", sql="SELECT 1", rows=({"n": 1},))]
        result = classifier.classify_and_rewrite("next q", history=history)
        self.assertEqual(result.intent, "NEW_QUERY")
        self.assertEqual(result.rewritten_question, "next q")
        self.assertFalse(result.reuses_prior_rows)

    def test_missing_required_field_falls_back(self) -> None:
        # Missing rewritten_question.
        llm = _make_llm_with_content('{"intent": "FOLLOWUP_NEW_SQL"}')
        classifier = FollowupClassifier(llm)
        history = [_turn("prev", sql="SELECT 1", rows=({"n": 1},))]
        result = classifier.classify_and_rewrite("next q", history=history)
        self.assertEqual(result.intent, "NEW_QUERY")
        self.assertEqual(result.rewritten_question, "next q")

    def test_unknown_intent_falls_back(self) -> None:
        llm = _make_llm_with_content(
            '{"intent": "SOMETHING_ELSE", "rewritten_question": "x", "reuses_prior_rows": false}'
        )
        classifier = FollowupClassifier(llm)
        history = [_turn("prev", sql="SELECT 1", rows=({"n": 1},))]
        result = classifier.classify_and_rewrite("next q", history=history)
        self.assertEqual(result.intent, "NEW_QUERY")

    def test_llm_exception_falls_back(self) -> None:
        llm = MagicMock()
        llm._chat.side_effect = RuntimeError("boom")
        classifier = FollowupClassifier(llm)
        history = [_turn("prev", sql="SELECT 1", rows=({"n": 1},))]
        result = classifier.classify_and_rewrite("next q", history=history)
        self.assertEqual(result.intent, "NEW_QUERY")
        self.assertEqual(result.rewritten_question, "next q")


class RenderTests(unittest.TestCase):
    def test_render_is_byte_stable(self) -> None:
        # Pin created_at so the pydantic Turn is bit-identical across calls.
        t = Turn(
            question="q1",
            sql="SELECT 1",
            rows=({"a": 1},),
            answer="one",
            created_at=1.0,
        )
        a = _render_followup_user("new q", [t])
        b = _render_followup_user("new q", [t])
        self.assertEqual(a, b)

    def test_render_includes_history_and_new_question(self) -> None:
        t = Turn(question="prev q", sql="SELECT 1", rows=({"n": 1},), answer="ans")
        rendered = _render_followup_user("the next question", [t])
        self.assertIn("prev q", rendered)
        self.assertIn("SELECT 1", rendered)
        self.assertIn("the next question", rendered)

    def test_render_skips_sql_line_when_sql_none(self) -> None:
        t = Turn(question="prev q", sql=None, rows=(), answer="ans")
        rendered = _render_followup_user("new", [t])
        self.assertNotIn("SQL:", rendered)


class FollowupResponseModelTests(unittest.TestCase):
    def test_extra_fields_ignored(self) -> None:
        resp = FollowupResponse.model_validate(
            {
                "intent": "NEW_QUERY",
                "rewritten_question": "x",
                "reuses_prior_rows": False,
                "extra": "ignored",
            }
        )
        self.assertEqual(resp.intent, "NEW_QUERY")


if __name__ == "__main__":
    unittest.main()
