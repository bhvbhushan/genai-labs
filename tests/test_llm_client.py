"""Tests for src.llm_client — OpenRouterLLMClient (transport + JSON + retry)."""

import os
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.config import Settings, get_settings  # noqa: E402
from src.llm_client import (  # noqa: E402
    OpenRouterLLMClient,
    _format_scalar,
    _is_auth_error,
    _rows_to_csv,
    build_default_llm_client,
)
from src.schema import SchemaCatalog  # noqa: E402


def _make_schema() -> SchemaCatalog:
    """Return a minimal SchemaCatalog usable in client construction."""
    return SchemaCatalog(table="t", columns=())


def _make_client(**overrides: Any) -> OpenRouterLLMClient:
    """Build a client with a MagicMock OpenRouter instance attached.

    All network I/O is intercepted by the mocked ``_client`` attribute. The
    caller can override constructor kwargs via ``overrides``.
    """
    schema = overrides.pop("schema", _make_schema())
    kwargs: dict[str, Any] = {
        "api_key": "sk-test",
        "model": "openai/test-model",
        "schema": schema,
        "retries": 0,
        "retry_base_s": 0.01,
    }
    kwargs.update(overrides)
    with patch("src.llm_client.OpenRouter") as mock_or:
        mock_or.return_value = MagicMock()
        return OpenRouterLLMClient(**kwargs)


def _fake_response(
    content: str,
    *,
    prompt_tokens: int | None = 10,
    completion_tokens: int | None = 5,
    total_tokens: int | None = 15,
    no_usage: bool = False,
) -> MagicMock:
    """Build a fake SDK response with .choices[0].message.content + .usage."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    res = MagicMock(spec=["choices", "usage"])
    res.choices = [choice]
    if no_usage:
        res.usage = None
    else:
        usage = MagicMock(spec=["prompt_tokens", "completion_tokens", "total_tokens"])
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        usage.total_tokens = total_tokens
        res.usage = usage
    return res


class ConstructorTests(unittest.TestCase):
    def test_system_prompt_is_byte_stable(self) -> None:
        schema = _make_schema()
        a = _make_client(schema=schema)
        b = _make_client(schema=schema)
        self.assertEqual(a._sql_system_message, b._sql_system_message)
        self.assertEqual(a._sql_system_message["role"], "system")
        # A second instance built with the same schema must produce the
        # identical system message bytes so the LLM provider can cache.
        self.assertIs(
            a._sql_system_message["content"].__class__,
            str,
        )

    def test_empty_api_key_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _make_client(api_key="")

    def test_model_name_property(self) -> None:
        client = _make_client(model="openai/gpt-5-mini")
        self.assertEqual(client.model_name, "openai/gpt-5-mini")


class BuildDefaultTests(unittest.TestCase):
    def setUp(self) -> None:
        get_settings.cache_clear()

    def tearDown(self) -> None:
        get_settings.cache_clear()

    def test_build_default_reads_settings(self) -> None:
        env = {
            "OPENROUTER_API_KEY": "sk-env",
            "OPENROUTER_MODEL": "openai/env-model",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]
            with (
                patch("src.llm_client.get_settings", return_value=settings),
                patch("src.llm_client.OpenRouter") as mock_or,
            ):
                mock_or.return_value = MagicMock()
                client = build_default_llm_client(_make_schema())
        self.assertEqual(client.model_name, "openai/env-model")
        # api_key must be propagated to the SDK constructor.
        mock_or.assert_called_once()
        kwargs = mock_or.call_args.kwargs
        self.assertEqual(kwargs.get("api_key"), "sk-env")


class GenerateSQLTests(unittest.TestCase):
    def test_happy_path_json(self) -> None:
        client = _make_client()
        client._client.chat.send = MagicMock(  # type: ignore[method-assign]
            return_value=_fake_response(
                '{"can_answer": true, "sql": "SELECT 1", "reason": null}',
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            )
        )
        out = client.generate_sql("how many rows?")
        self.assertEqual(out.sql, "SELECT 1")
        self.assertIsNone(out.error)
        self.assertEqual(out.llm_stats["prompt_tokens"], 10)
        self.assertEqual(out.llm_stats["completion_tokens"], 5)
        self.assertEqual(out.llm_stats["total_tokens"], 15)
        self.assertEqual(out.llm_stats["llm_calls"], 1)
        self.assertEqual(out.llm_stats["model"], "openai/test-model")

    def test_can_answer_false(self) -> None:
        client = _make_client()
        client._client.chat.send = MagicMock(  # type: ignore[method-assign]
            return_value=_fake_response(
                '{"can_answer": false, "sql": null, "reason": "zodiac column missing"}',
            )
        )
        out = client.generate_sql("what is my zodiac sign?")
        self.assertIsNone(out.sql)
        self.assertIsNone(out.error)
        self.assertEqual(len(out.intermediate_outputs), 1)
        self.assertEqual(
            out.intermediate_outputs[0].get("reason"),
            "zodiac column missing",
        )
        self.assertFalse(out.intermediate_outputs[0].get("can_answer"))

    def test_plain_text_fallback_extracts_select(self) -> None:
        client = _make_client()
        raw = "Sure, here's the query: SELECT * FROM t WHERE gender = 'Male'"
        client._client.chat.send = MagicMock(  # type: ignore[method-assign]
            return_value=_fake_response(raw)
        )
        out = client.generate_sql("males?")
        self.assertIsNotNone(out.sql)
        assert out.sql is not None
        self.assertTrue(out.sql.upper().startswith("SELECT"))
        self.assertIn("gender = 'Male'", out.sql)

    def test_validator_rejection_non_select_returns_none(self) -> None:
        # JSON-valid content with DROP — pydantic validator rejects. The
        # fallback path searches for 'select ' in the raw content; since
        # 'DROP' has no 'select', we end up with can_answer=False.
        client = _make_client()
        client._client.chat.send = MagicMock(  # type: ignore[method-assign]
            return_value=_fake_response(
                '{"can_answer": true, "sql": "DROP TABLE t", "reason": null}',
            )
        )
        out = client.generate_sql("drop it all")
        self.assertIsNone(out.sql)
        self.assertIsNone(out.error)

    def test_usage_missing_records_zeros(self) -> None:
        client = _make_client()
        client._client.chat.send = MagicMock(  # type: ignore[method-assign]
            return_value=_fake_response(
                '{"can_answer": true, "sql": "SELECT 1", "reason": null}',
                no_usage=True,
            )
        )
        out = client.generate_sql("x")
        # With usage missing, stats should reflect zeros but the call still
        # succeeded and the sql came through.
        self.assertEqual(out.sql, "SELECT 1")
        self.assertEqual(out.llm_stats["prompt_tokens"], 0)
        self.assertEqual(out.llm_stats["completion_tokens"], 0)
        self.assertEqual(out.llm_stats["total_tokens"], 0)
        self.assertEqual(out.llm_stats["llm_calls"], 1)

    def test_retry_succeeds_after_transient_error(self) -> None:
        client = _make_client(retries=2, retry_base_s=0.001)
        ok_response = _fake_response(
            '{"can_answer": true, "sql": "SELECT 1", "reason": null}',
        )
        client._client.chat.send = MagicMock(  # type: ignore[method-assign]
            side_effect=[ConnectionError("timeout"), ok_response],
        )
        # Pin the jitter source so the test is deterministic.
        with patch("src.llm_client.random.uniform", return_value=1.0):
            out = client.generate_sql("retry me")
        self.assertEqual(out.sql, "SELECT 1")
        self.assertIsNone(out.error)
        # send was called twice: one failure + one success.
        self.assertEqual(client._client.chat.send.call_count, 2)

    def test_auth_error_is_not_retried(self) -> None:
        client = _make_client(retries=3, retry_base_s=0.001)
        client._client.chat.send = MagicMock(  # type: ignore[method-assign]
            side_effect=Exception("Unauthorized: invalid API key"),
        )
        out = client.generate_sql("auth fail")
        self.assertIsNone(out.sql)
        self.assertIsNotNone(out.error)
        assert out.error is not None
        self.assertIn("Unauthorized", out.error)
        # No retry was attempted.
        self.assertEqual(client._client.chat.send.call_count, 1)

    def test_chat_failure_returns_error_output(self) -> None:
        client = _make_client(retries=0)
        client._client.chat.send = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("network down"),
        )
        out = client.generate_sql("x")
        self.assertIsNone(out.sql)
        self.assertIsNotNone(out.error)
        assert out.error is not None
        self.assertIn("network down", out.error)
        self.assertEqual(out.llm_stats["llm_calls"], 0)

    def test_empty_content_raises_into_error_branch(self) -> None:
        # Empty content → _finalize_chat raises RuntimeError, which bubbles
        # up as the generate_sql error field.
        client = _make_client()
        client._client.chat.send = MagicMock(  # type: ignore[method-assign]
            return_value=_fake_response(""),
        )
        out = client.generate_sql("empty")
        self.assertIsNone(out.sql)
        self.assertIsNotNone(out.error)


class GenerateAnswerTests(unittest.TestCase):
    def test_sql_none_returns_cannot_answer(self) -> None:
        client = _make_client()
        client._client.chat.send = MagicMock()  # type: ignore[method-assign]
        out = client.generate_answer("q", None, [{"x": 1}])
        self.assertIn("cannot answer", out.answer.lower())
        self.assertEqual(out.llm_stats["prompt_tokens"], 0)
        self.assertEqual(out.llm_stats["llm_calls"], 0)
        # No LLM call made.
        self.assertFalse(client._client.chat.send.called)

    def test_empty_rows_returns_no_rows_message(self) -> None:
        client = _make_client()
        client._client.chat.send = MagicMock()  # type: ignore[method-assign]
        out = client.generate_answer("q", "SELECT * FROM t", [])
        self.assertIn("no matching rows", out.answer.lower())
        self.assertEqual(out.llm_stats["total_tokens"], 0)
        self.assertFalse(client._client.chat.send.called)

    def test_multirow_triggers_llm_call(self) -> None:
        client = _make_client()
        client._client.chat.send = MagicMock(  # type: ignore[method-assign]
            return_value=_fake_response(
                "About 2000 males and 1000 females.",
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70,
            )
        )
        rows = [{"gender": "Male", "n": 2000}, {"gender": "Female", "n": 1000}]
        out = client.generate_answer("breakdown?", "SELECT gender, COUNT(*) FROM t", rows)
        self.assertEqual(out.answer, "About 2000 males and 1000 females.")
        self.assertIsNone(out.error)
        self.assertEqual(out.llm_stats["total_tokens"], 70)
        self.assertEqual(out.llm_stats["llm_calls"], 1)
        self.assertTrue(client._client.chat.send.called)
        # The user message must be CSV-formatted.
        kwargs = client._client.chat.send.call_args.kwargs
        user_msg = kwargs["messages"][-1]["content"]
        self.assertIn("gender,n", user_msg)
        self.assertIn("Male,2000", user_msg)

    def test_llm_exception_populates_error_field(self) -> None:
        client = _make_client(retries=0)
        client._client.chat.send = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("boom"),
        )
        rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        out = client.generate_answer("?", "SELECT a, b FROM t", rows)
        self.assertIn("Error generating answer", out.answer)
        self.assertIsNotNone(out.error)
        assert out.error is not None
        self.assertIn("boom", out.error)

    def test_rows_capped_to_max_rows_to_llm(self) -> None:
        client = _make_client(max_rows_to_llm=2)
        client._client.chat.send = MagicMock(  # type: ignore[method-assign]
            return_value=_fake_response("ok"),
        )
        rows = [{"x": i} for i in range(10)]
        client.generate_answer("q", "SELECT x FROM t", rows)
        kwargs = client._client.chat.send.call_args.kwargs
        user_msg = kwargs["messages"][-1]["content"]
        # Header + 2 rows; row 2 (x=2) must NOT appear.
        self.assertIn("x", user_msg)
        self.assertIn("0", user_msg)
        self.assertIn("1", user_msg)
        self.assertNotIn("\n2\n", user_msg)


class HelperFunctionTests(unittest.TestCase):
    def test_is_auth_error_markers(self) -> None:
        self.assertTrue(_is_auth_error(Exception("HTTP 401 Unauthorized")))
        self.assertTrue(_is_auth_error(Exception("invalid API key")))
        self.assertFalse(_is_auth_error(Exception("Connection reset")))
        self.assertFalse(_is_auth_error(Exception("500 internal server")))

    def test_rows_to_csv_header_and_data(self) -> None:
        rows: list[dict[str, Any]] = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        out = _rows_to_csv(rows)
        lines = out.splitlines()
        self.assertEqual(lines[0], "a,b")
        self.assertEqual(lines[1], "1,2")
        self.assertEqual(lines[2], "3,4")

    def test_rows_to_csv_empty(self) -> None:
        self.assertEqual(_rows_to_csv([]), "")

    def test_format_scalar_variants(self) -> None:
        self.assertEqual(_format_scalar(None), "None")
        self.assertEqual(_format_scalar(True), "True")
        self.assertEqual(_format_scalar(42), "42")
        self.assertEqual(_format_scalar(5.0), "5")
        self.assertEqual(_format_scalar(3.75), "3.75")
        self.assertEqual(_format_scalar("Male"), "Male")


if __name__ == "__main__":
    unittest.main()
