"""Tests for the deterministic 1x1 scalar short-circuit in generate_answer."""

import sys
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src import llm_client as llm_client_mod  # noqa: E402
from src.llm_client import OpenRouterLLMClient  # noqa: E402
from src.schema import SchemaCatalog  # noqa: E402


def _make_client() -> OpenRouterLLMClient:
    """Build a client whose OpenRouter SDK handle is a MagicMock.

    The ``_client.chat.send`` method is pre-replaced with a ``MagicMock``
    so tests can assert on ``.called`` / ``.call_count`` without pyright
    complaining about the real (typed) SDK method shape.
    """
    schema = SchemaCatalog(table="t", columns=())
    with patch("src.llm_client.OpenRouter") as mock_or:
        mock_or.return_value = MagicMock()
        client = OpenRouterLLMClient(
            api_key="sk-test",
            model="openai/test",
            schema=schema,
            retries=0,
            retry_base_s=0.01,
        )
    client._client.chat.send = MagicMock()  # type: ignore[method-assign]
    return client


def _send(client: OpenRouterLLMClient) -> MagicMock:
    """Return the ``MagicMock`` bound to ``client._client.chat.send``."""
    return cast("MagicMock", client._client.chat.send)


class ScalarShortCircuitTests(unittest.TestCase):
    def test_int_scalar(self) -> None:
        client = _make_client()
        out = client.generate_answer("how many?", "SELECT COUNT(*) FROM t", [{"count": 4287}])
        self.assertEqual(out.answer, "The answer is 4287.")
        self.assertEqual(out.llm_stats["llm_calls"], 0)
        self.assertEqual(out.llm_stats["total_tokens"], 0)
        self.assertFalse(_send(client).called)

    def test_float_integer_valued(self) -> None:
        client = _make_client()
        out = client.generate_answer("avg?", "SELECT AVG(x) FROM t", [{"avg_score": 5.0}])
        self.assertEqual(out.answer, "The answer is 5.")
        self.assertFalse(_send(client).called)

    def test_float_with_decimals(self) -> None:
        client = _make_client()
        out = client.generate_answer("avg?", "SELECT AVG(x) FROM t", [{"avg_score": 3.75}])
        self.assertIn("3.75", out.answer)
        self.assertFalse(_send(client).called)

    def test_string_scalar(self) -> None:
        client = _make_client()
        out = client.generate_answer(
            "top gender?",
            "SELECT gender FROM t LIMIT 1",
            [{"top_gender": "Male"}],
        )
        self.assertIn("Male", out.answer)
        self.assertFalse(_send(client).called)

    def test_none_scalar(self) -> None:
        client = _make_client()
        out = client.generate_answer("x?", "SELECT NULL FROM t LIMIT 1", [{"x": None}])
        self.assertEqual(out.answer, "The answer is None.")
        self.assertFalse(_send(client).called)

    def test_bool_scalar(self) -> None:
        client = _make_client()
        out = client.generate_answer(
            "any matches?",
            "SELECT EXISTS(...) FROM t",
            [{"has_match": True}],
        )
        self.assertEqual(out.answer, "The answer is True.")
        self.assertFalse(_send(client).called)


class NonShortCircuitTests(unittest.TestCase):
    def _ok_response(self) -> MagicMock:
        """Build a fake SDK response the shared LLM path can consume."""
        message = MagicMock()
        message.content = "ok"
        choice = MagicMock()
        choice.message = message
        usage = MagicMock(spec=["prompt_tokens", "completion_tokens", "total_tokens"])
        usage.prompt_tokens = 1
        usage.completion_tokens = 1
        usage.total_tokens = 2
        res = MagicMock(spec=["choices", "usage"])
        res.choices = [choice]
        res.usage = usage
        return res

    def test_multi_cell_single_row_triggers_llm(self) -> None:
        client = _make_client()
        client._client.chat.send = MagicMock(  # type: ignore[method-assign]
            return_value=self._ok_response(),
        )
        out = client.generate_answer("q", "SELECT a, b FROM t", [{"a": 1, "b": 2}])
        self.assertTrue(_send(client).called)
        self.assertEqual(out.answer, "ok")
        self.assertEqual(out.llm_stats["llm_calls"], 1)

    def test_multi_row_triggers_llm(self) -> None:
        client = _make_client()
        client._client.chat.send = MagicMock(  # type: ignore[method-assign]
            return_value=self._ok_response(),
        )
        rows: list[dict[str, Any]] = [{"x": 1}, {"x": 2}]
        out = client.generate_answer("q", "SELECT x FROM t", rows)
        self.assertTrue(_send(client).called)
        self.assertEqual(out.answer, "ok")


class ShortCircuitMetricTests(unittest.TestCase):
    def test_short_circuit_counter_incremented_once(self) -> None:
        client = _make_client()
        counter = MagicMock()
        # Patch the module-level instrument reference that generate_answer
        # uses to increment; the real instrument may be None pre-configure.
        with patch.object(llm_client_mod, "llm_short_circuit_total", counter):
            client.generate_answer(
                "q",
                "SELECT COUNT(*) FROM t",
                [{"n": 7}],
            )
        counter.add.assert_called_once()
        # First positional arg is the increment value.
        self.assertEqual(counter.add.call_args.args[0], 1)

    def test_short_circuit_noop_when_counter_none(self) -> None:
        # Guard path: instrument is None pre-configure; must not crash.
        client = _make_client()
        with patch.object(llm_client_mod, "llm_short_circuit_total", None):
            out = client.generate_answer(
                "q",
                "SELECT COUNT(*) FROM t",
                [{"n": 7}],
            )
        self.assertEqual(out.answer, "The answer is 7.")


if __name__ == "__main__":
    unittest.main()
