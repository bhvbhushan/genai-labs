"""Tests for src.response_cache — LRU cache of PipelineOutput by question hash."""

import sys
import threading
import unittest
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.response_cache import ResponseCache, _normalize_question, _question_key  # noqa: E402
from src.types import (  # noqa: E402
    AnswerGenerationOutput,
    PipelineOutput,
    SQLExecutionOutput,
    SQLGenerationOutput,
    SQLValidationOutput,
)


def _make_output(
    question: str = "q",
    answer: str = "a",
    rows: list[dict[str, Any]] | None = None,
) -> PipelineOutput:
    rows = rows if rows is not None else [{"n": 1}]
    gen = SQLGenerationOutput(
        sql="SELECT 1",
        timing_ms=1.0,
        llm_stats={
            "llm_calls": 1,
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
            "model": "m",
        },
        intermediate_outputs=[{"previous": True}],
        error=None,
    )
    val = SQLValidationOutput(is_valid=True, validated_sql="SELECT 1 LIMIT 1000", timing_ms=0.1)
    exe = SQLExecutionOutput(rows=rows, row_count=len(rows), timing_ms=2.0, error=None)
    ans = AnswerGenerationOutput(
        answer=answer,
        timing_ms=4.0,
        llm_stats={
            "llm_calls": 1,
            "prompt_tokens": 10,
            "completion_tokens": 6,
            "total_tokens": 16,
            "model": "m",
        },
        intermediate_outputs=[],
        error=None,
    )
    return PipelineOutput(
        status="success",
        question=question,
        request_id="rid",
        sql_generation=gen,
        sql_validation=val,
        sql_execution=exe,
        answer_generation=ans,
        sql="SELECT 1 LIMIT 1000",
        rows=rows,
        answer=answer,
        timings={
            "sql_generation_ms": 1.0,
            "sql_validation_ms": 0.1,
            "sql_execution_ms": 2.0,
            "answer_generation_ms": 4.0,
            "total_ms": 7.1,
        },
        total_llm_stats={
            "llm_calls": 2,
            "prompt_tokens": 15,
            "completion_tokens": 9,
            "total_tokens": 24,
            "model": "m",
        },
    )


class ResponseCacheTests(unittest.TestCase):
    def test_empty_cache_returns_none(self) -> None:
        cache = ResponseCache()
        self.assertIsNone(cache.get("nothing cached"))
        self.assertEqual(cache.stats.misses, 1)
        self.assertEqual(cache.stats.hits, 0)

    def test_put_then_get_returns_copy(self) -> None:
        cache = ResponseCache()
        out = _make_output(answer="hello")
        cache.put("q1", out)
        got = cache.get("q1")
        self.assertIsNotNone(got)
        assert got is not None
        # Not the same object — deep copy.
        self.assertIsNot(got, out)
        self.assertIsNot(got.sql_generation, out.sql_generation)
        self.assertEqual(got.answer, "hello")
        self.assertEqual(cache.stats.hits, 1)

    def test_hit_zeroes_llm_stats_and_timings(self) -> None:
        cache = ResponseCache()
        cache.put("q1", _make_output())
        got = cache.get("q1")
        assert got is not None
        self.assertEqual(got.total_llm_stats["llm_calls"], 0)
        self.assertEqual(got.total_llm_stats["total_tokens"], 0)
        self.assertEqual(got.total_llm_stats["prompt_tokens"], 0)
        self.assertEqual(got.total_llm_stats["completion_tokens"], 0)
        # Model is preserved for observability.
        self.assertEqual(got.total_llm_stats["model"], "m")
        self.assertEqual(got.timings["total_ms"], 0.0)
        self.assertEqual(got.timings["sql_generation_ms"], 0.0)
        self.assertEqual(got.timings["answer_generation_ms"], 0.0)

    def test_hit_marker_prepended_in_intermediate_outputs(self) -> None:
        cache = ResponseCache()
        cache.put("q1", _make_output())
        got = cache.get("q1")
        assert got is not None
        self.assertEqual(got.sql_generation.intermediate_outputs[0], {"cache_hit": True})
        # The original entry ("previous": True) follows after the marker.
        self.assertEqual(got.sql_generation.intermediate_outputs[1], {"previous": True})

    def test_question_normalization(self) -> None:
        self.assertEqual(_normalize_question("  Hello   World? "), "hello world?")
        self.assertEqual(_question_key("  WHAT IS X?  "), _question_key("what is x?"))
        cache = ResponseCache()
        cache.put("What is X?", _make_output())
        got = cache.get("  what is x?  ")
        self.assertIsNotNone(got)

    def test_lru_eviction(self) -> None:
        cache = ResponseCache(max_entries=3)
        cache.put("q1", _make_output(answer="a1"))
        cache.put("q2", _make_output(answer="a2"))
        cache.put("q3", _make_output(answer="a3"))
        # Insert a fourth — q1 is oldest and should be evicted.
        cache.put("q4", _make_output(answer="a4"))
        self.assertEqual(len(cache), 3)
        # q1 missing.
        self.assertNotIn("q1", cache)
        self.assertIn("q2", cache)
        self.assertIn("q3", cache)
        self.assertIn("q4", cache)

    def test_contains_reflects_state(self) -> None:
        cache = ResponseCache()
        self.assertNotIn("q", cache)
        cache.put("q", _make_output())
        self.assertIn("q", cache)
        # Normalization-sensitive.
        self.assertIn("  Q  ", cache)

    def test_clear_empties_and_resets_stats(self) -> None:
        cache = ResponseCache()
        cache.put("q", _make_output())
        cache.get("q")  # hit
        cache.get("absent")  # miss
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.stats.hits, 1)
        self.assertEqual(cache.stats.misses, 1)
        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.stats.hits, 0)
        self.assertEqual(cache.stats.misses, 0)

    def test_stats_hit_rate(self) -> None:
        cache = ResponseCache()
        self.assertEqual(cache.stats.hit_rate, 0.0)
        cache.put("q", _make_output())
        cache.get("q")  # hit
        cache.get("q")  # hit
        cache.get("absent")  # miss
        self.assertEqual(cache.stats.hits, 2)
        self.assertEqual(cache.stats.misses, 1)
        self.assertAlmostEqual(cache.stats.hit_rate, 2 / 3)

    def test_non_positive_max_entries_raises(self) -> None:
        with self.assertRaises(ValueError):
            ResponseCache(max_entries=0)
        with self.assertRaises(ValueError):
            ResponseCache(max_entries=-1)

    def test_thread_safe_concurrent_puts(self) -> None:
        cache = ResponseCache(max_entries=200)

        def worker(start: int) -> None:
            for i in range(start, start + 25):
                cache.put(f"q{i}", _make_output(answer=f"a{i}"))

        threads = [threading.Thread(target=worker, args=(i * 25,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # All 100 entries should have made it in without a crash.
        self.assertEqual(len(cache), 100)

    def test_whitespace_and_case_normalize_to_same_key(self) -> None:
        cache = ResponseCache()
        a = "  What  is   X?  "
        b = "WHAT IS X?"
        dummy = _make_output(answer="x")
        cache.put(a, dummy)
        self.assertIsNotNone(cache.get(b), "normalization should make these equivalent")

    def test_put_existing_key_updates_value_and_promotes(self) -> None:
        cache = ResponseCache(max_entries=3)
        cache.put("q1", _make_output(answer="first"))
        cache.put("q2", _make_output(answer="a2"))
        cache.put("q3", _make_output(answer="a3"))
        # Re-put q1 → promoted to MRU.
        cache.put("q1", _make_output(answer="second"))
        cache.put("q4", _make_output(answer="a4"))
        # q2 is now the oldest, not q1.
        self.assertIn("q1", cache)
        self.assertNotIn("q2", cache)
        got = cache.get("q1")
        assert got is not None
        self.assertEqual(got.answer, "second")


if __name__ == "__main__":
    unittest.main()
