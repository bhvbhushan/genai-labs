"""Tests for src.conversation — Turn model, summarize_rows, ConversationStore."""

import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from pydantic import ValidationError  # noqa: E402
from src.conversation import (  # noqa: E402
    ConversationStore,
    Turn,
    summarize_rows,
)


def _make_turn(
    question: str = "Q",
    *,
    answer: str = "A",
    rows: tuple[dict[str, object], ...] = (),
    sql: str | None = None,
) -> Turn:
    return Turn(question=question, answer=answer, rows=rows, sql=sql)


class TurnModelTests(unittest.TestCase):
    def test_turn_is_frozen(self) -> None:
        turn = _make_turn()
        with self.assertRaises(ValidationError):
            turn.question = "mutated"  # type: ignore[misc]

    def test_turn_defaults(self) -> None:
        turn = Turn(question="q")
        self.assertEqual(turn.answer, "")
        self.assertEqual(turn.rows, ())
        self.assertIsNone(turn.sql)
        self.assertIsNone(turn.rewritten_question)
        self.assertEqual(turn.intent, "NEW_QUERY")
        self.assertGreater(turn.created_at, 0.0)

    def test_turn_accepts_rows_and_sql(self) -> None:
        rows = ({"a": 1, "b": 2},)
        turn = Turn(question="q", sql="SELECT 1", rows=rows, answer="A")
        self.assertEqual(turn.rows, rows)
        self.assertEqual(turn.sql, "SELECT 1")


class SummarizeRowsTests(unittest.TestCase):
    def test_empty_rows(self) -> None:
        self.assertEqual(summarize_rows(()), "(no rows)")
        self.assertEqual(summarize_rows([]), "(no rows)")

    def test_single_row(self) -> None:
        out = summarize_rows([{"a": 1, "b": "x"}])
        self.assertIn("a, b", out)
        self.assertIn("1, x", out)

    def test_header_and_multiple_rows(self) -> None:
        rows = [{"a": i, "b": i * 2} for i in range(3)]
        out = summarize_rows(rows)
        lines = out.splitlines()
        self.assertEqual(lines[0], "a, b")
        self.assertEqual(len(lines), 4)  # header + 3 rows

    def test_overflow_suffix(self) -> None:
        rows = [{"v": i} for i in range(10)]
        out = summarize_rows(rows, max_rows=3)
        self.assertIn("... (7 more)", out)
        # header + 3 data rows + overflow line
        self.assertEqual(len(out.splitlines()), 5)


class ConversationStoreTests(unittest.TestCase):
    def test_empty_get_history(self) -> None:
        store = ConversationStore()
        self.assertEqual(store.get_history("unknown"), [])
        self.assertEqual(store.last_turns("unknown", 5), [])

    def test_append_and_get_roundtrip(self) -> None:
        store = ConversationStore()
        t1 = _make_turn("q1")
        t2 = _make_turn("q2")
        store.append("c1", t1)
        store.append("c1", t2)
        got = store.get_history("c1")
        self.assertEqual(len(got), 2)
        self.assertEqual(got[0].question, "q1")
        self.assertEqual(got[1].question, "q2")

    def test_last_turns(self) -> None:
        store = ConversationStore()
        for i in range(5):
            store.append("c1", _make_turn(f"q{i}"))
        last2 = store.last_turns("c1", 2)
        self.assertEqual([t.question for t in last2], ["q3", "q4"])

    def test_last_turns_zero_or_negative(self) -> None:
        store = ConversationStore()
        store.append("c1", _make_turn("q1"))
        self.assertEqual(store.last_turns("c1", 0), [])
        self.assertEqual(store.last_turns("c1", -5), [])

    def test_conversation_level_lru_eviction(self) -> None:
        store = ConversationStore(max_conversations=3)
        store.append("a", _make_turn("qa"))
        store.append("b", _make_turn("qb"))
        store.append("c", _make_turn("qc"))
        # Touch "a" so it's MRU; "b" becomes LRU.
        store.get_history("a")
        store.append("d", _make_turn("qd"))
        self.assertNotIn("b", store)
        self.assertIn("a", store)
        self.assertIn("c", store)
        self.assertIn("d", store)
        self.assertEqual(len(store), 3)

    def test_turn_level_cap(self) -> None:
        store = ConversationStore(max_turns_per_conversation=3)
        for i in range(5):
            store.append("c1", _make_turn(f"q{i}"))
        history = store.get_history("c1")
        self.assertEqual(len(history), 3)
        # Oldest two dropped; we keep q2, q3, q4.
        self.assertEqual([t.question for t in history], ["q2", "q3", "q4"])

    def test_row_trim_on_append(self) -> None:
        store = ConversationStore(max_rows_in_history=3)
        big_rows: tuple[dict[str, object], ...] = tuple({"v": i} for i in range(10))
        store.append("c1", _make_turn("q", rows=big_rows))
        stored = store.get_history("c1")[0]
        self.assertEqual(len(stored.rows), 3)
        self.assertEqual(stored.rows[0], {"v": 0})
        self.assertEqual(stored.rows[2], {"v": 2})

    def test_contains_and_len(self) -> None:
        store = ConversationStore()
        self.assertEqual(len(store), 0)
        self.assertNotIn("c1", store)
        store.append("c1", _make_turn())
        self.assertIn("c1", store)
        self.assertEqual(len(store), 1)

    def test_clear(self) -> None:
        store = ConversationStore()
        store.append("c1", _make_turn())
        store.append("c2", _make_turn())
        store.clear("c1")
        self.assertNotIn("c1", store)
        self.assertIn("c2", store)
        # Clearing unknown id is a no-op.
        store.clear("unknown")
        self.assertIn("c2", store)

    def test_get_history_promotes_to_mru(self) -> None:
        store = ConversationStore(max_conversations=2)
        store.append("a", _make_turn("qa"))
        store.append("b", _make_turn("qb"))
        # Access "a" so it's MRU; then add "c" -> "b" (LRU) gets evicted.
        store.get_history("a")
        store.append("c", _make_turn("qc"))
        self.assertIn("a", store)
        self.assertIn("c", store)
        self.assertNotIn("b", store)

    def test_append_on_existing_promotes_to_mru(self) -> None:
        store = ConversationStore(max_conversations=2)
        store.append("a", _make_turn("qa"))
        store.append("b", _make_turn("qb"))
        # Append to "a" so it's MRU; then add "c" -> "b" evicted.
        store.append("a", _make_turn("qa2"))
        store.append("c", _make_turn("qc"))
        self.assertIn("a", store)
        self.assertIn("c", store)
        self.assertNotIn("b", store)


if __name__ == "__main__":
    unittest.main()
