"""Tests for src.validator — SQLValidator policy gate + auto-LIMIT."""

import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.schema import SchemaCatalog  # noqa: E402
from src.validator import SQLValidator  # noqa: E402


def _make_validator(row_limit: int = 1000) -> SQLValidator:
    """Build a validator whose allowlist is the single table ``"t"``."""
    schema = SchemaCatalog(table="t", columns=())
    return SQLValidator(schema, row_limit=row_limit)


class SQLValidatorTests(unittest.TestCase):
    # 1. None input → rejected with "No SQL provided".
    def test_none_input_rejected(self) -> None:
        out = _make_validator().validate(None)
        self.assertFalse(out.is_valid)
        self.assertIsNone(out.validated_sql)
        self.assertEqual(out.error, "No SQL provided")

    # 2. Empty string → rejected.
    def test_empty_string_rejected(self) -> None:
        out = _make_validator().validate("")
        self.assertFalse(out.is_valid)
        self.assertEqual(out.error, "No SQL provided")

    # 3. Whitespace-only → rejected.
    def test_whitespace_only_rejected(self) -> None:
        out = _make_validator().validate("   \n\t  ")
        self.assertFalse(out.is_valid)
        self.assertEqual(out.error, "No SQL provided")

    # 4. Happy-path SELECT → valid, validated_sql starts with SELECT.
    def test_happy_path_select(self) -> None:
        out = _make_validator().validate("SELECT * FROM t WHERE x = 1")
        self.assertTrue(out.is_valid, msg=out.error)
        self.assertIsNotNone(out.validated_sql)
        assert out.validated_sql is not None  # for mypy
        self.assertTrue(out.validated_sql.upper().startswith("SELECT"))
        self.assertIsNone(out.error)

    # 5. DELETE rejected, error names the kind.
    def test_delete_rejected(self) -> None:
        out = _make_validator().validate("DELETE FROM t")
        self.assertFalse(out.is_valid)
        self.assertIsNone(out.validated_sql)
        assert out.error is not None
        self.assertIn("Delete", out.error)

    # 6. INSERT rejected.
    def test_insert_rejected(self) -> None:
        out = _make_validator().validate("INSERT INTO t VALUES (1)")
        self.assertFalse(out.is_valid)
        assert out.error is not None
        self.assertIn("Insert", out.error)

    # 7. UPDATE rejected.
    def test_update_rejected(self) -> None:
        out = _make_validator().validate("UPDATE t SET x = 1")
        self.assertFalse(out.is_valid)
        assert out.error is not None
        self.assertIn("Update", out.error)

    # 8. DROP TABLE rejected.
    def test_drop_table_rejected(self) -> None:
        out = _make_validator().validate("DROP TABLE t")
        self.assertFalse(out.is_valid)
        assert out.error is not None
        self.assertIn("Drop", out.error)

    # 9. CREATE TABLE rejected.
    def test_create_table_rejected(self) -> None:
        out = _make_validator().validate("CREATE TABLE x (a INTEGER)")
        self.assertFalse(out.is_valid)
        assert out.error is not None
        self.assertIn("Create", out.error)

    # 10. ALTER TABLE rejected.
    def test_alter_table_rejected(self) -> None:
        out = _make_validator().validate("ALTER TABLE t ADD COLUMN y INTEGER")
        self.assertFalse(out.is_valid)
        assert out.error is not None
        self.assertIn("Alter", out.error)

    # 11. PRAGMA rejected.
    def test_pragma_rejected(self) -> None:
        out = _make_validator().validate("PRAGMA table_info(t)")
        self.assertFalse(out.is_valid)
        self.assertIsNone(out.validated_sql)
        self.assertIsNotNone(out.error)

    # 12. ATTACH rejected.
    def test_attach_rejected(self) -> None:
        out = _make_validator().validate('ATTACH DATABASE "other.db" AS x')
        self.assertFalse(out.is_valid)
        self.assertIsNotNone(out.error)

    # 13. Multiple statements rejected.
    def test_multiple_statements_rejected(self) -> None:
        out = _make_validator().validate("SELECT 1; DELETE FROM t")
        self.assertFalse(out.is_valid)
        self.assertEqual(out.error, "Multiple statements not allowed")

    # 14. Line-comment smuggling: the validated (round-tripped) output
    # must not contain the raw line-comment marker.
    def test_line_comment_smuggling_stripped(self) -> None:
        out = _make_validator().validate("SELECT * FROM t -- DROP TABLE t")
        # Either rejected, or the round-tripped SQL has no -- artifact.
        if out.is_valid:
            assert out.validated_sql is not None
            self.assertNotIn("--", out.validated_sql)
            self.assertNotIn("DROP", out.validated_sql.upper())
        else:
            self.assertIsNotNone(out.error)

    # 15. Block-comment smuggling: validated output has no /*.
    def test_block_comment_smuggling_stripped(self) -> None:
        out = _make_validator().validate("SELECT * /* hack */ FROM t")
        self.assertTrue(out.is_valid, msg=out.error)
        assert out.validated_sql is not None
        self.assertNotIn("/*", out.validated_sql)
        self.assertNotIn("hack", out.validated_sql)

    # 16. Unknown table rejected with "Unknown table".
    def test_unknown_table_rejected(self) -> None:
        out = _make_validator().validate("SELECT * FROM other")
        self.assertFalse(out.is_valid)
        assert out.error is not None
        self.assertIn("Unknown table", out.error)
        self.assertIn("other", out.error)

    # 17. Auto-LIMIT injected when missing.
    def test_auto_limit_injected(self) -> None:
        out = _make_validator(row_limit=1000).validate("SELECT * FROM t")
        self.assertTrue(out.is_valid, msg=out.error)
        assert out.validated_sql is not None
        self.assertIn("LIMIT 1000", out.validated_sql.upper())

    # 18. Existing LIMIT preserved (not overwritten).
    def test_existing_limit_preserved(self) -> None:
        out = _make_validator(row_limit=1000).validate("SELECT * FROM t LIMIT 10")
        self.assertTrue(out.is_valid, msg=out.error)
        assert out.validated_sql is not None
        upper = out.validated_sql.upper()
        self.assertIn("LIMIT 10", upper)
        self.assertNotIn("LIMIT 1000", upper)

    # 19. CTE pass-through: inner terminal is SELECT.
    def test_cte_allowed(self) -> None:
        out = _make_validator().validate("WITH cte AS (SELECT x FROM t) SELECT * FROM cte")
        self.assertTrue(out.is_valid, msg=out.error)
        assert out.validated_sql is not None
        self.assertIn("WITH", out.validated_sql.upper())
        # The CTE alias 'cte' should not trigger the table allowlist.
        self.assertNotIn("Unknown table", out.validated_sql)

    # 20. Subquery-hidden DELETE: either "Multiple statements" or
    # "Disallowed". Both are acceptable outcomes.
    def test_delete_hidden_in_subquery_rejected(self) -> None:
        out = _make_validator().validate(
            "SELECT * FROM t WHERE x IN (SELECT x FROM t); DELETE FROM t"
        )
        self.assertFalse(out.is_valid)
        assert out.error is not None
        self.assertTrue(
            "Multiple statements" in out.error or "Disallowed" in out.error,
            msg=f"unexpected error: {out.error}",
        )

    # 21. Timing instrumentation: timing_ms is populated on happy path.
    def test_timing_ms_populated(self) -> None:
        out = _make_validator().validate("SELECT * FROM t")
        self.assertGreater(out.timing_ms, 0.0)

    # 22. Unparseable SQL → specific parse-error message.
    def test_unparseable_sql_rejected(self) -> None:
        out = _make_validator().validate("SELECT * FROM")
        self.assertFalse(out.is_valid)
        assert out.error is not None
        self.assertIn("Unparseable SQL", out.error)

    # 23. Case-insensitive table match.
    def test_table_match_case_insensitive(self) -> None:
        out = _make_validator().validate("SELECT * FROM T")
        self.assertTrue(out.is_valid, msg=out.error)


if __name__ == "__main__":
    unittest.main()
