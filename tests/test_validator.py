"""Tests for src.validator — SQLValidator policy gate + auto-LIMIT."""

import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.schema import ColumnInfo, SchemaCatalog  # noqa: E402
from src.validator import SQLValidator  # noqa: E402


def _make_validator(row_limit: int = 1000) -> SQLValidator:
    """Build a validator whose allowlist is the single table ``"t"``."""
    schema = SchemaCatalog(table="t", columns=())
    return SQLValidator(schema, row_limit=row_limit)


def _make_validator_with_columns(row_limit: int = 1000) -> SQLValidator:
    """Build a validator with a small column allowlist on ``gaming_mental_health``."""
    schema = SchemaCatalog(
        table="gaming_mental_health",
        columns=(
            ColumnInfo(name="age", sql_type="INTEGER", kind="numeric"),
            ColumnInfo(name="addiction_level", sql_type="REAL", kind="numeric"),
            ColumnInfo(name="gender", sql_type="TEXT", kind="categorical"),
        ),
    )
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

    # 14. Line-comment smuggling: validation must succeed (the trailing
    # ``-- DROP TABLE t`` is a comment, not a second statement) and the
    # round-tripped output must be fully stripped of comment syntax.
    def test_line_comment_smuggling_stripped(self) -> None:
        out = _make_validator().validate("SELECT * FROM t -- DROP TABLE t")
        self.assertTrue(out.is_valid, msg=out.error)
        assert out.validated_sql is not None
        self.assertNotIn("--", out.validated_sql)
        self.assertNotIn("DROP", out.validated_sql.upper())

    # 15. Block-comment smuggling: validation must succeed and the
    # round-tripped output must contain no block-comment syntax or payload.
    def test_block_comment_smuggling_stripped(self) -> None:
        out = _make_validator().validate("SELECT * /* hack */ FROM t")
        self.assertTrue(out.is_valid, msg=out.error)
        assert out.validated_sql is not None
        self.assertNotIn("/*", out.validated_sql)
        self.assertNotIn("*/", out.validated_sql)
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

    # 24. UNION of two SELECTs against the allowed table passes and auto-LIMIT
    # is still injected on the top-level statement.
    def test_union_allowed_and_limit_injected(self) -> None:
        out = _make_validator().validate(
            "SELECT x FROM t WHERE x > 1 UNION SELECT x FROM t WHERE x < 0"
        )
        self.assertTrue(out.is_valid, msg=out.error)
        assert out.validated_sql is not None
        self.assertIn("UNION", out.validated_sql.upper())
        self.assertIn("LIMIT 1000", out.validated_sql.upper())

    # 25. CTE whose alias shadows the real table — the CTE body doesn't
    # reference any other table, so this must be accepted.
    def test_cte_shadowing_allowed_table_is_accepted(self) -> None:
        out = _make_validator().validate("WITH t AS (SELECT 1 AS x) SELECT * FROM t")
        self.assertTrue(out.is_valid, msg=out.error)

    # 26. CTE body references an unknown table — even though the outer
    # SELECT hits the CTE alias, the inner reference must be rejected.
    def test_cte_body_references_unknown_table_is_rejected(self) -> None:
        out = _make_validator().validate("WITH cte AS (SELECT x FROM evil) SELECT * FROM cte")
        self.assertFalse(out.is_valid)
        assert out.error is not None
        self.assertIn("evil", out.error)

    # 27. Non-positive ``row_limit`` must raise at construction time.
    def test_row_limit_must_be_positive(self) -> None:
        schema = SchemaCatalog(table="t", columns=())
        with self.assertRaises(ValueError):
            SQLValidator(schema, row_limit=0)
        with self.assertRaises(ValueError):
            SQLValidator(schema, row_limit=-5)

    # 28. Qualified references like ``other_db.t`` bypass the allowlist when
    # only ``table.name`` (last component) is compared. Must be rejected.
    def test_qualified_table_reference_rejected(self) -> None:
        out = _make_validator().validate("SELECT * FROM other_db.t")
        self.assertFalse(out.is_valid)
        assert out.error is not None
        self.assertIn("Qualified", out.error)

    # 29. Column allowlist: unknown column is rejected.
    def test_unknown_column_rejected(self) -> None:
        out = _make_validator_with_columns().validate(
            "SELECT zodiac_sign FROM gaming_mental_health"
        )
        self.assertFalse(out.is_valid)
        assert out.error is not None
        self.assertIn("zodiac_sign", out.error)

    # 30. Column allowlist: known column is accepted.
    def test_known_column_accepted(self) -> None:
        out = _make_validator_with_columns().validate(
            "SELECT addiction_level FROM gaming_mental_health"
        )
        self.assertTrue(out.is_valid, msg=out.error)

    # 31. Outer-SELECT alias is not treated as an unknown column when it
    # is later referenced in HAVING / ORDER BY.
    def test_column_alias_in_output_is_not_treated_as_unknown(self) -> None:
        out = _make_validator_with_columns().validate("SELECT age AS a FROM gaming_mental_health")
        self.assertTrue(out.is_valid, msg=out.error)

    # 32. CTE output name referenced in outer SELECT is allowed.
    def test_cte_output_referenced_in_outer_select_is_allowed(self) -> None:
        out = _make_validator_with_columns().validate(
            "WITH c AS (SELECT age AS x FROM gaming_mental_health) SELECT x FROM c"
        )
        self.assertTrue(out.is_valid, msg=out.error)

    # 33. COUNT(*) does not trip the column allowlist (star projection).
    def test_count_star_accepted(self) -> None:
        out = _make_validator_with_columns().validate("SELECT COUNT(*) FROM gaming_mental_health")
        self.assertTrue(out.is_valid, msg=out.error)

    # 34. Aggregate over a known column is accepted.
    def test_aggregate_over_known_column_accepted(self) -> None:
        out = _make_validator_with_columns().validate(
            "SELECT AVG(addiction_level) FROM gaming_mental_health"
        )
        self.assertTrue(out.is_valid, msg=out.error)

    # 35. Aggregate over an unknown column is rejected.
    def test_aggregate_over_unknown_column_rejected(self) -> None:
        out = _make_validator_with_columns().validate(
            "SELECT AVG(horoscope) FROM gaming_mental_health"
        )
        self.assertFalse(out.is_valid)
        assert out.error is not None
        self.assertIn("horoscope", out.error)


if __name__ == "__main__":
    unittest.main()
