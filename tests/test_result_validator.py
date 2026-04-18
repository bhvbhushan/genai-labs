"""Tests for src.result_validator — schema-aware result plausibility checks."""

import sys
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.result_validator import (  # noqa: E402
    WARN_NUMERIC_OUT_OF_RANGE,
    WARN_UNKNOWN_CATEGORICAL_VALUE,
    WARN_ZERO_ROWS_NO_FILTER,
    ResultValidator,
)
from src.schema import ColumnInfo, SchemaCatalog  # noqa: E402


def _schema() -> SchemaCatalog:
    return SchemaCatalog(
        table="t",
        columns=(
            ColumnInfo(
                name="age",
                sql_type="INTEGER",
                kind="numeric",
                min_value=13.0,
                max_value=59.0,
            ),
            ColumnInfo(
                name="addiction_level",
                sql_type="REAL",
                kind="numeric",
                min_value=0.0,
                max_value=10.0,
            ),
            ColumnInfo(
                name="gender",
                sql_type="TEXT",
                kind="categorical",
                sample_values=("Female", "Male", "Other"),
            ),
            ColumnInfo(
                name="note",
                sql_type="TEXT",
                kind="text",
            ),
        ),
    )


class ResultValidatorTests(unittest.TestCase):
    def test_zero_rows_no_where_flag(self) -> None:
        v = ResultValidator(_schema())
        warnings = v.validate([], "SELECT age FROM t")
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].kind, WARN_ZERO_ROWS_NO_FILTER)

    def test_zero_rows_with_where_not_flagged(self) -> None:
        v = ResultValidator(_schema())
        warnings = v.validate([], "SELECT age FROM t WHERE age > 1000")
        self.assertEqual(warnings, [])

    def test_numeric_above_max(self) -> None:
        v = ResultValidator(_schema())
        warnings = v.validate([{"age": 200}], "SELECT age FROM t WHERE 1=1")
        kinds = [w.kind for w in warnings]
        self.assertIn(WARN_NUMERIC_OUT_OF_RANGE, kinds)
        flagged = next(w for w in warnings if w.kind == WARN_NUMERIC_OUT_OF_RANGE)
        self.assertEqual(flagged.column, "age")
        self.assertEqual(flagged.value, 200.0)

    def test_numeric_below_min(self) -> None:
        v = ResultValidator(_schema())
        warnings = v.validate(
            [{"addiction_level": -1.5}],
            "SELECT addiction_level FROM t WHERE 1=1",
        )
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].kind, WARN_NUMERIC_OUT_OF_RANGE)
        self.assertEqual(warnings[0].column, "addiction_level")

    def test_numeric_inside_range_no_warning(self) -> None:
        v = ResultValidator(_schema())
        warnings = v.validate(
            [{"age": 25, "addiction_level": 5.5}],
            "SELECT age, addiction_level FROM t WHERE 1=1",
        )
        self.assertEqual(warnings, [])

    def test_bool_is_not_checked_as_numeric(self) -> None:
        v = ResultValidator(_schema())
        # True is-instance int but should be skipped — addiction_level's declared
        # max is 10 and True would parse as 1.0 (inside range anyway), but a
        # False value of 0 below min 0? 0 is not below 0 → fine. Use a column
        # whose min excludes 0: age has min 13. A bool False (0) would breach
        # it IF we didn't skip bools.
        warnings = v.validate([{"age": False}], "SELECT age FROM t WHERE 1=1")
        self.assertEqual(warnings, [])

    def test_categorical_known_value_ok(self) -> None:
        v = ResultValidator(_schema())
        warnings = v.validate([{"gender": "Male"}], "SELECT gender FROM t WHERE 1=1")
        self.assertEqual(warnings, [])

    def test_categorical_unknown_value_flagged(self) -> None:
        v = ResultValidator(_schema())
        warnings = v.validate(
            [{"gender": "Martian"}],
            "SELECT gender FROM t WHERE 1=1",
        )
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].kind, WARN_UNKNOWN_CATEGORICAL_VALUE)
        self.assertEqual(warnings[0].column, "gender")
        self.assertEqual(warnings[0].value, "Martian")

    def test_column_not_in_schema_ignored(self) -> None:
        # Aggregate aliases like "avg_age" aren't in the schema; the validator
        # must silently skip them rather than flagging "unknown column".
        v = ResultValidator(_schema())
        warnings = v.validate(
            [{"avg_age": 25.0}],
            "SELECT AVG(age) AS avg_age FROM t WHERE 1=1",
        )
        self.assertEqual(warnings, [])

    def test_empty_rows_with_none_sql(self) -> None:
        v = ResultValidator(_schema())
        self.assertEqual(v.validate([], None), [])

    def test_text_kind_not_flagged_even_with_unexpected_values(self) -> None:
        # Free-text columns (kind="text") have no sample enumeration; any
        # value is acceptable.
        v = ResultValidator(_schema())
        warnings = v.validate(
            [{"note": "anything goes here"}],
            "SELECT note FROM t WHERE 1=1",
        )
        self.assertEqual(warnings, [])

    def test_zero_rows_without_where_flags_even_if_identifier_contains_where(self) -> None:
        # Column name 'nowhere' contains substring 'where' but is not a WHERE
        # clause. The word-boundary regex catches this case.
        schema = SchemaCatalog(
            table="t",
            columns=(
                ColumnInfo(
                    name="nowhere",
                    sql_type="INTEGER",
                    kind="numeric",
                    min_value=0.0,
                    max_value=10.0,
                ),
            ),
        )
        v = ResultValidator(schema)
        warnings = v.validate([], "SELECT nowhere FROM t")
        kinds = [w.kind for w in warnings]
        self.assertIn(WARN_ZERO_ROWS_NO_FILTER, kinds)


if __name__ == "__main__":
    unittest.main()
