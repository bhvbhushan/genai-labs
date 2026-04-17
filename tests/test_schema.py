"""Tests for src.schema — SchemaCatalog introspection + prompt rendering."""

import os
import sqlite3
import sys
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.schema import ColumnInfo, SchemaCatalog  # noqa: E402


def _seed(
    db_path: Path,
    ddl: str,
    inserts: Sequence[tuple[str, Sequence[tuple[object, ...]]]],
) -> None:
    """Create a table via ``ddl`` and insert rows. ``inserts`` is a list of
    (parameterized SQL, list of row-tuples)."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(ddl)
        for sql, rows in inserts:
            conn.executemany(sql, rows)
        conn.commit()
    finally:
        conn.close()


class SchemaCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        # mkstemp returns (fd, path). We close the fd immediately so from_db
        # can reopen read-only by path; we clean up the tempfile in tearDown.
        fd, path = tempfile.mkstemp(suffix=".sqlite")
        os.close(fd)
        self.db_path = Path(path)

    def tearDown(self) -> None:
        self.db_path.unlink(missing_ok=True)

    def _get(self, catalog: SchemaCatalog, name: str) -> ColumnInfo:
        for col in catalog.columns:
            if col.name == name:
                return col
        raise AssertionError(f"Column {name!r} not found in catalog")

    # 1. Happy path — all three kinds.
    def test_happy_path_three_kinds(self) -> None:
        # gender: 2 distinct values (categorical).
        # score: 20 distinct numeric values > cap=5 → numeric with min/max.
        # note: 100 distinct TEXT values > cap=5 → text.
        # flag: 1 distinct + nulls → categorical (not all_null, since "x" exists).
        rows = [
            ("Male" if i % 2 == 0 else "Female", float(i % 20), f"free {i}", "x" if i % 3 else None)
            for i in range(100)
        ]
        _seed(
            self.db_path,
            "CREATE TABLE t (gender TEXT, score REAL, note TEXT, flag TEXT)",
            [("INSERT INTO t VALUES (?, ?, ?, ?)", rows)],
        )
        catalog = SchemaCatalog.from_db(self.db_path, "t", sample_distinct_cap=5)

        gender = self._get(catalog, "gender")
        self.assertEqual(gender.kind, "categorical")
        self.assertEqual(gender.sample_values, ("Female", "Male"))
        self.assertIsNone(gender.min_value)
        self.assertFalse(gender.all_null)

        score = self._get(catalog, "score")
        self.assertEqual(score.kind, "numeric")
        self.assertIsNone(score.sample_values)
        self.assertEqual(score.min_value, 0.0)
        self.assertEqual(score.max_value, 19.0)

        note = self._get(catalog, "note")
        self.assertEqual(note.kind, "text")
        self.assertIsNone(note.sample_values)
        self.assertIsNone(note.min_value)
        self.assertFalse(note.all_null)

        flag = self._get(catalog, "flag")
        self.assertEqual(flag.kind, "categorical")
        self.assertFalse(flag.all_null)
        self.assertEqual(flag.sample_values, ("x",))

    # 2. Table missing.
    def test_missing_table_raises(self) -> None:
        _seed(
            self.db_path,
            "CREATE TABLE real_table (x INTEGER)",
            [("INSERT INTO real_table VALUES (?)", [(1,)])],
        )
        with self.assertRaises(ValueError) as ctx:
            SchemaCatalog.from_db(self.db_path, "no_such_table")
        self.assertIn("no_such_table", str(ctx.exception))

    # 3. Empty table — schema introspectable, every column all-null.
    def test_empty_table_all_null(self) -> None:
        _seed(self.db_path, "CREATE TABLE empty_t (a INTEGER, b TEXT)", [])
        catalog = SchemaCatalog.from_db(self.db_path, "empty_t")
        self.assertEqual(len(catalog.columns), 2)
        for col in catalog.columns:
            self.assertTrue(col.all_null, f"expected all_null for {col.name}")
            self.assertIsNone(col.sample_values)
            self.assertIsNone(col.min_value)
            self.assertIsNone(col.max_value)
        self.assertEqual(self._get(catalog, "a").kind, "numeric")
        self.assertEqual(self._get(catalog, "b").kind, "text")

    # 4. All-NULL column.
    def test_all_null_column(self) -> None:
        _seed(
            self.db_path,
            "CREATE TABLE t (a INTEGER, b TEXT)",
            [("INSERT INTO t VALUES (?, ?)", [(i, None) for i in range(10)])],
        )
        catalog = SchemaCatalog.from_db(self.db_path, "t")
        b = self._get(catalog, "b")
        self.assertTrue(b.all_null)
        self.assertIsNone(b.sample_values)
        a = self._get(catalog, "a")
        self.assertFalse(a.all_null)

    # 5. Distinct-cap boundary — cap and cap+1.
    def test_distinct_cap_boundary(self) -> None:
        cap = 5
        # TEXT column at exactly `cap` distinct → categorical.
        # TEXT column at cap+1 → text (no sample_values).
        # Numeric column at cap+1 → numeric with min/max.
        _seed(
            self.db_path,
            "CREATE TABLE t (cat TEXT, txt TEXT, num REAL)",
            [
                (
                    "INSERT INTO t VALUES (?, ?, ?)",
                    [
                        (f"c{i % cap}", f"t{i % (cap + 1)}", float(i % (cap + 1)))
                        for i in range(200)
                    ],
                ),
            ],
        )
        catalog = SchemaCatalog.from_db(self.db_path, "t", sample_distinct_cap=cap)

        cat = self._get(catalog, "cat")
        self.assertEqual(cat.kind, "categorical")
        self.assertIsNotNone(cat.sample_values)
        assert cat.sample_values is not None  # for mypy
        self.assertEqual(len(cat.sample_values), cap)

        txt = self._get(catalog, "txt")
        self.assertEqual(txt.kind, "text")
        self.assertIsNone(txt.sample_values)

        num = self._get(catalog, "num")
        self.assertEqual(num.kind, "numeric")
        self.assertEqual(num.min_value, 0.0)
        self.assertEqual(num.max_value, float(cap))

    # 6. to_prompt is byte-stable and matches expected shape.
    def test_to_prompt_byte_stable_and_format(self) -> None:
        # s has 11 distinct numeric values > cap=3 → numeric with min/max.
        # c has 80 distinct text values > cap=3 → text (no sample list).
        rows = [
            ("Male" if i % 2 == 0 else "Female", float(i % 11), f"free {i}", None)
            for i in range(80)
        ]
        _seed(
            self.db_path,
            "CREATE TABLE t (g TEXT, s REAL, c TEXT, nul TEXT)",
            [("INSERT INTO t VALUES (?, ?, ?, ?)", rows)],
        )
        catalog = SchemaCatalog.from_db(self.db_path, "t", sample_distinct_cap=3)
        prompt_a = catalog.to_prompt()
        prompt_b = catalog.to_prompt()
        self.assertEqual(prompt_a, prompt_b)
        self.assertTrue(prompt_a.startswith("Table: t"))
        self.assertIn("Columns:", prompt_a)
        # Categorical: comma-joined values.
        self.assertIn("- g (TEXT, categorical): Female, Male", prompt_a)
        # Numeric: min dash max.
        self.assertIn("- s (REAL, numeric): 0.0 – 10.0", prompt_a)  # noqa: RUF001
        # Text: line with no colon-list trailing.
        self.assertIn("- c (TEXT, text)\n", prompt_a + "\n")
        self.assertNotIn("- c (TEXT, text):", prompt_a)
        # all-null:
        self.assertIn("- nul (TEXT, text): <all-null>", prompt_a)
        # No trailing whitespace on any line.
        for line in prompt_a.split("\n"):
            self.assertEqual(line, line.rstrip())
        # No blank lines.
        self.assertNotIn("\n\n", prompt_a)

    # 7. column_names returns a frozenset matching the tuple.
    def test_column_names_frozenset(self) -> None:
        _seed(self.db_path, "CREATE TABLE t (a INTEGER, b TEXT, c REAL)", [])
        catalog = SchemaCatalog.from_db(self.db_path, "t")
        names = catalog.column_names()
        self.assertIsInstance(names, frozenset)
        self.assertEqual(names, frozenset({"a", "b", "c"}))

    # 8. Identifier quoting with a space in the column name.
    def test_quoted_identifier_with_space(self) -> None:
        _seed(
            self.db_path,
            'CREATE TABLE t ("my col" TEXT)',
            [('INSERT INTO t ("my col") VALUES (?)', [("alpha",), ("beta",)])],
        )
        catalog = SchemaCatalog.from_db(self.db_path, "t", sample_distinct_cap=5)
        col = self._get(catalog, "my col")
        self.assertEqual(col.kind, "categorical")
        self.assertEqual(col.sample_values, ("alpha", "beta"))

    # 9. Read-only open: row count unchanged after introspection.
    def test_read_only_open(self) -> None:
        _seed(
            self.db_path,
            "CREATE TABLE t (a INTEGER)",
            [("INSERT INTO t VALUES (?)", [(i,) for i in range(10)])],
        )
        SchemaCatalog.from_db(self.db_path, "t")
        conn = sqlite3.connect(self.db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(count, 10)

    # 10. sample_rows caps how many rows inform the distinct set.
    def test_sample_rows_caps_distinct_set(self) -> None:
        rows: list[tuple[object, ...]] = [("A",) for _ in range(5)]
        rows.extend(("B",) for _ in range(995))
        _seed(
            self.db_path,
            "CREATE TABLE t (v TEXT)",
            [("INSERT INTO t VALUES (?)", rows)],
        )
        catalog = SchemaCatalog.from_db(self.db_path, "t", sample_rows=5, sample_distinct_cap=5)
        v = self._get(catalog, "v")
        self.assertEqual(v.kind, "categorical")
        self.assertEqual(v.sample_values, ("A",))


if __name__ == "__main__":
    unittest.main()
