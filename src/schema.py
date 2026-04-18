"""Database schema introspection and LLM-readable prompt rendering.

``SchemaCatalog.from_db`` runs a small, deterministic sample (first N rows via
a ``LIMIT`` subquery) to classify each column as ``categorical``, ``numeric``,
or ``text``. It uses a read-only SQLite connection and is safe on the 10M-row
dataset because each per-column query is bounded by ``sample_rows``.

``SchemaCatalog.to_prompt`` renders a compact byte-stable schema block that
goes into the SQL-generation system prompt.
"""

import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.observability import Timer, get_logger, log_event

_logger = get_logger(__name__)

# SQLite declared-type prefixes that map to the "numeric" kind.
_NUMERIC_TYPE_PREFIXES: tuple[str, ...] = ("INT", "REAL", "FLOA", "DOUB", "NUM", "DEC")

# SQLite declared-type prefixes that indicate BLOB affinity.
_BLOB_TYPE_PREFIXES: tuple[str, ...] = ("BLOB", "BYTEA")


def _quote_ident(name: str) -> str:
    """Double-quote a SQLite identifier, escaping embedded double-quotes."""
    return '"' + name.replace('"', '""') + '"'


def _is_numeric_type(sql_type: str) -> bool:
    """Classify a declared SQLite type as numeric per SQLite affinity rules."""
    upper = sql_type.upper()
    return any(upper.startswith(prefix) for prefix in _NUMERIC_TYPE_PREFIXES)


def _is_blob_type(sql_type: str) -> bool:
    """Classify a declared SQLite type as BLOB affinity."""
    upper = sql_type.upper()
    return any(upper.startswith(prefix) for prefix in _BLOB_TYPE_PREFIXES)


def _format_num(value: float) -> str:
    """Render a numeric value for LLM prompts; drop trailing .0 for ints."""
    if value == int(value):
        return str(int(value))
    return str(value)


class ColumnInfo(BaseModel):
    """Metadata for one column, including a cardinality signal."""

    model_config = ConfigDict(frozen=True)

    name: str
    sql_type: str
    kind: Literal["categorical", "numeric", "text"]
    sample_values: tuple[str, ...] | None = None
    min_value: float | None = None
    max_value: float | None = None
    all_null: bool = False


class SchemaCatalog(BaseModel):
    """Immutable per-deployment snapshot of table schema + prompt rendering."""

    model_config = ConfigDict(frozen=True)

    table: str
    columns: tuple[ColumnInfo, ...]

    @classmethod
    def from_db(
        cls,
        db_path: Path,
        table: str,
        *,
        sample_rows: int = 10_000,
        sample_distinct_cap: int = 20,
    ) -> "SchemaCatalog":
        """Introspect ``table`` in the SQLite file at ``db_path`` (read-only)."""
        if sample_rows <= 0:
            raise ValueError(f"sample_rows must be positive, got {sample_rows}")
        if sample_distinct_cap <= 0:
            raise ValueError(f"sample_distinct_cap must be positive, got {sample_distinct_cap}")
        with Timer() as timer:
            uri = f"file:{db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            try:
                columns = cls._introspect(conn, table, db_path, sample_rows, sample_distinct_cap)
            finally:
                conn.close()
            catalog = cls(table=table, columns=columns)

        log_event(
            _logger,
            "schema_introspected",
            stage="init",
            duration_ms=timer.ms,
            table=table,
            columns=len(columns),
        )
        return catalog

    @staticmethod
    def _introspect(
        conn: sqlite3.Connection,
        table: str,
        db_path: Path,
        sample_rows: int,
        sample_distinct_cap: int,
    ) -> tuple[ColumnInfo, ...]:
        """Run PRAGMA + per-column sampling. Returns a tuple of ColumnInfo."""
        quoted_table = _quote_ident(table)
        cursor = conn.execute(f"PRAGMA table_info({quoted_table})")
        pragma_rows = cursor.fetchall()
        if not pragma_rows:
            raise ValueError(f"Table {table!r} not found in {db_path}")

        infos: list[ColumnInfo] = []
        # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk).
        for _cid, col_name, declared_type, *_rest in pragma_rows:
            sql_type = str(declared_type) if declared_type else ""
            infos.append(
                _classify_column(
                    conn,
                    table=table,
                    col_name=col_name,
                    sql_type=sql_type,
                    sample_rows=sample_rows,
                    sample_distinct_cap=sample_distinct_cap,
                )
            )
        return tuple(infos)

    def to_prompt(self) -> str:
        """Render a compact, byte-stable schema block for LLM consumption."""
        lines: list[str] = [f"Table: {self.table}", "Columns:"]
        for col in self.columns:
            lines.append(_render_column(col))
        return "\n".join(lines)

    def column_names(self) -> frozenset[str]:
        """Return the set of column names as a frozenset."""
        return frozenset(col.name for col in self.columns)


def _classify_column(
    conn: sqlite3.Connection,
    *,
    table: str,
    col_name: str,
    sql_type: str,
    sample_rows: int,
    sample_distinct_cap: int,
) -> ColumnInfo:
    """Sample one column and produce its ``ColumnInfo``."""
    quoted_col = _quote_ident(col_name)
    quoted_table = _quote_ident(table)
    is_numeric = _is_numeric_type(sql_type)
    is_blob = _is_blob_type(sql_type)

    # BLOB: short-circuit to text kind with no sample values — avoids leaking
    # Python bytes-repr (b'\\x00...') into the LLM prompt.
    if is_blob:
        probe_sql = f"SELECT 1 FROM {quoted_table} WHERE {quoted_col} IS NOT NULL LIMIT 1"
        probe = conn.execute(probe_sql).fetchone()
        return ColumnInfo(
            name=col_name,
            sql_type=sql_type,
            kind="text",
            all_null=probe is None,
        )

    distinct_sql = (
        f"SELECT DISTINCT {quoted_col} FROM "
        f"(SELECT {quoted_col} FROM {quoted_table} LIMIT ?) "
        f"WHERE {quoted_col} IS NOT NULL LIMIT ?"
    )
    rows = conn.execute(distinct_sql, (sample_rows, sample_distinct_cap + 1)).fetchall()

    if not rows:
        kind: Literal["categorical", "numeric", "text"] = "numeric" if is_numeric else "text"
        return ColumnInfo(name=col_name, sql_type=sql_type, kind=kind, all_null=True)

    if len(rows) <= sample_distinct_cap:
        if is_numeric:
            # Sort numerically so Likert-style values render as 0,1,2,...10
            # rather than the lexicographic 1,10,2,3,... — cleaner prompt +
            # stops the LLM from inferring a string enum.
            sorted_numeric = sorted(float(r[0]) for r in rows)
            values = tuple(_format_num(v) for v in sorted_numeric)
        else:
            values = tuple(sorted(str(r[0]) for r in rows))
        return ColumnInfo(
            name=col_name,
            sql_type=sql_type,
            kind="categorical",
            sample_values=values,
        )

    # More than the cap: numeric gets min/max, text gets no extras.
    if is_numeric:
        minmax_sql = (
            f"SELECT MIN({quoted_col}), MAX({quoted_col}) FROM "
            f"(SELECT {quoted_col} FROM {quoted_table} LIMIT ?)"
        )
        min_raw, max_raw = conn.execute(minmax_sql, (sample_rows,)).fetchone()
        return ColumnInfo(
            name=col_name,
            sql_type=sql_type,
            kind="numeric",
            min_value=float(min_raw) if min_raw is not None else None,
            max_value=float(max_raw) if max_raw is not None else None,
        )

    return ColumnInfo(name=col_name, sql_type=sql_type, kind="text")


def _is_contiguous_int_run(values: tuple[str, ...]) -> bool:
    """True when all strings parse as integers forming a gap-free run."""
    if not values:
        return False
    try:
        ints = sorted(int(v) for v in values)
    except ValueError:
        return False
    return ints == list(range(ints[0], ints[-1] + 1))


def _render_column(col: ColumnInfo) -> str:
    """Render one column line of the schema prompt.

    Compact format, ~7 tokens/column:
    - all-null:              "  name: TYPE (all-null)"
    - numeric w/ range:      "  name: TYPE, min-max"
    - categorical int run:   "  name: TYPE, min-max"
    - categorical ints:      "  name: TYPE, 1,3,7"
    - categorical text:      "  name: TYPE, Female/Male/Other"
    - free text:             "  name: TYPE"
    """
    base = f"  {col.name}: {col.sql_type}"
    if col.all_null:
        return f"{base} (all-null)"
    if col.kind == "numeric" and col.min_value is not None and col.max_value is not None:
        return f"{base}, {_format_num(col.min_value)}-{_format_num(col.max_value)}"
    if col.kind == "categorical" and col.sample_values is not None:
        values = col.sample_values
        if _is_contiguous_int_run(values):
            ints = sorted(int(v) for v in values)
            return f"{base}, {ints[0]}-{ints[-1]}"
        # All-int but non-contiguous: comma join for readability.
        try:
            ints_only = [int(v) for v in values]
        except ValueError:
            ints_only = None
        if ints_only is not None:
            return f"{base}, {', '.join(str(i) for i in ints_only)}"
        return f"{base}, {'/'.join(values)}"
    return base
