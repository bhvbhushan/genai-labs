"""SQL safety gate: sqlglot AST policy checks + auto-LIMIT injection.

The validator closes the trust boundary between LLM-generated SQL and the
SQLite executor. Rules are applied fail-fast in a fixed order (see
``SQLValidator.validate``). On rejection, a specific error string flows
through ``SQLValidationOutput`` so the pipeline can surface a useful status.

Happy-path validation is pure AST work — no I/O, no network — so it is
cheap on the hot path. The validator is constructed once per pipeline init
with the schema's single allowed table name and an auto-LIMIT cap.
"""

import time

from sqlglot import ParseError, exp, parse

from src.observability import get_logger, log_event
from src.schema import SchemaCatalog
from src.types import SQLValidationOutput

_logger = get_logger(__name__)

# Disallowed top-level statement kinds checked as a defence-in-depth walk
# over the AST. ``exp.Command`` catches PRAGMA/ATTACH/etc. that sqlglot
# parses into a generic Command node on some dialects. Every explicit DML /
# DDL class is listed alongside so the error message is specific.
_DISALLOWED_NODE_TYPES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Pragma,
    exp.Attach,
    exp.Detach,
    exp.Command,
)


class SQLValidator:
    """Validate generated SQL via sqlglot AST and policy rules."""

    def __init__(self, schema: SchemaCatalog, *, row_limit: int = 1000) -> None:
        if row_limit <= 0:
            raise ValueError(f"row_limit must be positive, got {row_limit}")
        self._allowed_table: str = schema.table.lower()
        self._row_limit: int = row_limit

    def validate(self, sql: str | None) -> SQLValidationOutput:
        """Return a ``SQLValidationOutput``; never raises on bad input."""
        start = time.perf_counter()
        result = self._validate_inner(sql)
        result.timing_ms = (time.perf_counter() - start) * 1000.0
        if not result.is_valid:
            log_event(
                _logger,
                "sql_rejected",
                stage="validation",
                duration_ms=result.timing_ms,
                error=result.error,
            )
        return result

    def _validate_inner(self, sql: str | None) -> SQLValidationOutput:
        # 1. None / empty input.
        if sql is None or not sql.strip():
            return SQLValidationOutput(is_valid=False, validated_sql=None, error="No SQL provided")

        # 2. Parse.
        try:
            parsed = parse(sql, read="sqlite")
        except ParseError as exc:
            return SQLValidationOutput(
                is_valid=False,
                validated_sql=None,
                error=f"Unparseable SQL: {exc}",
            )

        # Drop None entries (sqlglot yields None for empty statements).
        statements = [s for s in parsed if s is not None]

        # 3. Exactly one statement.
        if len(statements) == 0:
            return SQLValidationOutput(is_valid=False, validated_sql=None, error="No SQL provided")
        if len(statements) > 1:
            return SQLValidationOutput(
                is_valid=False,
                validated_sql=None,
                error="Multiple statements not allowed",
            )

        stmt = statements[0]

        # 4. Top-level must be SELECT or UNION (which has SELECT children).
        # sqlglot wraps WITH/CTE inside a Select, so the Select check covers
        # CTEs; Union is allowed because both sides are Selects per spec.
        if not isinstance(stmt, (exp.Select, exp.Union)):
            return SQLValidationOutput(
                is_valid=False,
                validated_sql=None,
                error=f"Non-SELECT statements are rejected: {stmt.__class__.__name__}",
            )

        # 5. Defence-in-depth: reject DML/DDL anywhere in the AST.
        for node in stmt.walk():
            if isinstance(node, _DISALLOWED_NODE_TYPES):
                return SQLValidationOutput(
                    is_valid=False,
                    validated_sql=None,
                    error=f"Disallowed operation: {node.__class__.__name__}",
                )

        # 6. Every referenced table must match the allowlist (excluding
        # CTE-defined names, which are synthetic aliases).
        cte_names: set[str] = {c.alias_or_name.lower() for c in stmt.find_all(exp.CTE)}
        for table in stmt.find_all(exp.Table):
            tname = table.name.lower()
            if tname in cte_names:
                continue
            if tname != self._allowed_table:
                return SQLValidationOutput(
                    is_valid=False,
                    validated_sql=None,
                    error=f"Unknown table: {table.name}",
                )

        # 7. Auto-LIMIT injection when absent. Preserve any existing limit
        # even if larger than our cap — executor enforces a hard row cap too.
        if stmt.args.get("limit") is None:
            stmt = stmt.limit(self._row_limit, copy=False)

        # 8 & 9. Round-trip strips comments (``comments=False``) and
        # normalizes whitespace, closing the comment-smuggling vector.
        validated = stmt.sql(dialect="sqlite", comments=False)

        return SQLValidationOutput(is_valid=True, validated_sql=validated, error=None)
