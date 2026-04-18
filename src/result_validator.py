"""Schema-aware validation of SQL execution results.

Emits non-fatal warnings rather than errors: a query that returns
a numeric outside its column's declared min/max is suspicious but
still a valid result to surface, just with observability attached.

Used to close the 'result validation' slot of the CHECKLIST —
SQL validation is at the gate, this is at the output.
"""

import re
from dataclasses import dataclass
from typing import Any

from src.schema import SchemaCatalog

WARN_ZERO_ROWS_NO_FILTER = "zero_rows_no_filter"
WARN_NUMERIC_OUT_OF_RANGE = "numeric_out_of_range"
WARN_UNKNOWN_CATEGORICAL_VALUE = "unknown_categorical_value"

_WHERE_RE = re.compile(r"\bwhere\b", re.IGNORECASE)


@dataclass(frozen=True)
class ResultWarning:
    kind: str
    column: str | None
    value: Any
    detail: str


class ResultValidator:
    def __init__(self, schema: SchemaCatalog) -> None:
        self._schema = schema
        self._col_by_name = {c.name: c for c in schema.columns}

    def validate(
        self,
        rows: list[dict[str, Any]],
        sql: str | None,
    ) -> list[ResultWarning]:
        """Return 0 or more warnings about returned rows."""
        if sql is None:
            return []
        warnings: list[ResultWarning] = []

        # Rule 1: zero rows on a query with no WHERE clause is suspicious.
        # Cheap heuristic: if 'where' doesn't appear (case-insensitive) in
        # the SQL and rows == [], flag it. Don't bother parsing the AST
        # here; we already trust the validator before us.
        if not rows and not _WHERE_RE.search(sql):
            warnings.append(
                ResultWarning(
                    kind=WARN_ZERO_ROWS_NO_FILTER,
                    column=None,
                    value=None,
                    detail="Query has no WHERE clause but returned zero rows.",
                )
            )

        # Rule 2: numeric values outside the declared column range.
        for row in rows:
            for col_name, value in row.items():
                col = self._col_by_name.get(col_name)
                if col is None:
                    continue
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                v = float(value)
                if col.min_value is not None and v < col.min_value:
                    warnings.append(
                        ResultWarning(
                            kind=WARN_NUMERIC_OUT_OF_RANGE,
                            column=col_name,
                            value=v,
                            detail=f"{v} below declared min {col.min_value}",
                        )
                    )
                if col.max_value is not None and v > col.max_value:
                    warnings.append(
                        ResultWarning(
                            kind=WARN_NUMERIC_OUT_OF_RANGE,
                            column=col_name,
                            value=v,
                            detail=f"{v} above declared max {col.max_value}",
                        )
                    )

        # Rule 3: string values outside the declared categorical sample.
        for row in rows:
            for col_name, value in row.items():
                col = self._col_by_name.get(col_name)
                if col is None or col.kind != "categorical":
                    continue
                if col.sample_values is None:
                    continue
                if not isinstance(value, str):
                    continue
                if value not in col.sample_values:
                    warnings.append(
                        ResultWarning(
                            kind=WARN_UNKNOWN_CATEGORICAL_VALUE,
                            column=col_name,
                            value=value,
                            detail=f"value {value!r} not in sample {list(col.sample_values)}",
                        )
                    )
        return warnings
