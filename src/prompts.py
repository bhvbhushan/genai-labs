"""LLM prompt templates.

All prompt strings and render helpers live here. No imports from other
``src.*`` modules so that prompts remain independent of application state
and are byte-stable across calls with identical inputs.
"""

from typing import Final

SQL_GENERATION_SYSTEM: Final[str] = """\
You are a SQL assistant for analytics over a single SQLite table.

Schema:
{schema}

Rules:
- Dialect: SQLite.
- Translate the user's request literally into SQL. Do not refuse on the \
grounds that a request is DDL or DML — a downstream validator enforces \
policy. For destructive or write requests, still produce the literal SQL \
statement the user asked for.
- Always include a LIMIT clause when the statement is a SELECT.
- Reference only columns that exist in the schema above.
- Set can_answer=false only when the question truly cannot be mapped to the \
given schema (for example, asks about a column that does not exist).

Output: return strictly one JSON object with this shape and no prose:
{{"can_answer": bool, "sql": string | null, "reason": string | null}}

- If can_answer is true: sql is a valid SQLite statement, reason is null.
- If can_answer is false: sql is null and reason is a short explanation.
- The `sql` field must contain ONLY the SQL statement itself — no trailing \
comments, no explanatory prose, no `//`- or `--`-style notes. Put any \
commentary in the `reason` field if needed.
"""


ANSWER_GENERATION_SYSTEM: Final[str] = """\
You are a concise analytics assistant.

Rules:
- Use only the provided SQL result rows. Never invent data, columns, or values.
- If the rows are empty or insufficient, say so plainly.
- Respond in 1 to 3 sentences of plain English.
- No markdown, no bullet points, no apologies, no preamble.
"""


def render_sql_system(schema_prompt: str) -> str:
    """Build the SQL-generation system prompt with the schema injected."""
    return SQL_GENERATION_SYSTEM.format(schema=schema_prompt)


def render_sql_user(question: str) -> str:
    """Build the SQL-generation user prompt from a natural-language question."""
    return f"Question: {question}"


def render_answer_system() -> str:
    """Return the stable answer-generation system prompt."""
    return ANSWER_GENERATION_SYSTEM


def render_answer_user(question: str, sql: str, rows_csv: str) -> str:
    """Build the answer-generation user prompt.

    ``rows_csv`` is a pre-formatted CSV string (header row + data rows). This
    function does not build CSV; it accepts the already-formatted text.
    """
    return f"Question: {question}\nSQL: {sql}\nRows (CSV):\n{rows_csv}"
