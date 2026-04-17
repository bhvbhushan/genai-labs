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
- SELECT statements only. No DDL (CREATE/ALTER/DROP) and no DML \
(INSERT/UPDATE/DELETE/REPLACE).
- Always include a LIMIT clause.
- Reference only columns that exist in the schema above.
- If the question cannot be answered with the given schema, set \
can_answer=false.

Output: return strictly one JSON object with this shape and no prose:
{{"can_answer": bool, "sql": string | null, "reason": string | null}}

- If can_answer is true: sql is a valid SQLite SELECT string, reason is null.
- If can_answer is false: sql is null and reason is a short explanation.
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
