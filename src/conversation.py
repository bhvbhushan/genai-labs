"""Multi-turn conversation state — in-memory ``ConversationStore`` + ``Turn``.

The store is a small, boring data structure whose only interesting property is
bounded memory. Two LRU caps keep it honest:

- **Conversation-level LRU** (``max_conversations``): when a new conversation
  arrives and we're at the cap, the least-recently-used conversation is evicted.
- **Per-conversation turn cap** (``max_turns_per_conversation``): each
  conversation keeps only the most-recent N turns; older ones drop off.

Rows stored in a ``Turn`` are also trimmed at append time (``max_rows_in_history``)
so a single wide query cannot balloon history memory.

The interface (``get_history`` / ``last_turns`` / ``append`` / ``clear``) is
designed so a Redis / Postgres backend is a drop-in later — the pipeline does
not know or care that the current implementation is an ``OrderedDict``.
"""

import time
from collections import OrderedDict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

FollowupIntent = Literal["NEW_QUERY", "FOLLOWUP_NEW_SQL", "FOLLOWUP_REINTERPRET"]


class Turn(BaseModel):
    """One turn of a conversation.

    ``rows`` is a frozen ``tuple`` of dicts so the model stays hashable-ish and
    immutable; the store trims rows at append time to ``max_rows_in_history``.
    """

    model_config = ConfigDict(frozen=True)

    question: str
    rewritten_question: str | None = None
    intent: FollowupIntent = "NEW_QUERY"
    sql: str | None = None
    rows: tuple[dict[str, Any], ...] = ()
    answer: str = ""
    created_at: float = Field(default_factory=lambda: time.time())


def summarize_rows(
    rows: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    max_rows: int = 5,
) -> str:
    """Compact text digest of ``rows`` for the followup classifier prompt.

    - Empty → ``"(no rows)"``.
    - Otherwise: header line (comma-joined column names) + up to ``max_rows``
      rows (comma-joined values). Overflow suffixed ``"... (N more)"``.

    This is data, not prompt text — keep it terse and byte-stable.
    """
    if not rows:
        return "(no rows)"

    columns = list(rows[0].keys())
    lines: list[str] = [", ".join(columns)]
    for row in rows[:max_rows]:
        values = [str(row.get(col, "")) for col in columns]
        lines.append(", ".join(values))

    remaining = len(rows) - max_rows
    if remaining > 0:
        lines.append(f"... ({remaining} more)")

    return "\n".join(lines)


class ConversationStore:
    """In-memory multi-turn store with LRU eviction.

    Not thread-safe; each pipeline request runs single-threaded. The class is
    structured so that swapping for Redis / a DB is trivial (put / get /
    delete). ``__contains__`` does NOT promote the entry to MRU — only
    ``get_history`` / ``last_turns`` / ``append`` do, so membership checks
    stay side-effect free.
    """

    def __init__(
        self,
        *,
        max_conversations: int = 1000,
        max_turns_per_conversation: int = 20,
        max_rows_in_history: int = 30,
    ) -> None:
        self._max_conversations = max_conversations
        self._max_turns_per_conversation = max_turns_per_conversation
        self._max_rows_in_history = max_rows_in_history
        # OrderedDict preserves insertion order; we use move_to_end on access
        # and on append so popitem(last=False) always drops the LRU entry.
        self._store: OrderedDict[str, list[Turn]] = OrderedDict()

    def get_history(self, conversation_id: str) -> list[Turn]:
        """Return all turns for ``conversation_id`` (oldest → newest)."""
        if conversation_id not in self._store:
            return []
        self._store.move_to_end(conversation_id)
        return list(self._store[conversation_id])

    def last_turns(self, conversation_id: str, n: int) -> list[Turn]:
        """Return the most-recent ``n`` turns (oldest → newest)."""
        if conversation_id not in self._store:
            return []
        self._store.move_to_end(conversation_id)
        turns = self._store[conversation_id]
        if n <= 0:
            return []
        return list(turns[-n:])

    def append(self, conversation_id: str, turn: Turn) -> None:
        """Append a turn; evict LRU convo on cap; trim per-convo turn cap."""
        # Trim rows on the Turn itself so every caller gets the invariant.
        if len(turn.rows) > self._max_rows_in_history:
            trimmed_rows = tuple(turn.rows[: self._max_rows_in_history])
            turn = turn.model_copy(update={"rows": trimmed_rows})

        if conversation_id in self._store:
            self._store.move_to_end(conversation_id)
            turns = self._store[conversation_id]
            turns.append(turn)
            # Drop oldest turns past the per-convo cap.
            overflow = len(turns) - self._max_turns_per_conversation
            if overflow > 0:
                del turns[:overflow]
            return

        # New conversation — evict LRU convo if at cap.
        if len(self._store) >= self._max_conversations:
            self._store.popitem(last=False)
        self._store[conversation_id] = [turn]

    def clear(self, conversation_id: str) -> None:
        """Remove ``conversation_id`` and all its turns. No-op if unknown."""
        self._store.pop(conversation_id, None)

    def __contains__(self, conversation_id: object) -> bool:
        return conversation_id in self._store

    def __len__(self) -> int:
        return len(self._store)


__all__ = [
    "ConversationStore",
    "FollowupIntent",
    "Turn",
    "summarize_rows",
]
