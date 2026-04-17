"""Semantic response cache for the analytics pipeline.

Exact-match-on-normalized-question cache. Intentionally simple: no
embedding similarity, no TTL, LRU eviction only. Hits the repeat-
prompt optimization path in the benchmark and in any production
scenario with paraphrase-free repetition without the cost of an
embedding model. A later iteration can add similarity-based lookup
(embed question, search with threshold) without touching callers.
"""

import hashlib
import threading
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from src.types import PipelineOutput, SQLGenerationOutput


def _normalize_question(q: str) -> str:
    """Trim + lowercase + collapse internal whitespace."""
    return " ".join(q.strip().lower().split())


def _question_key(q: str) -> str:
    normalized = _normalize_question(q)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0


class ResponseCache:
    """LRU cache of PipelineOutput by question hash. Thread-safe."""

    def __init__(self, *, max_entries: int = 1024) -> None:
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}")
        self._max = max_entries
        self._data: OrderedDict[str, PipelineOutput] = OrderedDict()
        self._lock = threading.Lock()
        self.stats = CacheStats()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, question: object) -> bool:
        if not isinstance(question, str):
            return False
        with self._lock:
            return _question_key(question) in self._data

    def get(self, question: str) -> PipelineOutput | None:
        """Return a DEEP COPY of the cached output with zeroed LLM stats
        + zeroed timings to reflect the zero cost of a cache hit.
        """
        key = _question_key(question)
        with self._lock:
            if key not in self._data:
                self.stats.misses += 1
                return None
            self._data.move_to_end(key)
            cached = self._data[key]
            self.stats.hits += 1
        return _rewrite_for_hit(cached)

    def put(self, question: str, output: PipelineOutput) -> None:
        """Store a successful response. Callers should gate on status=='success'."""
        key = _question_key(question)
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._data[key] = output
                return
            self._data[key] = output
            if len(self._data) > self._max:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.stats = CacheStats()


def _rewrite_for_hit(output: PipelineOutput) -> PipelineOutput:
    """Return a deep copy with zeroed LLM stats and a cache_hit marker."""
    copied = deepcopy(output)
    # Zero LLM stats and timings to reflect the zero cost of this call.
    zero_stats: dict[str, Any] = {
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "model": copied.total_llm_stats.get("model", "unknown"),
    }
    copied.total_llm_stats = zero_stats
    copied.timings = {
        "sql_generation_ms": 0.0,
        "sql_validation_ms": 0.0,
        "sql_execution_ms": 0.0,
        "answer_generation_ms": 0.0,
        "total_ms": 0.0,
    }
    # Preserve original SQL/rows/answer/status. Mark via intermediate_outputs
    # of sql_generation so it's observable that this was a cache hit.
    hit_marker: dict[str, Any] = {"cache_hit": True}
    if copied.sql_generation is not None:
        copied.sql_generation = _replace_intermediate(copied.sql_generation, hit_marker)
    return copied


def _replace_intermediate(
    output: SQLGenerationOutput, extra: dict[str, Any]
) -> SQLGenerationOutput:
    """Return a copy of ``output`` with ``extra`` prepended to intermediate_outputs."""
    new_inter = [extra, *output.intermediate_outputs]
    return replace(output, intermediate_outputs=new_inter)
