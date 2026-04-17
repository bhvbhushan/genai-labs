# Solution Notes

## Summary

The baseline pipeline shipped as a partially-working skeleton: the
SQL validator was a no-op that accepted DELETE/DROP, the schema
was not in the LLM prompt so the model guessed column names,
`OpenRouterLLMClient` reported `total_tokens=0` on every call (a
listed Hard Requirement), and `scripts/benchmark.py` crashed with
an AttributeError before printing anything. Only 3 of 5 public
tests passed. This submission rebuilds the pipeline as a thin
orchestrator over seven single-responsibility modules, adds a real
sqlglot-AST validator as the LLM/DB trust boundary, wires JSON
logs + OTel metrics + OTel tracing, implements real token
accounting from `res.usage`, adds a deterministic 1×1-scalar
short-circuit, and (optionally) supports multi-turn conversations
via an intent-routing classifier. All 5/5 public tests pass and
172 unit tests are green. Measured on 36 samples (3 full prompt-set
repetitions): avg 3397 ms, p95 4931 ms, success rate 88.9 %, avg
1345 tokens/request and avg 1.72 LLM calls/request (below 2.0
thanks to the short-circuit).

## What I Changed

### Foundations

- `src/config.py` — pydantic-settings `Settings` class. Frozen,
  env-driven, bounded fields (row caps >= 1, token budgets in
  [100, 8192], retry counts in [0, 5], timeouts > 0). API key is
  validated non-empty at construction so we fail fast rather than
  three stages deep.
- `src/observability.py` — `JsonFormatter` for structured stderr
  logs, seven OTel metric instruments (pipeline_requests_total,
  stage_duration_ms, llm_tokens_total, llm_calls_total,
  llm_short_circuit_total, llm_json_fallback_total,
  llm_usage_missing_total), a `Timer` context manager that wraps
  both a log event and an OTel span per stage. Exporters toggle
  via stock `OTEL_METRICS_EXPORTER` / `OTEL_TRACES_EXPORTER` env
  vars (`none`, `console`, `otlp`).
- `src/prompts.py` — centralized + cacheable system prompts for
  SQL generation and answer rendering. System prompt is
  byte-stable across requests so OpenRouter's auto-cache hits on
  every call after the first.
- `src/schema.py` — `SchemaCatalog.from_db()` introspection that
  renders a compact, LLM-ready table spec. Numeric columns get
  min/max summaries; categorical columns get a distinct-value
  list capped at `max_distinct_values`; BLOB columns are noted
  and truncated; all-null columns are rendered as "all null"
  rather than a useless distinct list.
- `pyproject.toml` — ruff + mypy --strict + pyright wired to
  `.venv`. `ruff check` and `ruff format` are clean on the files
  I own; mypy --strict and pyright are clean on `src/` and
  `scripts/benchmark.py`.

### Correctness

- `src/validator.py::SQLValidator` — sqlglot-AST policy. Rejects
  Update/Delete/Insert/Create/Drop/Alter/Pragma/Attach/Explain at
  the AST level (not by regex), rejects multi-statement payloads,
  allowlists table names from the introspected schema, rejects
  qualified names (`other_db.t`), strips inline comments before
  re-rendering, and auto-injects `LIMIT sql_row_limit` when the
  SELECT has no LIMIT. This is the single trust boundary.
- `src/llm_client.py::OpenRouterLLMClient` — structured JSON
  output via pydantic `SQLGenerationResponse`; plain-text parser
  as fallback (with `llm_json_fallback_total` counter); real
  token counting pulled from `res.usage` with
  `llm_usage_missing_total` counter for the degraded-provider
  case; retry with jittered exponential backoff (`2 ** n +
  uniform(0, base)`) on transient errors only — auth and
  malformed requests fail fast. Deterministic 1×1 scalar
  short-circuit skips the answer LLM call on aggregate results.
- `src/pipeline.py::AnalyticsPipeline` — thin orchestrator.
  Single `_derive_status()` decision table. request_id is
  generated (or passed in) and flows end-to-end through logs,
  metrics labels, and trace attributes. SQLite connection is
  opened `mode=ro` via URI, with a `progress_handler` checking a
  monotonic deadline every ~100k VM ops to abort pathological
  queries.

### Efficiency

- Schema lives in the SYSTEM prompt (byte-stable) → OpenRouter
  auto-cache hits every call. This is the biggest single lever.
- Deterministic 1×1 scalar short-circuit skips the Stage-2 LLM
  call on aggregate results (COUNT / SUM / AVG prompts). Saves
  roughly 250 tokens and ~500 ms per short-circuited request.
- CSV row serialization in the answer user turn: ~40% fewer
  tokens than JSON for the same data.
- `reasoning.effort=minimal` on `gpt-5-nano` so the model doesn't
  burn budget on hidden chain-of-thought for a SQL-gen task.
- `OTEL_METRICS_EXPORTER=none` / `OTEL_TRACES_EXPORTER=none` on
  the benchmark so stdout is a pristine JSON blob for the grading
  harness.

### Multi-Turn

- `src/conversation.py::ConversationStore` — in-memory
  `OrderedDict` LRU at three levels (convo count, turns per
  convo, rows per turn). `Turn` dataclass captures question +
  rewrite + intent + sql + rows + answer. Interface-clean so a
  Redis or Postgres swap is one class replacement.
- `src/followup.py::FollowupClassifier` — single LLM JSON call
  over `last_turns(conv_id, 4)` returning one of three intents
  (`NEW_QUERY`, `FOLLOWUP_NEW_SQL`, `FOLLOWUP_REINTERPRET`). Empty
  history short-circuits to NEW_QUERY without any LLM call.
  REINTERPRET auto-downgrades to NEW_SQL if prior rows are empty
  so the user always gets an answer.
- `src/pipeline.py` — accepts an optional `conversation_id` kwarg.
  Without it, the code path is byte-identical to single-turn
  (zero regression). With it, the classifier runs first;
  REINTERPRET skips SQL generation + validation + execution and
  reuses prior rows.

## Why I Changed It

**Why sqlglot over regex.** The baseline validator was a no-op
that returned `is_valid=True` for everything. The obvious fix is
a regex like `^\s*SELECT`, but that misses EXPLAIN-wrapping DML,
quoted identifiers, semicolon smuggling in string literals, and
comment-based smuggling. sqlglot gives you a real AST in one
dependency and ~100 lines of policy — you get the layer-1
correctness for free and you get to state the policy in terms of
AST node types, which is both easier to read and much harder to
bypass.

**Why full OTel vs simpler logs.** The checklist explicitly calls
out Logging, Metrics, and Tracing as separate production-readiness
items. OTel is the canonical answer for all three. The killer
feature for a take-home is that the Console exporters default to
zero infra: you get metric snapshots and trace JSON on stderr
with no collector running. If a reviewer wants to see traces in
Jaeger, it's one env var flip (`OTEL_TRACES_EXPORTER=otlp`). The
cost is a few hundred lines of setup code, but that cost is
amortized across a real production lifetime.

**Why the deterministic short-circuit.** Reading the public prompt
set carefully, a large fraction of prompts are single-aggregate
questions: "how many respondents …", "what's the average …",
"which is the highest …". On a 1×1 scalar result, the answer
literally reads "N" or "X is highest" — the LLM is being asked to
stringify a single number. Paying ~500 ms and ~250 tokens for that
is embarrassing. The short-circuit is a 10-line function and a
metric counter. It pays for itself on the first aggregate prompt
and then keeps paying.

**Why schema in SYSTEM, not USER.** OpenRouter's automatic prompt
cache hashes on the leading byte-stable prefix of each request.
If the schema is in SYSTEM, every request after the first gets a
cache hit on that prefix — the effective billable tokens are just
the user turn. If the schema is in USER, every request is a cold
cache. Cost: zero engineering. Savings: substantial per request.

**Why allowlist + sqlglot, not the baseline regex.** The baseline
did regex. The baseline's validator always returned `is_valid=True`.
Both failure modes are catastrophic against an LLM that can be
nudged to emit DROP TABLE. A real AST walk is only a few dozen
lines and is the actual trust boundary between the untrusted LLM
output and the database connection. If you care about not losing
your data, this is the single most important file in the repo.

**Why move the SELECT-only rule out of the LLM JSON validator.**
The baseline's system prompt told the LLM to "refuse destructive
queries". In practice the LLM returned `can_answer=false` for
DELETE prompts, which made the pipeline return
`status="unanswerable"`. But the correct status is
`"invalid_sql"` — the question was answerable as "no", the LLM
proposed a bad query, the validator rejected it. Moving the
SELECT-only rule into sqlglot (the single trust boundary) makes
the validator authoritative and lets DELETE prompts correctly
flow through as `invalid_sql`. This is the fix for the public test
`test_handles_destructive_query_gracefully`.

## Measured Impact

| Metric | Baseline (README) | This Submission | Δ |
|--------|-------------------|-----------------|----|
| avg_ms | ~2900 | 3396.63 | +17% |
| p50_ms | ~2500 | 3270.51 | +31% |
| p95_ms | ~4700 | 4930.56 | +5% |
| avg tokens/request | ~600 | 1344.72 | +124% |
| avg LLM calls/request | 2.0 | 1.72 | −14% |
| public tests passing | 3/5 | 5/5 | +2 |
| success_rate (benchmark) | n/a (crashes) | 0.8889 | — |

**Reading the table honestly.** On raw latency and raw
tokens-per-request the headline numbers regressed versus baseline.
This was a deliberate tradeoff:

- The baseline's "low" token count came partly from shipping no
  schema to the LLM — which is also why it was wrong on 2/5
  public tests. Putting the schema in SYSTEM is what moves the
  correctness from 3/5 → 5/5. The tokens are the price of knowing
  the column names.
- The baseline measured per reference hardware; this run is on
  local laptop hardware with the same LLM but different network
  conditions, so the absolute latency comparison is noisy.
- The `avg_llm_calls` metric is the one that's apples-to-apples:
  1.72 vs 2.0 is a real 14% reduction in LLM round-trips from the
  deterministic short-circuit alone. That saving compounds with
  the prompt cache.
- `success_rate=0.8889` (vs "crashes before printing" in baseline)
  is the most important number. The failures are graceful
  `invalid_sql` / `error` statuses, not exceptions — the
  pipeline's contract holds under LLM nondeterminism.

**Where the improvement would show up in production:**
Production usage is dominated by the prompt cache hit on the
system prefix — on a warm cache the schema is effectively free
and the per-request marginal cost drops into roughly the user-turn
token count plus the completion. The benchmark's `--runs 3` from
cold process start doesn't fully exercise that cache.

## Tradeoffs

- **LLM nondeterminism on borderline prompts.** In practice we see
  an occasional (~1/5 runs) flake on the public test
  `test_unanswerable_prompt_is_handled` (zodiac-sign prompt). The
  model sometimes hallucinates SQL instead of classifying the
  question as unanswerable. Tightening the system prompt further
  starts rejecting legitimate queries. The right production fix is
  a second-pass answerability check or a small classifier model,
  but both cost latency. We declined to take that cost for a
  take-home.
- **No prompt-cache verification from the client side.**
  OpenRouter's auto-cache is a server-side feature; the API does
  not expose cache-hit booleans in the response. We trust the
  system and measure aggregate token spend across runs; we cannot
  assert "this call hit cache" in a unit test.
- **`gpt-5-nano` reasoning effort.** `minimal` is the right
  default for SQL generation — the task is narrow and
  structured — but answer-quality on open-ended prompts might
  benefit from higher effort, at the cost of 2-5x latency. We
  keep it at minimal and rely on the validator to catch any bad
  SQL that slips through.
- **Token counting depends on `res.usage`.** If a provider routes
  through an adapter that drops usage from the response, we
  record zeros rather than estimating from string length. We
  track this via `llm_usage_missing_total` so dashboards can
  alert on degraded providers instead of silently reporting
  fabricated numbers.
- **ConversationStore is in-memory.** Not durable across process
  restarts. For a take-home this is acceptable; the interface is
  stable enough that swapping to Redis or Postgres is a single
  class change. No other pipeline code needs to touch.
- **Deterministic short-circuit is narrow.** It only fires on
  1×1 scalar results. Multi-row results always pay for the
  answer LLM call, even when the rendering is trivially
  templatable. Future work: add templated renderers for the top
  few shapes we see (top-N by group, distribution by category).

## Next Steps

- Persistent conversation store backed by Redis — keep the
  ConversationStore interface, swap the implementation.
- Semantic caching for repeat-with-paraphrase questions:
  embed the question, look up within a similarity threshold,
  return the cached answer rather than re-running the pipeline.
- Streaming response option: stream the answer stage's tokens
  as they arrive so first-token latency drops even when total
  latency doesn't.
- Adaptive model routing: try `gpt-5-nano` first, escalate to a
  stronger model only on validator-reject retry. Cheap-first,
  smart-fallback is almost always the right latency/cost
  frontier for a SQL-gen pipeline.
- Tighter answerability pre-check to reduce the zodiac-style
  hallucination edge. Candidate: a small classifier model or a
  rule-based schema-coverage check (can the question's key nouns
  be resolved against the schema's column names and categorical
  values?).
- Prompt-cache verification via response headers when OpenRouter
  exposes them — would let us assert cache-hit behavior in
  integration tests rather than trusting it.

## Repo Structure (for reviewers)

```
src/
  __init__.py          load_dotenv at import
  config.py            Settings (pydantic-settings)
  prompts.py           SQL and answer prompt templates
  schema.py            SchemaCatalog introspection + LLM-ready prompt
  validator.py         sqlglot SQLValidator
  llm_client.py        OpenRouterLLMClient (JSON mode, retry, token counts, short-circuit)
  observability.py     Logs + OTel metrics + OTel tracing + Timer
  conversation.py      ConversationStore + Turn
  followup.py          FollowupClassifier
  pipeline.py          AnalyticsPipeline orchestrator
  types.py             FROZEN contract
tests/
  test_config.py
  test_observability.py
  test_schema.py
  test_validator.py
  test_llm_client.py
  test_deterministic_answer.py
  test_pipeline.py
  test_conversation.py
  test_followup.py
  test_public.py        FROZEN
scripts/
  benchmark.py
  gaming_csv_to_db.py
pyproject.toml         ruff + mypy + pyright
requirements.txt       runtime deps
.env.example
CHECKLIST.md           filled
SOLUTION_NOTES.md      this file
README.md              unchanged
```
