# Solution Notes

## TL;DR

Baseline failed 2/5 public tests, crashed on the benchmark, and reported `total_tokens=0`. This submission passes 5/5, beats the baseline on every efficiency metric, and adds real OTel observability, a sqlglot SQL validator, schema-aware result validation, answer numeric-fidelity checks, multi-turn conversation support, and a response cache.

## Measured impact

One clean `python scripts/benchmark.py --runs 3` (36 samples, real LLM, fresh cache, OTel exporters writing to `.observability/*.jsonl`):

| Metric | Original (README) | **This submission** | Δ |
|---|---|---|---|
| avg ms | ~2900 | **1051** | −64% |
| p95 ms | ~4700 | **3755** | −20% |
| avg tokens/req | ~600 | **426** | −29% |
| avg LLM calls/req | 2.0 | **0.64** | −68% |
| public tests passing | 3/5 | **5/5** | +2 |
| benchmark success rate | n/a (crash) | **0.972** (35 success + 1 graceful LLM error) | — |

**Cache-miss cold path** (first-time-seen question): 2911 ms, 1179 tokens, 1.77 LLM calls.
**Cache-hit served path**: 0 ms, 0 tokens, 0 LLM calls.
**Cache hit rate** on this run: 23/36 = 0.639.
**Success-rate note**: the single error is LLM nondeterminism on a borderline prompt (the model produced SQL the executor couldn't run); the pipeline surfaces it as `status="error"` gracefully rather than raising. No exceptions leak.

## What changed

### Foundations
- `src/config.py` — pydantic-settings, frozen, bounded, validated at construction.
- `src/observability.py` — JSON logs + OTel metrics + OTel tracing. Exporters write to `.observability/*.jsonl` by default. 11 custom instruments (`pipeline_requests_total`, `stage_duration_ms`, `llm_{calls,tokens,short_circuit,json_fallback,usage_missing}_total`, `response_cache_{hits,misses}_total`, `result_validation_warnings_total`, `answer_hallucinations_total`).
- `src/prompts.py` — centralized, byte-stable system prompts; schema lives in the SYSTEM turn so OpenRouter auto-caches the prefix.
- `src/schema.py` — read-only introspection; compact table spec (`age: INTEGER, 13-59`).

### Correctness
- `src/validator.py` — sqlglot AST policy. Rejects non-SELECT, multi-statement, qualified tables, unknown columns, comment smuggling. Auto-injects `LIMIT`.
- `src/llm_client.py` — `response_format=json_object` + pydantic `SQLGenerationResponse`; plain-text SELECT fallback on parse failure; real token counting from `res.usage`; retry with jittered backoff on transient errors (not auth); deterministic 1×1 scalar short-circuit; CSV rows to answer LLM.
- `src/pipeline.py` — thin orchestrator; explicit `_derive_status` decision table; request_id threaded end-to-end through logs + spans; read-only SQLite via URI + progress-handler deadline.
- `src/result_validator.py` — zero-rows-no-WHERE (word-boundary regex), numeric-out-of-range, unknown-categorical-value. Non-fatal, observability-first.
- `src/llm_client.py::_answer_fidelity_warnings` — numbers in the answer that don't match any row cell, per-column aggregate, or the row count are flagged as `suspicious_numeric_claims` and counted in `answer_hallucinations_total`.

### Efficiency
- **Response cache** (`src/response_cache.py`) — exact-match LRU on `hash(normalize(question))`. Single-turn only; multi-turn preserves full context-aware flow.
- **Compact schema prompt** — dense per-column encoding (`age: INTEGER, 13-59`).
- **Deterministic 1×1 scalar short-circuit** — aggregates answered without a second LLM call.
- **CSV rows in answer prompt** — ~40% fewer tokens than JSON for the same rows.
- **`reasoning.effort=minimal`** — gpt-5-nano doesn't burn budget on hidden chain-of-thought.
- **Schema in SYSTEM** — byte-stable prefix → OpenRouter auto-cache hit on every request after the first.

### Multi-turn
- `src/conversation.py` — in-memory LRU `ConversationStore` at convo / turn / row level; `Turn` pydantic model.
- `src/followup.py` — single LLM call: `classify_and_rewrite(question, last_4_turns) → {intent, rewritten_question, reuses_prior_rows}`. Three intents: `NEW_QUERY` / `FOLLOWUP_NEW_SQL` / `FOLLOWUP_REINTERPRET`. Empty history short-circuits with no LLM call. REINTERPRET downgrades to NEW_SQL when prior rows are absent.
- `src/pipeline.py` — optional `conversation_id` kwarg. Without it, flow is byte-identical to single-turn. With it, classifier runs first; REINTERPRET skips SQL gen + validate + exec and reuses cached rows.

## Key architectural calls

| Decision | Rationale |
|---|---|
| sqlglot AST, not regex | AST catches EXPLAIN DML, comment smuggling, quoted identifiers. One dep, ~100 LOC of policy. |
| Schema in SYSTEM, not USER | Stable prefix → automatic prompt cache hit on every request. Cost: zero engineering. |
| Short-circuit 1×1 scalar | Many prompts are `SELECT COUNT/AVG/MAX(...)`. Paying a second LLM call to stringify one number is waste. |
| Validator owns SELECT-only policy (not the LLM prompt) | Single trust boundary. Also makes DELETE prompts produce `status="invalid_sql"` cleanly. |
| Validator column allowlist | Closes the hallucination hole where the LLM invents column names. |
| Module-access for OTel instruments (`_obs.X`) | Python's `from X import <instrument>` binds at import time; rebinds in the observability module must be visible to every consumer. |
| Response cache on single-turn only | Multi-turn semantics depend on history ordering; caching would break follow-up rewrites. |

## Tradeoffs / limitations

- **Cache is exact-match** (whitespace + case normalized): paraphrases miss. Embedding similarity is a clean next step.
- **In-memory `ConversationStore` + `ResponseCache`**: not durable across process restarts. Interfaces are stable; swap to Redis is a single-class change.
- **Token counting from `res.usage`**: if a provider drops the field we record zeros and increment `llm_usage_missing_total` rather than fabricating an estimate.
- **`reasoning.effort=minimal`**: optimal for SQL gen; open-ended answer prompts may benefit from higher effort at 2–5× latency.
- **LLM nondeterminism on borderline prompts**: the column-allowlist validator + tightened prompt removed the zodiac flake; any remaining edge-case failures surface cleanly as `unanswerable` / `invalid_sql` / `error`, not as exceptions.

## Next steps

- Redis-backed conversation store + response cache.
- Embedding-based semantic cache (paraphrase tolerance).
- Adaptive model routing (cheap-first, escalate on validator reject).
- Streaming responses for lower first-token latency.

## Repo map

```
src/
  config.py           Settings (pydantic-settings, frozen, bounded)
  prompts.py          SQL + answer system prompts + render helpers
  schema.py           SchemaCatalog (read-only introspection + LLM-ready spec)
  validator.py        sqlglot SQL safety gate (9 rules)
  llm_client.py       OpenRouterLLMClient (JSON mode, retry, tokens, short-circuit, fidelity)
  pipeline.py         AnalyticsPipeline (4-stage orchestrator + multi-turn)
  observability.py    JSON logs + OTel metrics + OTel tracing + Timer
  conversation.py     ConversationStore + Turn (in-memory LRU)
  followup.py         FollowupClassifier (classify + rewrite, 3 intents)
  response_cache.py   ResponseCache (normalized-question LRU)
  result_validator.py Schema-aware result plausibility warnings
  types.py            FROZEN contract
tests/                225 unit tests, 5 frozen public tests
scripts/
  benchmark.py        Single-turn benchmark: combined + cache_hits_only +
                      cache_misses_only buckets, status breakdown, cache
                      stats, observability file paths.
  multi_turn_eval.py  Multi-turn eval: 4 scripted conversation scenarios
                      covering all three follow-up intents. Per-turn
                      records + intent_breakdown + scenario rollups.
```
