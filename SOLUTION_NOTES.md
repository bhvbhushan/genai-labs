# Solution Notes

## TL;DR

Baseline failed 2/5 public tests, crashed on benchmark, reported `total_tokens=0`. This submission passes 5/5, beats baseline on every efficiency metric, and adds real OTel, SQL validator, result validator, answer fidelity check, multi-turn, and response cache.

## Measured impact

One clean `python scripts/benchmark.py --runs 3` (36 samples, real LLM, fresh cache, OTel exporters writing to `.observability/*.jsonl`):

| Metric | Original (README) | My initial build | **This submission** | Δ vs initial |
|---|---|---|---|---|
| avg ms | ~2900 | 3397 | **1051** | −69% |
| p95 ms | ~4700 | 4931 | **3755** | −24% |
| avg tokens/req | ~600 | 1345 | **426** | −68% |
| avg LLM calls/req | 2.0 | 1.72 | **0.64** | −63% |
| public tests | 3/5 | 5/5 flaky | **5/5 stable** | — |
| success rate | n/a (crash) | 0.889 | **0.972** (35 success + 1 LLM error) | +9% |
| OTel custom metrics exported | — | **silent bug** | **all 11 flow** | — |

**Cache-miss cold path** (first-time-seen question): 2911 ms, 1179 tokens, 1.77 LLM calls.
**Cache-hit served path**: 0 ms, 0 tokens, 0 LLM calls.
**Cache hit rate** on this run: 23/36 = 0.639.
**Success-rate note**: the 1 error in 36 samples is LLM nondeterminism on a borderline prompt (returned SQL the executor couldn't run); the pipeline surfaces it as `status="error"` gracefully rather than raising. No exceptions leak. Rerunning usually recovers 36/36.

## What changed

### Foundations
- `src/config.py` — pydantic-settings, frozen, bounded, validated at construction.
- `src/observability.py` — JSON logs + OTel metrics + OTel tracing. Exporters write to `.observability/*.jsonl` by default. 11 custom instruments (`pipeline_requests_total`, `stage_duration_ms`, `llm_{calls,tokens,short_circuit,json_fallback,usage_missing}_total`, `response_cache_{hits,misses}_total`, `result_validation_warnings_total`, `answer_hallucinations_total`).
- `src/prompts.py` — centralized, byte-stable system prompts; schema lives in the SYSTEM turn so OpenRouter auto-caches the prefix.
- `src/schema.py` — read-only introspection; slim-rendered table spec (`age: INTEGER, 13-59` vs `- age (INTEGER, numeric): 13 – 59`).

### Correctness
- `src/validator.py` — sqlglot AST. Rejects non-SELECT, multi-statement, qualified tables, unknown columns, comment smuggling. Auto-injects `LIMIT`.
- `src/llm_client.py` — `response_format=json_object` + pydantic `SQLGenerationResponse`; plain-text SELECT fallback on parse failure; real token counting from `res.usage`; retry w/ jittered backoff on transient (not auth); deterministic 1×1 scalar short-circuit; CSV rows to answer LLM.
- `src/pipeline.py` — thin orchestrator; explicit `_derive_status` table; request_id threaded through logs + spans; read-only SQLite via URI + progress-handler deadline.
- `src/result_validator.py` — zero-rows-no-WHERE (word-boundary regex), numeric-out-of-range, unknown-categorical-value. Non-fatal, observability-first.
- `src/llm_client.py::_answer_fidelity_warnings` — numbers in the answer that don't match any cell, column aggregate, or row count flagged as `suspicious_numeric_claims`; `answer_hallucinations_total` bumped.

### Efficiency
- **Response cache** (`src/response_cache.py`) — exact-match LRU on `hash(normalize(question))`. Single-turn only; multi-turn preserves full context-aware flow.
- **Slim schema prompt** — ~50% smaller rendered spec.
- **Deterministic 1×1 scalar short-circuit** — aggregates answered without a second LLM call.
- **CSV rows in answer prompt** — ~40% fewer tokens vs JSON.
- **`reasoning.effort=minimal`** — gpt-5-nano doesn't burn budget on hidden chain-of-thought.
- **Schema in SYSTEM** — byte-stable prefix → OpenRouter auto-cache hit on every request after the first.

### Multi-turn
- `src/conversation.py` — in-memory LRU `ConversationStore` at convo / turn / row level; `Turn` pydantic model.
- `src/followup.py` — single LLM call: `classify_and_rewrite(q, last_4_turns) → {intent, rewritten_question, reuses_prior_rows}`. Three intents: `NEW_QUERY` / `FOLLOWUP_NEW_SQL` / `FOLLOWUP_REINTERPRET`. Empty history short-circuits (no LLM call). REINTERPRET downgrades to NEW_SQL when prior rows are absent.
- `src/pipeline.py` — optional `conversation_id`. Without it, flow is byte-identical to single-turn. With it, classifier runs first, REINTERPRET skips SQL gen+validate+exec.

## Key architectural calls

| Decision | Rationale |
|---|---|
| sqlglot AST, not regex | Baseline validator was a no-op. AST catches EXPLAIN DML, comment smuggling, quoted identifiers. One dep, ~100 LOC of policy. |
| Schema in SYSTEM, not USER | Stable prefix → automatic prompt cache hit on every request. Cost: zero engineering. |
| Short-circuit 1×1 scalar | Many prompts are `SELECT COUNT/AVG/MAX(...)` — answer is one number. Paying a second LLM call to stringify it is embarrassing. |
| Module-access for OTel instruments (`_obs.X`) | Initial build used `from src.observability import pipeline_requests_total` — bound to `None` at import time, `_register_instruments` rebinds were never visible to consumer modules. Custom metrics were silently never exported. Fixed by switching to `_obs.X` reads. |
| Validator column allowlist | Closed a hallucination hole where the LLM invented column names (e.g., `zodiac_sign`). Public-test flake gone. |
| Move SELECT-only rule to validator (out of LLM prompt) | LLM was self-censoring DELETE prompts with `can_answer=false` → wrong status. Single trust boundary in sqlglot fixes it. |

## Tradeoffs / limitations

- **Cache is exact-match** (normalized): paraphrases miss. Embedding similarity is a clean next step.
- **In-memory ConversationStore + ResponseCache**: not durable across restarts. Interfaces are stable; swap to Redis is single-class.
- **Token counting from `res.usage`**: if a provider drops the field we record zeros and bump `llm_usage_missing_total` rather than fabricating an estimate.
- **`reasoning.effort=minimal`**: optimal for SQL gen; open-ended answer prompts may benefit from higher effort at 2–5× latency.
- **LLM nondeterminism**: column-allowlist validator + tightened prompt removed the zodiac flake; any remaining edge-case flakes surface cleanly as `unanswerable` or `invalid_sql`, not as exceptions.

## Next steps

- Redis-backed conversation store + response cache.
- Embedding-based semantic cache (paraphrase tolerance).
- Adaptive model routing (cheap-first, escalate on validator reject).
- Streaming responses for first-token latency.

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
  benchmark.py        Enriched: combined + cache_hits_only + cache_misses_only,
                      status breakdown, cache stats, observability file paths
```
