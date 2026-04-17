"""Production observability: structured JSON logging, OTel metrics, OTel tracing.

Single entry point: ``configure_observability()``. Idempotent under a lock so
startup paths and tests can call it safely. Exporters default to file-based
Console output under ``.observability/{metrics,traces}.jsonl``; switch to
OTLP HTTP when ``settings.otlp_endpoint`` is set. Set
``OTEL_METRICS_EXPORTER=none`` / ``OTEL_TRACES_EXPORTER=none`` to register
instruments without attaching any reader / processor (opt-out for tests).

Public surface:
    configure_observability(...)    -> None
    shutdown_observability()        -> None  (flush exporters at process end)
    get_logger(name)                -> logging.Logger
    get_tracer()                    -> trace.Tracer
    Timer(span_name=..., **attrs)   -> context manager (plain timer or span)
    log_event(logger, event, ...)   -> canonical structured event emitter
    pipeline_requests_total, stage_duration_ms, llm_tokens_total,
    llm_calls_total, llm_short_circuit_total, llm_json_fallback_total,
    llm_usage_missing_total          -> OTel instruments (populated after
                                       configure_observability())
"""

import contextlib
import json
import logging
import os
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import IO, Any

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import Counter, Histogram
from opentelemetry.metrics import MeterProvider as APIMeterProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    MetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
)
from opentelemetry.trace import StatusCode
from opentelemetry.trace.status import Status

from src.config import get_settings

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OBSERVABILITY_DIR_NAME = ".observability"
# Export interval tuned so a ~30s benchmark produces at least one export flush.
_METRIC_EXPORT_INTERVAL_MS = 5000
# Max wait for a best-effort force-flush at process shutdown.
_FORCE_FLUSH_TIMEOUT_MS = 5000

_TRACER_NAME = "genai_labs.pipeline"
_METER_NAME = "genai_labs.pipeline"

# Standard LogRecord attributes we must strip before serializing extras.
_RESERVED_LOGRECORD_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "taskName",
    }
)

# Canonical ordered event keys for JSON output. None values are OMITTED.
_EVENT_KEYS: tuple[str, ...] = (
    "request_id",
    "stage",
    "duration_ms",
    "status",
    "error",
)

# ─────────────────────────────────────────────────────────────────────────────
# Metric instruments — populated by configure_observability().
# Typed as ``Counter | None`` / ``Histogram | None`` so imports work pre-init.
# ─────────────────────────────────────────────────────────────────────────────
pipeline_requests_total: Counter | None = None
stage_duration_ms: Histogram | None = None
llm_tokens_total: Counter | None = None
llm_calls_total: Counter | None = None
llm_short_circuit_total: Counter | None = None
llm_json_fallback_total: Counter | None = None
llm_usage_missing_total: Counter | None = None
response_cache_hits_total: Counter | None = None
response_cache_misses_total: Counter | None = None
result_validation_warnings_total: Counter | None = None
answer_hallucinations_total: Counter | None = None

_CONFIGURED: bool = False
_LOCK = threading.Lock()
# References kept so shutdown_observability() can force-flush the exact
# SDK-backed providers we installed — ``metrics.get_meter_provider()`` only
# returns the first provider ever set (OTel's global singleton policy), so
# under test or re-init the API-level lookup will point at a stale provider.
_METER_PROVIDER: MeterProvider | None = None
_TRACER_PROVIDER: TracerProvider | None = None


class JsonFormatter(logging.Formatter):
    """One JSON object per record. None-valued canonical keys are omitted."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=UTC)
        # Millisecond precision, explicit 'Z' suffix.
        iso_ts = ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"

        payload: dict[str, Any] = {
            "ts": iso_ts,
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "message": record.getMessage(),
        }

        # Pull canonical keys off the record (set via logger extra=).
        for key in _EVENT_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value

        # Pull any other user-supplied extras (skip reserved LogRecord attrs
        # and canonical keys already handled).
        skip = _RESERVED_LOGRECORD_KEYS | set(_EVENT_KEYS) | {"ts", "level", "logger", "event"}
        for key, value in record.__dict__.items():
            if key in skip or key.startswith("_"):
                continue
            if value is None:
                continue
            payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def _install_log_handler(level: str, log_format: str) -> None:
    """Attach a single stderr handler to the root logger (idempotent)."""
    root = logging.getLogger()
    root.setLevel(level)

    # Remove any handler we previously installed so level/format updates apply.
    for h in list(root.handlers):
        if getattr(h, "_genai_labs_handler", False):
            root.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(level)
    handler._genai_labs_handler = True  # type: ignore[attr-defined]
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(handler)


def _open_observability_file(kind: str) -> IO[Any]:
    """Open ``<repo_root>/.observability/<kind>.jsonl`` in append-text mode.

    ``kind`` is ``"metrics"`` or ``"traces"``. The parent directory is
    created on demand. The file object is returned to the caller and left
    open for the process lifetime — the OS flushes and closes it at exit.
    """
    out_dir = _REPO_ROOT / _OBSERVABILITY_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{kind}.jsonl"
    return path.open("a", encoding="utf-8")


def _console_stream(default: IO[Any]) -> IO[Any]:
    """Resolve the Console exporter stream.

    Default is the caller-provided file stream. ``PIPELINE_OTEL_EXPORTER_STREAM=stderr``
    swaps in ``sys.stderr`` for surgical debugging.
    """
    if os.environ.get("PIPELINE_OTEL_EXPORTER_STREAM", "").lower() == "stderr":
        return sys.stderr
    return default


def _build_metric_exporter(
    otlp_endpoint: str | None,
    exporter_name: str = "console",
) -> MetricExporter:
    """Pick a metric exporter. ``exporter_name="otlp"`` forces OTLP (requires
    ``otlp_endpoint``; falls back to Console with a WARN log if unset). Any
    other value uses OTLP if ``otlp_endpoint`` is set, else Console — where
    Console writes to ``.observability/metrics.jsonl`` by default.
    """
    if exporter_name == "otlp":
        if otlp_endpoint:
            return OTLPMetricExporter(endpoint=otlp_endpoint)
        logging.getLogger(__name__).warning(
            "OTEL_METRICS_EXPORTER=otlp but no OTLP endpoint set; falling back to Console."
        )
        return ConsoleMetricExporter(out=_console_stream(_open_observability_file("metrics")))
    if otlp_endpoint:
        return OTLPMetricExporter(endpoint=otlp_endpoint)
    return ConsoleMetricExporter(out=_console_stream(_open_observability_file("metrics")))


def _build_span_exporter(
    otlp_endpoint: str | None,
    exporter_name: str = "console",
) -> SpanExporter:
    """Pick a span exporter. See ``_build_metric_exporter`` for semantics."""
    if exporter_name == "otlp":
        if otlp_endpoint:
            return OTLPSpanExporter(endpoint=otlp_endpoint)
        logging.getLogger(__name__).warning(
            "OTEL_TRACES_EXPORTER=otlp but no OTLP endpoint set; falling back to Console."
        )
        return ConsoleSpanExporter(out=_console_stream(_open_observability_file("traces")))
    if otlp_endpoint:
        return OTLPSpanExporter(endpoint=otlp_endpoint)
    return ConsoleSpanExporter(out=_console_stream(_open_observability_file("traces")))


def _register_instruments(meter_provider: APIMeterProvider) -> None:
    """Create the module-level instruments against ``meter_provider``."""
    global pipeline_requests_total, stage_duration_ms, llm_tokens_total
    global llm_calls_total, llm_short_circuit_total, llm_json_fallback_total
    global llm_usage_missing_total
    global response_cache_hits_total, response_cache_misses_total
    global result_validation_warnings_total, answer_hallucinations_total

    meter = meter_provider.get_meter(_METER_NAME)

    pipeline_requests_total = meter.create_counter(
        name="pipeline_requests_total",
        description="Total pipeline requests by terminal status.",
        unit="1",
    )
    stage_duration_ms = meter.create_histogram(
        name="stage_duration_ms",
        description="Per-stage wall-clock duration in milliseconds.",
        unit="ms",
    )
    llm_tokens_total = meter.create_counter(
        name="llm_tokens_total",
        description="LLM tokens consumed, by stage and kind (prompt/completion).",
        unit="1",
    )
    llm_calls_total = meter.create_counter(
        name="llm_calls_total",
        description="LLM call count by stage and outcome (success/retry/error).",
        unit="1",
    )
    llm_short_circuit_total = meter.create_counter(
        name="llm_short_circuit_total",
        description="Pipeline short-circuits before reaching the LLM.",
        unit="1",
    )
    llm_json_fallback_total = meter.create_counter(
        name="llm_json_fallback_total",
        description="LLM JSON-parse fallbacks by stage.",
        unit="1",
    )
    llm_usage_missing_total = meter.create_counter(
        name="llm_usage_missing_total",
        description="LLM responses lacking token-usage metadata, by stage.",
        unit="1",
    )
    response_cache_hits_total = meter.create_counter(
        name="response_cache_hits_total",
        description="ResponseCache hits — requests served without calling the LLM.",
        unit="1",
    )
    response_cache_misses_total = meter.create_counter(
        name="response_cache_misses_total",
        description="ResponseCache misses — requests that fell through to the pipeline.",
        unit="1",
    )
    result_validation_warnings_total = meter.create_counter(
        name="result_validation_warnings_total",
        description="Schema-aware plausibility warnings on executed rows, by kind.",
        unit="1",
    )
    answer_hallucinations_total = meter.create_counter(
        name="answer_hallucinations_total",
        description="Answer-generation responses with numeric claims not found in rows.",
        unit="1",
    )


def configure_observability(
    *,
    metric_exporter: MetricExporter | None = None,
    span_exporter: SpanExporter | None = None,
) -> None:
    """Initialize logging, metrics, tracing. Idempotent and thread-safe.

    Optional ``metric_exporter`` / ``span_exporter`` arguments are injection
    points for tests; in production they are built from ``get_settings()``.
    """
    global _CONFIGURED, _METER_PROVIDER, _TRACER_PROVIDER
    with _LOCK:
        if _CONFIGURED:
            return

        settings = get_settings()
        _install_log_handler(settings.log_level, settings.log_format)

        # Metrics: "none" → instruments still register (no reader), zero export.
        if metric_exporter is None and settings.metrics_exporter == "none":
            meter_provider = MeterProvider(metric_readers=[])
        else:
            mexp = metric_exporter or _build_metric_exporter(
                settings.otlp_endpoint, settings.metrics_exporter
            )
            reader = PeriodicExportingMetricReader(
                mexp,
                export_interval_millis=_METRIC_EXPORT_INTERVAL_MS,
            )
            meter_provider = MeterProvider(metric_readers=[reader])
        metrics.set_meter_provider(meter_provider)
        _register_instruments(meter_provider)
        _METER_PROVIDER = meter_provider

        # Traces: "none" → no span processor attached; spans are no-ops.
        tracer_provider = TracerProvider()
        if span_exporter is not None or settings.traces_exporter != "none":
            sexp = span_exporter or _build_span_exporter(
                settings.otlp_endpoint, settings.traces_exporter
            )
            tracer_provider.add_span_processor(BatchSpanProcessor(sexp))
        trace.set_tracer_provider(tracer_provider)
        _TRACER_PROVIDER = tracer_provider

        _CONFIGURED = True


def _reset_for_testing() -> None:
    """Clear the idempotence sentinel and instrument refs. Tests only."""
    global _CONFIGURED, _METER_PROVIDER, _TRACER_PROVIDER
    global pipeline_requests_total, stage_duration_ms
    global llm_tokens_total, llm_calls_total, llm_short_circuit_total
    global llm_json_fallback_total, llm_usage_missing_total
    global response_cache_hits_total, response_cache_misses_total
    global result_validation_warnings_total, answer_hallucinations_total
    with _LOCK:
        _CONFIGURED = False
        _METER_PROVIDER = None
        _TRACER_PROVIDER = None
        pipeline_requests_total = None
        stage_duration_ms = None
        llm_tokens_total = None
        llm_calls_total = None
        llm_short_circuit_total = None
        llm_json_fallback_total = None
        llm_usage_missing_total = None
        response_cache_hits_total = None
        response_cache_misses_total = None
        result_validation_warnings_total = None
        answer_hallucinations_total = None


def shutdown_observability() -> None:
    """Flush OTel exporters. Safe to call before ``configure_observability``
    and safe to call multiple times. Never raises — shutdown is best-effort.

    Callers should invoke at process end (e.g. at the tail of a benchmark
    ``main()``) so a short run still exports its final metrics window.
    """
    with _LOCK:
        if not _CONFIGURED:
            return
        mp = _METER_PROVIDER
        tp = _TRACER_PROVIDER
    # Force-flush the SDK-backed providers we installed. ``metrics.get_meter_provider()``
    # may point at a stale singleton under re-init, so we hold direct references.
    # Exceptions are swallowed: shutdown is best-effort and must not raise.
    if mp is not None:
        with contextlib.suppress(Exception):
            mp.force_flush(timeout_millis=_FORCE_FLUSH_TIMEOUT_MS)
    if tp is not None:
        with contextlib.suppress(Exception):
            tp.force_flush(timeout_millis=_FORCE_FLUSH_TIMEOUT_MS)


def get_logger(name: str) -> logging.Logger:
    """Return a stdlib logger. Caller should prefer ``__name__``."""
    return logging.getLogger(name)


def get_tracer() -> trace.Tracer:
    """Return the pipeline-scoped tracer."""
    return trace.get_tracer(_TRACER_NAME)


class Timer:
    """Context manager: wall-clock milliseconds, optionally as an OTel span.

    Usage::

        with Timer() as t:
            ...work...
        print(t.ms)

        with Timer("sql_generation", stage="sql") as t:
            t.set_attribute("rows", 3)
    """

    def __init__(self, span_name: str | None = None, **span_attrs: Any) -> None:
        self._span_name = span_name
        self._span_attrs = span_attrs
        self._start: float = 0.0
        self._end: float | None = None
        self._span_cm: Any = None
        self._span: Any = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        if self._span_name is not None:
            tracer = get_tracer()
            self._span_cm = tracer.start_as_current_span(
                self._span_name, attributes=self._span_attrs
            )
            self._span = self._span_cm.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._end = time.perf_counter()
        if self._span is not None:
            if exc is not None:
                self._span.set_status(Status(StatusCode.ERROR, str(exc)))
                self._span.record_exception(exc)
            self._span_cm.__exit__(exc_type, exc, tb)

    def set_attribute(self, key: str, value: Any) -> None:
        """Forward an attribute to the open span (no-op without a span)."""
        if self._span is not None:
            self._span.set_attribute(key, value)

    @property
    def ms(self) -> float:
        """Elapsed milliseconds. Live while inside the with-block."""
        end = self._end if self._end is not None else time.perf_counter()
        return (end - self._start) * 1000.0


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    request_id: str | None = None,
    stage: str | None = None,
    duration_ms: float | None = None,
    status: str | None = None,
    error: str | None = None,
    **extra: Any,
) -> None:
    """Emit a canonical structured INFO log with stable keys."""
    payload: dict[str, Any] = {
        "request_id": request_id,
        "stage": stage,
        "duration_ms": duration_ms,
        "status": status,
        "error": error,
        **extra,
    }
    logger.info(event, extra=payload)
