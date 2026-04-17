"""Production observability: structured JSON logging, OTel metrics, OTel tracing.

Single entry point: ``configure_observability()``. Idempotent under a lock so
startup paths and tests can call it safely. Exporters default to Console
(stderr); switch to OTLP HTTP when ``settings.otlp_endpoint`` is set.

Public surface:
    configure_observability(...)    -> None
    get_logger(name)                -> logging.Logger
    get_tracer()                    -> trace.Tracer
    Timer(span_name=..., **attrs)   -> context manager (plain timer or span)
    log_event(logger, event, ...)   -> canonical structured event emitter
    pipeline_requests_total, stage_duration_ms, llm_tokens_total,
    llm_calls_total, llm_short_circuit_total, llm_json_fallback_total,
    llm_usage_missing_total          -> OTel instruments (populated after
                                       configure_observability())
"""

import json
import logging
import sys
import threading
import time
from datetime import UTC, datetime
from types import TracebackType
from typing import Any

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

_CONFIGURED: bool = False
_LOCK = threading.Lock()


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


def _build_metric_exporter(otlp_endpoint: str | None) -> MetricExporter:
    if otlp_endpoint:
        return OTLPMetricExporter(endpoint=otlp_endpoint)
    return ConsoleMetricExporter(out=sys.stderr)


def _build_span_exporter(otlp_endpoint: str | None) -> SpanExporter:
    if otlp_endpoint:
        return OTLPSpanExporter(endpoint=otlp_endpoint)
    return ConsoleSpanExporter(out=sys.stderr)


def _register_instruments(meter_provider: APIMeterProvider) -> None:
    """Create the seven module-level instruments against ``meter_provider``."""
    global pipeline_requests_total, stage_duration_ms, llm_tokens_total
    global llm_calls_total, llm_short_circuit_total, llm_json_fallback_total
    global llm_usage_missing_total

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


def configure_observability(
    *,
    metric_exporter: MetricExporter | None = None,
    span_exporter: SpanExporter | None = None,
) -> None:
    """Initialize logging, metrics, tracing. Idempotent and thread-safe.

    Optional ``metric_exporter`` / ``span_exporter`` arguments are injection
    points for tests; in production they are built from ``get_settings()``.
    """
    global _CONFIGURED
    with _LOCK:
        if _CONFIGURED:
            return

        settings = get_settings()
        _install_log_handler(settings.log_level, settings.log_format)

        mexp = metric_exporter or _build_metric_exporter(settings.otlp_endpoint)
        reader = PeriodicExportingMetricReader(mexp)
        meter_provider = MeterProvider(metric_readers=[reader])
        metrics.set_meter_provider(meter_provider)
        _register_instruments(meter_provider)

        sexp = span_exporter or _build_span_exporter(settings.otlp_endpoint)
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(BatchSpanProcessor(sexp))
        trace.set_tracer_provider(tracer_provider)

        _CONFIGURED = True


def _reset_for_testing() -> None:
    """Clear the idempotence sentinel and instrument refs. Tests only."""
    global _CONFIGURED, pipeline_requests_total, stage_duration_ms
    global llm_tokens_total, llm_calls_total, llm_short_circuit_total
    global llm_json_fallback_total, llm_usage_missing_total
    with _LOCK:
        _CONFIGURED = False
        pipeline_requests_total = None
        stage_duration_ms = None
        llm_tokens_total = None
        llm_calls_total = None
        llm_short_circuit_total = None
        llm_json_fallback_total = None
        llm_usage_missing_total = None


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
