"""Tests for src.observability (logging, metrics, tracing, Timer)."""

import io
import json
import logging
import os
import re
import sys
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from opentelemetry.exporter.otlp.proto.http.metric_exporter import (  # noqa: E402
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: E402
    OTLPSpanExporter,
)
from src import observability  # noqa: E402
from src.config import Settings, get_settings  # noqa: E402
from src.observability import (  # noqa: E402
    JsonFormatter,
    Timer,
    _build_metric_exporter,
    _build_span_exporter,
    _reset_for_testing,
    configure_observability,
    get_logger,
    get_tracer,
    log_event,
)


def _build_test_settings(
    otlp_endpoint: str | None = None,
    *,
    log_format: str = "json",
    metrics_exporter: str = "console",
    traces_exporter: str = "console",
) -> Settings:
    env = {
        "OPENROUTER_API_KEY": "sk-test",
        "PIPELINE_LOG_FORMAT": log_format,
        "OTEL_METRICS_EXPORTER": metrics_exporter,
        "OTEL_TRACES_EXPORTER": traces_exporter,
    }
    if otlp_endpoint is not None:
        env["OTEL_EXPORTER_OTLP_ENDPOINT"] = otlp_endpoint
    with patch.dict(os.environ, env, clear=True):
        return Settings(_env_file=None)  # type: ignore[call-arg]


def _make_record(msg: str, **extras: object) -> logging.LogRecord:
    logger = logging.getLogger("test.observability.unit")
    return logger.makeRecord(
        name=logger.name,
        level=logging.INFO,
        fn="test.py",
        lno=1,
        msg=msg,
        args=(),
        exc_info=None,
        extra=extras or None,
    )


class JsonFormatterTests(unittest.TestCase):
    def test_basic_record_has_canonical_keys(self) -> None:
        fmt = JsonFormatter()
        record = _make_record("pipeline_start")
        out = json.loads(fmt.format(record))
        for key in ("ts", "level", "logger", "event", "message"):
            self.assertIn(key, out)
        self.assertEqual(out["event"], "pipeline_start")
        self.assertEqual(out["message"], "pipeline_start")
        self.assertEqual(out["level"], "INFO")

    def test_ts_is_iso8601_utc(self) -> None:
        fmt = JsonFormatter()
        record = _make_record("x")
        out = json.loads(fmt.format(record))
        ts: str = out["ts"]
        self.assertTrue(ts.endswith("Z"))
        # Strip Z and parse — should succeed.
        parsed = datetime.fromisoformat(ts[:-1])
        self.assertIsNotNone(parsed)

    def test_extras_included(self) -> None:
        fmt = JsonFormatter()
        record = _make_record(
            "evt",
            request_id="abc",
            stage="sql_generation",
            duration_ms=12.5,
            status="success",
        )
        out = json.loads(fmt.format(record))
        self.assertEqual(out["request_id"], "abc")
        self.assertEqual(out["stage"], "sql_generation")
        self.assertEqual(out["duration_ms"], 12.5)
        self.assertEqual(out["status"], "success")

    def test_none_values_omitted(self) -> None:
        fmt = JsonFormatter()
        record = _make_record(
            "evt",
            request_id=None,
            stage="sql",
            duration_ms=None,
            status=None,
            error=None,
        )
        out = json.loads(fmt.format(record))
        self.assertNotIn("request_id", out)
        self.assertNotIn("duration_ms", out)
        self.assertNotIn("status", out)
        self.assertNotIn("error", out)
        self.assertEqual(out["stage"], "sql")

    def test_standard_logrecord_attrs_not_leaked(self) -> None:
        fmt = JsonFormatter()
        record = _make_record("evt", stage="s")
        out = json.loads(fmt.format(record))
        # These are internal LogRecord fields; they must not appear.
        for leaked in ("msg", "args", "levelno", "pathname", "filename", "created"):
            self.assertNotIn(leaked, out)

    def test_arbitrary_extra_included(self) -> None:
        fmt = JsonFormatter()
        record = _make_record("evt", custom_field="hello")
        out = json.loads(fmt.format(record))
        self.assertEqual(out["custom_field"], "hello")


class TimerPlainTests(unittest.TestCase):
    def test_elapsed_is_nonneg_float(self) -> None:
        with Timer() as t:
            pass
        self.assertIsInstance(t.ms, float)
        self.assertGreaterEqual(t.ms, 0.0)

    def test_elapsed_nondecreasing_in_block(self) -> None:
        with Timer() as t:
            first = t.ms
            time.sleep(0.002)
            second = t.ms
        self.assertLessEqual(first, second)
        self.assertLessEqual(second, t.ms)

    def test_final_elapsed_frozen_after_exit(self) -> None:
        with Timer() as t:
            time.sleep(0.001)
        final = t.ms
        time.sleep(0.005)
        self.assertEqual(final, t.ms)


class TimerSpanTests(unittest.TestCase):
    def test_span_opened_with_attributes(self) -> None:
        span = MagicMock()
        span_cm = MagicMock()
        span_cm.__enter__.return_value = span
        span_cm.__exit__.return_value = False
        tracer = MagicMock()
        tracer.start_as_current_span.return_value = span_cm

        with (
            patch("src.observability.get_tracer", return_value=tracer),
            Timer("sql_generation", stage="sql") as t,
        ):
            t.set_attribute("rows", 3)

        tracer.start_as_current_span.assert_called_once_with(
            "sql_generation", attributes={"stage": "sql"}
        )
        span.set_attribute.assert_called_once_with("rows", 3)
        span_cm.__exit__.assert_called_once()

    def test_exception_sets_error_status_and_records(self) -> None:
        span = MagicMock()
        span_cm = MagicMock()
        span_cm.__enter__.return_value = span
        span_cm.__exit__.return_value = False
        tracer = MagicMock()
        tracer.start_as_current_span.return_value = span_cm

        boom = RuntimeError("boom")
        with (
            patch("src.observability.get_tracer", return_value=tracer),
            self.assertRaises(RuntimeError),
            Timer("bad"),
        ):
            raise boom

        span.set_status.assert_called_once()
        status_arg = span.set_status.call_args[0][0]
        # Status has status_code attribute.
        from opentelemetry.trace import StatusCode

        self.assertEqual(status_arg.status_code, StatusCode.ERROR)
        span.record_exception.assert_called_once_with(boom)

    def test_set_attribute_without_span_is_noop(self) -> None:
        # No span_name → no span; set_attribute must not raise.
        with Timer() as t:
            t.set_attribute("k", 1)  # should be a silent no-op
        # (nothing to assert beyond no-exception)


class ConfigureObservabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_for_testing()
        get_settings.cache_clear()

    def tearDown(self) -> None:
        _reset_for_testing()
        get_settings.cache_clear()

    def test_idempotent(self) -> None:
        settings = _build_test_settings()
        with patch("src.observability.get_settings", return_value=settings):
            configure_observability(
                metric_exporter=MagicMock(),
                span_exporter=MagicMock(),
            )
            self.assertTrue(observability._CONFIGURED)
            # Second call must not raise.
            configure_observability(
                metric_exporter=MagicMock(),
                span_exporter=MagicMock(),
            )
            self.assertTrue(observability._CONFIGURED)

    def test_instruments_registered(self) -> None:
        from opentelemetry.metrics import Counter, Histogram

        settings = _build_test_settings()
        with patch("src.observability.get_settings", return_value=settings):
            configure_observability(
                metric_exporter=MagicMock(),
                span_exporter=MagicMock(),
            )

        self.assertIsNotNone(observability.pipeline_requests_total)
        self.assertIsNotNone(observability.stage_duration_ms)
        self.assertIsNotNone(observability.llm_tokens_total)
        self.assertIsNotNone(observability.llm_calls_total)
        self.assertIsNotNone(observability.llm_short_circuit_total)
        self.assertIsNotNone(observability.llm_json_fallback_total)
        self.assertIsNotNone(observability.llm_usage_missing_total)

        self.assertIsInstance(observability.pipeline_requests_total, Counter)
        self.assertIsInstance(observability.stage_duration_ms, Histogram)
        self.assertIsInstance(observability.llm_tokens_total, Counter)
        self.assertIsInstance(observability.llm_calls_total, Counter)
        self.assertIsInstance(observability.llm_short_circuit_total, Counter)
        self.assertIsInstance(observability.llm_json_fallback_total, Counter)
        self.assertIsInstance(observability.llm_usage_missing_total, Counter)


class LogEventTests(unittest.TestCase):
    def test_log_event_emits_json_with_extras(self) -> None:
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(JsonFormatter())
        handler.setLevel(logging.INFO)

        logger = get_logger("test.log_event")
        logger.propagate = False
        logger.setLevel(logging.INFO)
        # Ensure a clean handler set for this test.
        for h in list(logger.handlers):
            logger.removeHandler(h)
        logger.addHandler(handler)

        try:
            log_event(
                logger,
                "pipeline_start",
                request_id="r1",
                stage="gen",
                duration_ms=7.25,
                status="success",
            )
        finally:
            logger.removeHandler(handler)

        line = buf.getvalue().strip()
        self.assertTrue(line, "expected a single JSON line")
        payload = json.loads(line)
        self.assertEqual(payload["event"], "pipeline_start")
        self.assertEqual(payload["request_id"], "r1")
        self.assertEqual(payload["stage"], "gen")
        self.assertEqual(payload["duration_ms"], 7.25)
        self.assertEqual(payload["status"], "success")
        self.assertNotIn("error", payload)  # None → omitted


class HumanLogFormatTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_for_testing()
        get_settings.cache_clear()

    def tearDown(self) -> None:
        # Strip any sentinel handlers we installed so the root logger is clean.
        root = logging.getLogger()
        for h in list(root.handlers):
            if getattr(h, "_genai_labs_handler", False):
                root.removeHandler(h)
        _reset_for_testing()
        get_settings.cache_clear()

    def test_human_log_format_produces_plain_text(self) -> None:
        settings = _build_test_settings(log_format="human")
        with patch("src.observability.get_settings", return_value=settings):
            configure_observability(
                metric_exporter=MagicMock(),
                span_exporter=MagicMock(),
            )

        # Capture output from the installed sentinel handler.
        root = logging.getLogger()
        sentinel = next(h for h in root.handlers if getattr(h, "_genai_labs_handler", False))
        buf = io.StringIO()
        assert isinstance(sentinel, logging.StreamHandler)
        sentinel.stream = buf  # type: ignore[assignment]

        logger = logging.getLogger("test.human")
        logger.info("hello")

        line = buf.getvalue().strip()
        self.assertRegex(
            line,
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \[INFO\] test\.human: hello$",
        )
        # Confirm it is NOT JSON (the json branch would produce '{"ts": ...}').
        self.assertFalse(line.startswith("{"))
        # Extra belt-and-braces check via re.match on the exact shape.
        self.assertIsNotNone(
            re.match(
                r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \[INFO\] test\.human: hello$",
                line,
            )
        )


class ExporterBranchTests(unittest.TestCase):
    def test_build_metric_exporter_returns_otlp_when_endpoint_set(self) -> None:
        exporter = _build_metric_exporter("http://localhost:4318/v1/metrics")
        self.assertIsInstance(exporter, OTLPMetricExporter)

    def test_build_span_exporter_returns_otlp_when_endpoint_set(self) -> None:
        exporter = _build_span_exporter("http://localhost:4318/v1/traces")
        self.assertIsInstance(exporter, OTLPSpanExporter)


class HandlerCleanupIdempotenceTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_for_testing()
        get_settings.cache_clear()
        # Wipe any leftover sentinels from prior tests.
        root = logging.getLogger()
        for h in list(root.handlers):
            if getattr(h, "_genai_labs_handler", False):
                root.removeHandler(h)

    def tearDown(self) -> None:
        root = logging.getLogger()
        for h in list(root.handlers):
            if getattr(h, "_genai_labs_handler", False):
                root.removeHandler(h)
        _reset_for_testing()
        get_settings.cache_clear()

    def test_double_configure_installs_exactly_one_handler(self) -> None:
        settings = _build_test_settings()
        with patch("src.observability.get_settings", return_value=settings):
            configure_observability(
                metric_exporter=MagicMock(),
                span_exporter=MagicMock(),
            )
            # Reset sentinel to force the second call to rerun the install path
            # — this exercises the "remove old handler, add new handler" loop.
            observability._CONFIGURED = False
            configure_observability(
                metric_exporter=MagicMock(),
                span_exporter=MagicMock(),
            )

        root = logging.getLogger()
        sentinels = [h for h in root.handlers if getattr(h, "_genai_labs_handler", False)]
        self.assertEqual(
            len(sentinels),
            1,
            f"handler cleanup must dedupe: found {len(sentinels)} sentinels",
        )


class ExporterNoneModeTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_for_testing()
        get_settings.cache_clear()

    def tearDown(self) -> None:
        root = logging.getLogger()
        for h in list(root.handlers):
            if getattr(h, "_genai_labs_handler", False):
                root.removeHandler(h)
        _reset_for_testing()
        get_settings.cache_clear()

    def test_metrics_none_instruments_still_usable(self) -> None:
        settings = _build_test_settings(metrics_exporter="none")
        with patch("src.observability.get_settings", return_value=settings):
            configure_observability(span_exporter=MagicMock())

        # Instruments are registered (non-None) even without a reader.
        self.assertIsNotNone(observability.pipeline_requests_total)
        # .add must not raise — no reader is attached but the instrument exists.
        assert observability.pipeline_requests_total is not None
        observability.pipeline_requests_total.add(1, {"status": "success"})

    def test_traces_none_tracer_is_noop(self) -> None:
        settings = _build_test_settings(traces_exporter="none")
        with patch("src.observability.get_settings", return_value=settings):
            configure_observability(metric_exporter=MagicMock())

        tracer = get_tracer()
        # start_as_current_span must return a working context manager even
        # when no span processor is attached (spans just don't get exported).
        with tracer.start_as_current_span("foo") as span:
            self.assertIsNotNone(span)

    def test_default_console_exporter_preserved(self) -> None:
        # No env overrides → default "console" exporter path is taken.
        settings = _build_test_settings()
        self.assertEqual(settings.metrics_exporter, "console")
        self.assertEqual(settings.traces_exporter, "console")

        with patch("src.observability.get_settings", return_value=settings):
            # No injected exporters — real Console exporters must be built.
            configure_observability()
        # Instruments still registered on the configured meter provider.
        self.assertIsNotNone(observability.pipeline_requests_total)


if __name__ == "__main__":
    unittest.main()
