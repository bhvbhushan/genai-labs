"""Tests for src.config Settings / get_settings()."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pydantic

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.config import Settings, get_settings  # noqa: E402


class SettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        get_settings.cache_clear()

    def tearDown(self) -> None:
        get_settings.cache_clear()

    def test_defaults_with_api_key(self) -> None:
        env = {"OPENROUTER_API_KEY": "sk-test-1234"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]

        self.assertEqual(settings.openrouter_api_key, "sk-test-1234")
        self.assertEqual(settings.model, "openai/gpt-5-nano")
        self.assertEqual(settings.table_name, "gaming_mental_health")
        self.assertEqual(settings.max_rows_return, 100)
        self.assertEqual(settings.max_rows_to_llm, 30)
        self.assertEqual(settings.sql_row_limit, 1000)
        self.assertEqual(settings.sql_timeout_s, 10.0)
        self.assertEqual(settings.llm_timeout_s, 30.0)
        self.assertEqual(settings.llm_retries, 1)
        self.assertEqual(settings.llm_retry_base_s, 0.3)
        self.assertEqual(settings.log_level, "INFO")
        self.assertEqual(settings.log_format, "json")
        self.assertIsNone(settings.otlp_endpoint)
        self.assertTrue(str(settings.db_path).endswith("data/gaming_mental_health.sqlite"))

    def test_env_overrides_prefixed_fields(self) -> None:
        env = {
            "OPENROUTER_API_KEY": "sk-test-xyz",
            "PIPELINE_MODEL": "foo",
            "PIPELINE_MAX_ROWS_RETURN": "5",
            "PIPELINE_LOG_LEVEL": "DEBUG",
            "PIPELINE_LOG_FORMAT": "human",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]

        self.assertEqual(settings.model, "foo")
        self.assertEqual(settings.max_rows_return, 5)
        self.assertEqual(settings.log_level, "DEBUG")
        self.assertEqual(settings.log_format, "human")

    def test_openrouter_model_unprefixed_env_var(self) -> None:
        env = {
            "OPENROUTER_API_KEY": "sk-test",
            "OPENROUTER_MODEL": "anthropic/claude-test",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]

        self.assertEqual(settings.model, "anthropic/claude-test")

    def test_otlp_endpoint_unprefixed(self) -> None:
        env = {
            "OPENROUTER_API_KEY": "sk-test",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]

        self.assertEqual(settings.otlp_endpoint, "http://localhost:4318")

    def test_missing_api_key_raises(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaises(pydantic.ValidationError),
        ):
            Settings(_env_file=None)  # type: ignore[call-arg]

    def test_get_settings_caches(self) -> None:
        env = {"OPENROUTER_API_KEY": "sk-test-cache"}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env, clear=True):
            cwd = Path.cwd()
            os.chdir(tmp)
            try:
                get_settings.cache_clear()
                first = get_settings()
                second = get_settings()
            finally:
                os.chdir(cwd)
        self.assertIs(first, second)

    def test_llm_retries_zero_accepted(self) -> None:
        env = {
            "OPENROUTER_API_KEY": "sk-test",
            "PIPELINE_LLM_RETRIES": "0",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]

        self.assertEqual(settings.llm_retries, 0)

    def test_negative_max_rows_return_rejected(self) -> None:
        env = {
            "OPENROUTER_API_KEY": "sk-test",
            "PIPELINE_MAX_ROWS_RETURN": "-1",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            self.assertRaises(pydantic.ValidationError),
        ):
            Settings(_env_file=None)  # type: ignore[call-arg]

    def test_empty_api_key_rejected(self) -> None:
        env = {"OPENROUTER_API_KEY": ""}
        with (
            patch.dict(os.environ, env, clear=True),
            self.assertRaises(pydantic.ValidationError),
        ):
            Settings(_env_file=None)  # type: ignore[call-arg]

    def test_exporter_defaults_are_console(self) -> None:
        env = {"OPENROUTER_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]

        self.assertEqual(settings.metrics_exporter, "console")
        self.assertEqual(settings.traces_exporter, "console")

    def test_exporter_env_vars_lowercased(self) -> None:
        env = {
            "OPENROUTER_API_KEY": "sk-test",
            "OTEL_METRICS_EXPORTER": "NONE",
            "OTEL_TRACES_EXPORTER": "OtLp",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]

        self.assertEqual(settings.metrics_exporter, "none")
        self.assertEqual(settings.traces_exporter, "otlp")

    def test_settings_frozen(self) -> None:
        env = {"OPENROUTER_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings(_env_file=None)  # type: ignore[call-arg]

        with self.assertRaises(pydantic.ValidationError):
            settings.model = "mutated"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
