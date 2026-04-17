"""Runtime configuration: single source of truth via pydantic-settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Frozen application settings loaded from env / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PIPELINE_",
        frozen=True,
        populate_by_name=True,
        extra="ignore",
    )

    # OpenRouter — uses unprefixed env vars (stock names from .env.example).
    openrouter_api_key: str = Field(
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "openrouter_api_key"),
        min_length=1,
    )
    model: str = Field(
        default="openai/gpt-5-nano",
        validation_alias=AliasChoices("OPENROUTER_MODEL", "model"),
    )

    # Database / table
    db_path: Path = Field(default=BASE_DIR / "data" / "gaming_mental_health.sqlite")
    table_name: str = Field(default="gaming_mental_health")

    # Row limits
    max_rows_return: int = Field(default=100, ge=1)
    max_rows_to_llm: int = Field(default=30, ge=1)
    sql_row_limit: int = Field(default=1000, ge=1)

    # Timeouts
    sql_timeout_s: float = Field(default=10.0, gt=0)
    llm_timeout_s: float = Field(default=30.0, gt=0)

    # Retry
    llm_retries: int = Field(default=1, ge=0)
    llm_retry_base_s: float = Field(default=0.3, gt=0)

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_format: Literal["json", "human"] = Field(default="json")

    # OpenTelemetry — uses the OTel standard unprefixed env var.
    otlp_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OTEL_EXPORTER_OTLP_ENDPOINT", "otlp_endpoint"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton (instantiated lazily)."""
    # Fields are populated from environment variables; mypy can't see that.
    return Settings()  # type: ignore[call-arg]
