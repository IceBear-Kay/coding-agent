"""Configuration models for external model providers."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from dotenv import dotenv_values
from pydantic import AnyHttpUrl, BaseModel, Field, SecretStr, ValidationError

from coding_agent.errors import ProviderConfigurationError

ENVIRONMENT_FIELDS = {
    "api_key": "DEEPSEEK_API_KEY",
    "base_url": "DEEPSEEK_BASE_URL",
    "model": "DEEPSEEK_MODEL",
    "timeout_seconds": "DEEPSEEK_TIMEOUT_SECONDS",
}


@dataclass(frozen=True, slots=True)
class ModelCapability:
    """Known model limits used to coordinate input and output budgets."""

    context_window_tokens: int
    max_output_tokens: int


MODEL_CAPABILITIES: dict[str, ModelCapability] = {
    "deepseek-v4-flash": ModelCapability(1_000_000, 32_768),
    "deepseek-v4-pro": ModelCapability(1_000_000, 32_768),
}


def load_startup_environment(startup_dir: Path | None = None) -> dict[str, str]:
    """Merge supported values from startup-directory ``.env`` and the process."""
    directory = Path.cwd() if startup_dir is None else Path(startup_dir)
    file_values = dotenv_values(directory / ".env", interpolate=False)
    merged: dict[str, str] = {
        key: value
        for key, value in file_values.items()
        if key in ENVIRONMENT_FIELDS.values() and isinstance(value, str)
    }
    for name in ENVIRONMENT_FIELDS.values():
        value = os.environ.get(name)
        if value is not None:
            merged[name] = value
    return merged


def model_capability(model: str) -> ModelCapability | None:
    """Return known limits for a model, or ``None`` for an unknown identifier."""
    return MODEL_CAPABILITIES.get(model.strip().lower())


class ProviderConfig(BaseModel):
    """Validated settings required by an OpenAI-compatible provider."""

    api_key: SecretStr = Field(min_length=1)
    base_url: AnyHttpUrl
    model: str = Field(min_length=1)
    timeout_seconds: float = Field(gt=0)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Self:
        """Load DeepSeek provider settings from environment variables."""
        source = os.environ if environ is None else environ
        values: dict[str, str] = {}
        missing: list[str] = []

        for field_name, environment_name in ENVIRONMENT_FIELDS.items():
            value = source.get(environment_name)
            if value is None or not value.strip():
                missing.append(environment_name)
            else:
                values[field_name] = value.strip()

        if missing:
            names = ", ".join(missing)
            raise ProviderConfigurationError(f"Missing environment variables: {names}")

        try:
            return cls.model_validate(values)
        except ValidationError as exc:
            invalid = {
                ENVIRONMENT_FIELDS[str(error["loc"][0])] for error in exc.errors() if error["loc"]
            }
            names = ", ".join(sorted(invalid))
            raise ProviderConfigurationError(f"Invalid environment variables: {names}") from exc
