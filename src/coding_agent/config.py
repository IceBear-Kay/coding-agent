"""Configuration models for external model providers."""

import os
from collections.abc import Mapping
from typing import Self

from pydantic import AnyHttpUrl, BaseModel, Field, SecretStr, ValidationError

ENVIRONMENT_FIELDS = {
    "api_key": "DEEPSEEK_API_KEY",
    "base_url": "DEEPSEEK_BASE_URL",
    "model": "DEEPSEEK_MODEL",
    "timeout_seconds": "DEEPSEEK_TIMEOUT_SECONDS",
}


class ProviderConfigurationError(ValueError):
    """Raised when provider configuration is missing or invalid."""


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
