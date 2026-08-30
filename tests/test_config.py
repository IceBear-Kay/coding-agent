import pytest

from coding_agent.config import ProviderConfig
from coding_agent.errors import ProviderConfigurationError


def valid_environment() -> dict[str, str]:
    return {
        "DEEPSEEK_API_KEY": "test-secret-key",
        "DEEPSEEK_BASE_URL": "https://api.example.com/v1",
        "DEEPSEEK_MODEL": "test-model",
        "DEEPSEEK_TIMEOUT_SECONDS": "30",
    }


def test_provider_config_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in valid_environment().items():
        monkeypatch.setenv(name, value)

    config = ProviderConfig.from_env()

    assert config.api_key.get_secret_value() == "test-secret-key"
    assert "test-secret-key" not in repr(config)
    assert str(config.base_url) == "https://api.example.com/v1"
    assert config.model == "test-model"
    assert config.timeout_seconds == 30


def test_provider_config_reports_missing_variables_without_values() -> None:
    environment = valid_environment()
    environment.pop("DEEPSEEK_API_KEY")

    with pytest.raises(ProviderConfigurationError) as exc_info:
        ProviderConfig.from_env(environment)

    assert str(exc_info.value) == "Missing environment variables: DEEPSEEK_API_KEY"
    assert "test-secret-key" not in str(exc_info.value)


def test_provider_config_reports_invalid_variables_without_secret() -> None:
    environment = valid_environment()
    environment["DEEPSEEK_BASE_URL"] = "not-a-url"
    environment["DEEPSEEK_TIMEOUT_SECONDS"] = "-1"

    with pytest.raises(ProviderConfigurationError) as exc_info:
        ProviderConfig.from_env(environment)

    assert str(exc_info.value) == (
        "Invalid environment variables: DEEPSEEK_BASE_URL, DEEPSEEK_TIMEOUT_SECONDS"
    )
    assert "test-secret-key" not in str(exc_info.value)


@pytest.mark.parametrize("model", ["flash-model", "pro-model"])
def test_provider_config_selects_model_from_environment(model: str) -> None:
    environment = valid_environment()
    environment["DEEPSEEK_MODEL"] = model

    config = ProviderConfig.from_env(environment)

    assert config.model == model
