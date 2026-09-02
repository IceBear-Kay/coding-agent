import pytest

from coding_agent.config import ProviderConfig, load_startup_environment, model_capability
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


def test_startup_dotenv_is_loaded_with_process_environment_precedence(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / ".env").write_text(
        "DEEPSEEK_API_KEY=file-key\nDEEPSEEK_BASE_URL=https://file.example\n"
        "DEEPSEEK_MODEL=deepseek-v4-flash\nDEEPSEEK_TIMEOUT_SECONDS=10\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "process-key")

    values = load_startup_environment(tmp_path)

    assert values["DEEPSEEK_API_KEY"] == "process-key"
    assert values["DEEPSEEK_BASE_URL"] == "https://file.example"
    capability = model_capability("deepseek-v4-pro")
    assert capability is not None
    assert capability.context_window_tokens == 1_000_000
    assert capability.max_output_tokens == 384_000
    assert model_capability("unknown-model") is None


@pytest.mark.parametrize("model", ["deepseek-v4-flash", "deepseek-v4-pro"])
def test_v4_model_capabilities_match_documented_limits(model: str) -> None:
    capability = model_capability(model)

    assert capability is not None
    assert capability.context_window_tokens == 1_000_000
    assert capability.max_output_tokens == 384_000


@pytest.mark.parametrize("model", ["deepseek-v4-flash", "deepseek-v4-pro"])
def test_v4_models_allow_65536_output_override(model: str) -> None:
    capability = model_capability(model)

    assert capability is not None
    assert capability.max_output_tokens >= 65_536
