import pytest

from coding_agent.errors import (
    FatalProviderError,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderNetworkError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderServerError,
    TransientProviderError,
)


@pytest.mark.parametrize(
    "error_type",
    [ProviderNetworkError, ProviderRateLimitError, ProviderServerError],
)
def test_transient_provider_errors_are_retryable(error_type: type[ProviderError]) -> None:
    error = error_type("Temporary provider failure")

    assert isinstance(error, TransientProviderError)
    assert error.retryable is True


@pytest.mark.parametrize(
    "error_type",
    [
        ProviderConfigurationError,
        ProviderAuthenticationError,
        ProviderRequestError,
        ProviderResponseError,
    ],
)
def test_fatal_provider_errors_are_not_retryable(error_type: type[ProviderError]) -> None:
    error = error_type("Fatal provider failure")

    assert isinstance(error, FatalProviderError)
    assert error.retryable is False


def test_provider_errors_preserve_safe_diagnostic_messages() -> None:
    error = ProviderAuthenticationError("Provider authentication failed")

    assert str(error) == "Provider authentication failed"
