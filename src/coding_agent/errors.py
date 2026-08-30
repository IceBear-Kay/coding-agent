"""Internal errors raised by model providers."""


class ProviderError(Exception):
    """Base class for provider failures exposed to the agent loop."""

    retryable = False


class TransientProviderError(ProviderError):
    """A temporary failure that may succeed when retried."""

    retryable = True


class FatalProviderError(ProviderError):
    """A non-recoverable failure that should stop the current run."""


class ProviderNetworkError(TransientProviderError):
    """A network or timeout failure while contacting the provider."""


class ProviderRateLimitError(TransientProviderError):
    """The provider rejected a request because of rate limits."""


class ProviderServerError(TransientProviderError):
    """The provider failed while processing an otherwise valid request."""


class ProviderConfigurationError(FatalProviderError):
    """Required provider configuration is missing or invalid."""


class ProviderAuthenticationError(FatalProviderError):
    """The provider rejected the configured credentials."""


class ProviderRequestError(FatalProviderError):
    """The provider rejected a non-retryable request."""


class ProviderResponseError(FatalProviderError):
    """The provider returned a response that could not be parsed."""
