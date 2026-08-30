def test_package_imports() -> None:
    import coding_agent

    assert coding_agent.__version__ == "0.1.0"


def test_package_exports_core_public_types() -> None:
    import coding_agent
    from coding_agent import (
        AgentState,
        FakeProvider,
        Message,
        ModelProvider,
        ModelResponse,
        OpenAICompatibleProvider,
        ProviderConfig,
        ToolCall,
        ToolResult,
        Usage,
    )

    assert coding_agent.AgentState is AgentState
    assert coding_agent.FakeProvider is FakeProvider
    assert coding_agent.Message is Message
    assert coding_agent.ModelProvider is ModelProvider
    assert coding_agent.ModelResponse is ModelResponse
    assert coding_agent.OpenAICompatibleProvider is OpenAICompatibleProvider
    assert coding_agent.ProviderConfig is ProviderConfig
    assert coding_agent.ToolCall is ToolCall
    assert coding_agent.ToolResult is ToolResult
    assert coding_agent.Usage is Usage
