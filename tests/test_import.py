def test_package_imports() -> None:
    import coding_agent

    assert coding_agent.__version__ == "0.1.0"


def test_package_exports_core_public_types() -> None:
    import coding_agent
    from coding_agent import (
        DEFAULT_COMMAND_OUTPUT_LIMIT_BYTES,
        DEFAULT_COMMAND_TIMEOUT_SECONDS,
        DEFAULT_MAX_FILE_BYTES,
        AgentState,
        ApprovalCallback,
        ApprovalRequest,
        CommandLimits,
        EditFileArguments,
        FakeProvider,
        LocalCommandRunner,
        Message,
        ModelProvider,
        ModelResponse,
        OpenAICompatibleProvider,
        ProviderConfig,
        RunCommandArguments,
        ToolCall,
        ToolOutput,
        ToolResult,
        Usage,
        WriteFileArguments,
        create_workspace_registry,
        edit_file_tool_spec,
        request_approval,
        run_command_tool_spec,
        write_file_tool_spec,
    )

    assert coding_agent.AgentState is AgentState
    assert coding_agent.ApprovalCallback is ApprovalCallback
    assert coding_agent.ApprovalRequest is ApprovalRequest
    assert coding_agent.CommandLimits is CommandLimits
    assert coding_agent.DEFAULT_COMMAND_OUTPUT_LIMIT_BYTES is DEFAULT_COMMAND_OUTPUT_LIMIT_BYTES
    assert coding_agent.DEFAULT_COMMAND_TIMEOUT_SECONDS is DEFAULT_COMMAND_TIMEOUT_SECONDS
    assert coding_agent.DEFAULT_MAX_FILE_BYTES is DEFAULT_MAX_FILE_BYTES
    assert coding_agent.EditFileArguments is EditFileArguments
    assert coding_agent.FakeProvider is FakeProvider
    assert coding_agent.Message is Message
    assert coding_agent.LocalCommandRunner is LocalCommandRunner
    assert coding_agent.ModelProvider is ModelProvider
    assert coding_agent.ModelResponse is ModelResponse
    assert coding_agent.OpenAICompatibleProvider is OpenAICompatibleProvider
    assert coding_agent.ProviderConfig is ProviderConfig
    assert coding_agent.RunCommandArguments is RunCommandArguments
    assert coding_agent.ToolCall is ToolCall
    assert coding_agent.ToolOutput is ToolOutput
    assert coding_agent.ToolResult is ToolResult
    assert coding_agent.Usage is Usage
    assert coding_agent.WriteFileArguments is WriteFileArguments
    assert coding_agent.create_workspace_registry is create_workspace_registry
    assert coding_agent.edit_file_tool_spec is edit_file_tool_spec
    assert coding_agent.request_approval is request_approval
    assert coding_agent.run_command_tool_spec is run_command_tool_spec
    assert coding_agent.write_file_tool_spec is write_file_tool_spec
