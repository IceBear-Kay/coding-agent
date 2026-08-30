# Architecture

## Goal

本项目从零实现一个本地 Coding Agent：大语言模型决定下一步行动，本地 Python 程序负责保存状态、执行工具、反馈结果并控制循环。

## Component boundaries

```text
CLI
  -> Agent Loop
       -> AgentState / Context Policy
       -> ModelProvider
       -> Tool Registry / Dispatcher
            -> Workspace Files
            -> Local Commands
```

- **CLI**：接收任务、工作区和运行参数，显示最终结果。
- **Agent Loop**：连接模型与工具，追加消息并决定何时结束。
- **AgentState**：保存消息、步骤、工作区和停止原因。
- **ModelProvider**：屏蔽模型厂商差异，输出统一 `ModelResponse`。
- **Tool Registry/Dispatcher**：暴露工具 Schema、验证参数并调度本地实现。
- **Context Policy**：限制工具输出，后续负责裁剪和压缩历史。

消息支持 `system`、`user`、`assistant` 和 `tool` 四种角色。Assistant 消息可以携带
`reasoning_content` 与结构化 `ToolCall`；Tool 消息通过 `tool_call_id` 关联调用并保存工具结果。

## Initial technology choices

- Python 3.12 与 uv。
- DeepSeek 的 OpenAI-compatible API 作为首个 Provider。
- 模型名称通过环境变量切换，不进入 Agent 核心逻辑。
- 使用原生 Tool Calling，不使用文本标签模拟工具协议。
- 使用 Pydantic 定义内部模型和工具参数。
- 使用 httpx 直接发送模型请求，以便自行处理请求、解析和错误映射。

## Agent loop

1. 保存用户任务。
2. 将消息和工具 Schema 发送给 Provider。
3. 保存 Assistant 文本与 Tool Calls。
4. 没有 Tool Call 时，将文本作为最终回答并结束。
5. 验证并执行每个 Tool Call。
6. 使用原始 `tool_call_id` 追加 Tool Result。
7. 达到最大步骤或致命错误时停止，否则继续循环。

Provider 将内部消息转换为 OpenAI-compatible 请求格式：工具参数编码为 JSON 字符串，
并在携带工具的后续请求中保留 Assistant 的 `reasoning_content`。

## Delivery stages

- M1：治理规范、Python/uv 骨架和 CI。
- M2：DeepSeek Provider、只读工具、Agent Loop 和 CLI。
- 后续：写文件、编辑文件、执行命令、会话和上下文压缩。

只读 M2 用于尽早验证完整数据流，但最终 Coding Agent 必须具备本地写入、修改和命令执行能力。

## Safety

- 所有路径解析后必须位于指定工作区。
- 工具参数必须通过 Schema/模型验证。
- 文件内容和命令输出设置大小上限。
- 命令执行设置超时并保留退出码、stdout 和 stderr。
- API Key 只从环境变量或未入库配置读取。
- 工具错误结构化返回；认证等不可恢复错误终止循环。

## Prohibited dependencies

不得使用 Agent 框架或 Agent SDK，例如 LangChain、LlamaIndex、OpenAI Agents SDK、Claude Agent SDK、AutoGen、CrewAI；不得依赖 API 服务端托管的文件或代码执行工具。

普通 HTTP、数据校验、测试、模型厂商 API 客户端等基础库不等于 Agent 框架，但新增依赖仍应在 PR 中说明用途和取舍。
