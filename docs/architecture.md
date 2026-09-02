# Architecture

## Goal

本项目从零实现一个本地 Coding Agent：大语言模型决定下一步行动，本地 Python 程序负责保存状态、执行工具、反馈结果并控制循环。

## Component boundaries

```text
CLI
  -> Agent Loop
       -> AgentState / Context Policy
       -> AgentSession / SessionStore
       -> ModelProvider
       -> Tool Registry / Dispatcher
            -> Workspace Files
            -> Document Parser (PDF/DOCX)
            -> Local Commands
```

- **CLI**：接收任务、工作区和运行参数，按需显示工具过程并显示最终结果。
- **Agent Loop**：连接模型与工具，追加消息并决定何时结束。
- **AgentState**：保存消息、步骤、工作区、停止原因和上下文裁剪统计；每次任务还记录独立的运行统计。
- **ModelProvider**：屏蔽模型厂商差异，输出统一 `ModelResponse`。
- **Tool Registry/Dispatcher**：暴露工具 Schema、验证参数并调度本地实现。
- **Document Parser**：在工作区边界内有界读取文本型 PDF 和简单 DOCX；打开文件前后复核文件描述符与父目录身份，解析在可终止的受控进程中进行，结果以有界结构化只读工具消息返回。
- **Context Policy**：按 UTF-8 字节预算和输入 token 粗估检查请求上下文；默认 `trim` 时按完整旧任务从最早开始裁剪，`stop` 在超限时停止，当前不提供摘要压缩。
- **Runtime budgets**：请求前同时检查字节预算与基于序列化 UTF-8 大小的输入 token 粗估，并为 `max_tokens` 输出额度预留模型窗口安全余量；粗估不等同于厂商 tokenizer。
- **AgentSession / SessionStore**：在聊天任务正常完成后提交完整历史；可选 JSON 存档支持跨进程恢复，使用会话生命周期独占锁、原子写入并校验工作区和历史结构。

CLI 通过轻量执行事件显示实际的工具调用和结构化结果摘要，包括文件相对路径、命令
退出状态、超时或截断信息；它不展示认证信息、完整 API 请求或模型的
`reasoning_content`。正常结束只表示模型不再请求工具，不等于任务已经通过独立测试。

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

工具返回的成功、失败、拒绝、超时和输出超限结果都会原样作为下一轮 `tool` 消息提供给
模型。临时 Provider 错误可在有限预算内重试，但不会重放上一轮已经完成的副作用操作；
`length`、`content_filter` 和 `insufficient_system_resource` 等非正常停止原因会保留并
立即结束本次循环，不执行响应中携带的工具调用。

Provider 将内部消息转换为 OpenAI-compatible 请求格式：工具参数编码为 JSON 字符串，
并在携带工具的后续请求中保留 Assistant 的 `reasoning_content`。

## Delivery stages

- M1：治理规范、Python/uv 骨架和 CI。
- M2：DeepSeek Provider、只读工具、Agent Loop 和 CLI。
- M3：经审批的写文件、精确编辑、本地命令执行和端到端编程任务。
- M5-B：完成聊天会话持久化与恢复、存档边界保护和会话级测试。
- M5-D：增加有界、只读的 PDF/DOCX 文本提取，并保留现有工具协议、路径保护和上下文预算。
- 后续：上下文压缩和更强的执行隔离。

只读 M2 用于尽早验证完整数据流，但最终 Coding Agent 必须具备本地写入、修改和命令执行能力。

## v0.1.0 当前能力与限制

- CLI 可以接收一次任务和工作区；省略位置任务时默认进入聊天，带位置任务时执行一次。写入、修改和本地命令工具默认开放，但每次副作用操作都需要逐次审批；可用 `--no-write`、`--no-exec` 或 `--read-only` 关闭对应入口。
- 只读工具为 `list_files`、`read_file` 和 `read_document`；目录扫描、文件输出和文档解析均有预算。
- `read_document` 使用 `pypdf` 提取 PDF 页面文字，使用 `python-docx` 按正文顺序提取 DOCX 段落和简单表格。源文件最多 5 MiB，PDF 单次最多 20 页，文本最多 32000 个字符，解析结果最多 256 KiB；损坏、加密、选定范围无文字、路径替换、不支持格式、路径越界和资源超限会返回结构化错误，工具不会修改原文档。
- 文档解析不提供 OCR、图片理解、旧版 `.doc`、`.docm`、宏执行或复杂版式/公式还原；受控解析进程和资源限制不构成操作系统级沙箱。
- `write_file` 只排他创建新 UTF-8 文件，`edit_file` 只执行一次精确文本替换。两者每次调用都展示完整预览并等待用户确认，批准后还会复核目标状态。
- `run_command` 接收结构化 argv、工作区内 cwd 和独立 stdin，以 `shell=False` 启动进程；stdout/stderr 共享有界读取预算，并返回退出码、耗时、截断和执行状态。超时、输出超限和中断会触发普通进程树清理。
- CLI 的 `run_command` 默认超时为 20 秒，可用参数覆盖；正常工具调用和结果摘要默认显示，也可单独隐藏而不改变内部事件或消息历史。
- Agent Loop 会保留完整对话历史、Assistant Tool Calls、`reasoning_content` 和原始 `tool_call_id`，并区分正常完成、`max_steps`、`interrupted`、Provider 错误及非正常 `finish_reason`。
- `--show-stats` 展示本次任务的 Provider 请求、工具调度、工具错误、服务端 `usage` Token、上下文字节数和耗时。统计只存在于 `AgentState`/`AgentRunResult`，不进入消息历史；缺失 `usage` 的请求单独标记为未知。
- 端到端离线测试使用 `FakeProvider` 驱动真实文件和受控本地进程，覆盖读取题目、批准写入、运行失败、精确编辑、再次运行成功以及拒绝和超时路径；OpenAI-compatible 协议使用 `httpx.MockTransport` 验证，不调用真实模型。
- 当前不删除文件，不提供自动审批、上下文摘要压缩或中断续跑。持久聊天可保存和恢复正常完成任务的完整历史，但不保存 Provider 配置、环境变量、API Key、审批状态或进程对象；本地命令以当前用户权限运行，路径检查、最小环境、资源预算和审批不等同于操作系统沙箱。CLI 从启动目录读取 `.env`，进程环境变量优先；未知模型不会静默套用已知窗口能力。

## Safety

- 所有路径解析后必须位于指定工作区。
- 工具参数必须通过 Schema/模型验证。
- 文件创建和修改在 CLI 中默认开放；仍需逐次审批，且审批后复核目标身份与内容。库级工具注册表默认仍为只读。
- 本地命令在 CLI 中默认开放；使用结构化参数和逐次审批，不经过隐式 Shell 解释，审批后复核工作目录与命令计划。库级工具注册表默认仍关闭执行。
- 文件内容和命令输出设置大小上限。
- 命令执行设置超时并保留退出码、stdout 和 stderr；Windows Job Object 与 Linux 进程组负责普通进程树清理。
- API Key 只从环境变量或未入库配置读取。
- 工具错误结构化返回；认证等不可恢复错误终止循环。

## Prohibited dependencies

不得使用 Agent 框架或 Agent SDK，例如 LangChain、LlamaIndex、OpenAI Agents SDK、Claude Agent SDK、AutoGen、CrewAI；不得依赖 API 服务端托管的文件或代码执行工具。

普通 HTTP、数据校验、测试、模型厂商 API 客户端等基础库不等于 Agent 框架，但新增依赖仍应在 PR 中说明用途和取舍。
