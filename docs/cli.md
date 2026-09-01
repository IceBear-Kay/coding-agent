# CLI 使用说明

`coding-agent` 带位置任务时执行一次任务；省略位置任务时默认进入连续聊天。聊天会在同一进程内保留正常完成的会话历史。CLI 默认开放 `list_files`、`read_file`、`write_file`、`edit_file` 和 `run_command`，但写入、修改和执行仍需逐次审批；可用 `--no-write`、`--no-exec` 或 `--read-only` 明确关闭对应工具。

## 直接运行

以下任务示例假定已按“Provider 配置”一节准备本地 `.env`，并在仓库根目录通过 `uv` 显式加载：

```powershell
uv run --env-file .env coding-agent "请读取 docs/architecture.md 并总结项目架构" --workspace . --max-steps 8
```

也可以省略任务参数，程序会默认进入连续聊天：

```powershell
uv run --env-file .env coding-agent --workspace .
```

如需省略位置任务但只运行一次，可使用 `--no-chat`：

```powershell
uv run --env-file .env coding-agent --no-chat --workspace .
```

也可以显式使用 `--chat`，且不能同时提供位置任务：

```powershell
uv run --env-file .env coding-agent --chat --workspace . --max-steps 8
```

进入会话后，普通非空文本会启动一个新任务；`/clear` 清空内存历史，`/exit` 正常退出。空输入不调用 Provider，等待任务时收到 EOF 也会正常退出。

## 持久聊天会话

使用 `--session ID` 创建持久会话，使用 `--resume ID` 恢复已有会话；两者互斥，且必须处于聊天模式（显式使用 `--chat` 或省略位置任务进入默认聊天）。`--session-dir PATH` 指定 JSON 存档目录，省略时默认为启动目录下的 `.local/sessions`。位置任务不能与持久会话参数同时使用，单独指定 `--session-dir` 会报参数错误。

```powershell
uv run --env-file .env coding-agent --chat --session demo-chat --session-dir .local/sessions --workspace .
uv run --env-file .env coding-agent --chat --resume demo-chat --session-dir .local/sessions --workspace .
```

启动时只显示会话 ID 和存档路径，不展示完整历史。恢复会话只加载已完成的历史，不重放 Provider 请求或历史工具调用；当前启动参数决定本次工具权限、审批行为和资源限制。只有停止原因为 `completed` 的任务才会写入存档，异常停止或中断任务不会提交为可恢复历史。

恢复后 CLI 会提示历史工具结果可能过时；继续任务前应重新读取并核验当前文件。

同一会话 ID 在持久会话使用期间由单个进程独占；其他进程会在恢复阶段被拒绝，不会调用模型或工具。正常退出、异常退出和 `Ctrl+C` 会释放本进程持有的锁，来源不明的 `.lock` 文件不会自动删除。

持久模式下输入 `/clear` 会保留旧 JSON 存档，创建新的空会话 ID 并切换；普通聊天的 `/clear` 仍只清空内存历史。存档保存完整消息历史，不保存裁剪后的请求上下文、API Key、环境变量、Provider 配置、审批状态或进程对象。存档目录不允许 Agent 文件工具直接访问，也不支持中断续跑、跨设备同步、数据库和自动迁移。

## 参数速查

- `task`：一次任务的文本；提供后执行一次，省略时默认进入聊天。
- `--chat` / `--no-chat`：显式开启或关闭连续任务会话；两者互斥，`--chat` 不能与位置任务同时使用。
- `--session ID`：创建指定 ID 的持久聊天会话；需处于聊天模式（可省略 `--chat`），不能与位置任务同时使用。
- `--resume ID`：恢复指定 ID 的持久聊天会话；需处于聊天模式（可省略 `--chat`），不能与位置任务同时使用。
- `--session-dir PATH`：持久会话 JSON 存档目录；默认是启动目录下 `.local/sessions`，必须与 `--session` 或 `--resume` 一起使用。
- `--workspace` / `-w`：工作区目录，默认是当前目录。
- `--allow-write` / `--no-write`：默认开放或关闭 `write_file` 和 `edit_file`；每个副作用操作仍需单独审批。
- `--allow-exec` / `--no-exec`：默认开放或关闭 `run_command`；每个命令仍需单独审批。
- `--read-only`：同时关闭写入和执行工具；不能与显式 `--allow-write` 或 `--allow-exec` 同时使用。
- `--show-tool-events` / `--hide-tool-events`：默认显示或隐藏正常工具调用和结果提示；隐藏时审批、错误、重要警告、最终回答和停止原因仍显示。
- `--show-stats`：默认关闭；任务结束时显示本次运行的耗时、Provider 请求、工具调度、上下文和服务端 Token 摘要，不改变模型行为。
- `--max-steps`：每个任务允许的 Provider 调用次数，默认 8；临时错误重试也计入预算。
- `--max-retries`：每个任务中单次临时 Provider 错误最多重试次数，默认 2；设为 0 可关闭自动重试。
- `--max-context-bytes`：每次 Provider 请求前的上下文 UTF-8 字节预算，默认 262144；必须为正整数，超限时按 `--context-policy` 处理。
- `--context-policy`：上下文超预算时的策略，默认 `stop`；`stop` 立即停止，`trim` 按完整旧任务从最早开始裁剪。
- `--command-timeout`：命令超时上限，默认 20 秒，最大 60 秒。
- `--command-output-limit`：stdout 和 stderr 共享的输出字节上限，默认 65536。

查看当前版本的完整参数说明：

```powershell
uv run coding-agent --help
```

`--help` 只检查 CLI 帮助入口，不创建 Provider，也不验证 `.env` 中的配置。

工具调用和结果的正常过程提示默认显示。使用 `--hide-tool-events` 可隐藏这些提示，但不会影响 Provider 消息、工具执行或历史；审批预览、审批结果、工具错误、重要警告、最终回答和停止原因始终保留。`--show-tool-events` 可显式恢复显示，两者互斥。

`--max-context-bytes` 使用内部消息、工具参数、工具结果、`reasoning_content` 和工具 Schema 的紧凑 JSON UTF-8 字节数作为统一口径。它是软件级输入预算，不是模型 Token 数、API 请求精确字节数或模型上下文窗口保证。每次实际 Provider 请求前都会检查预算；默认 `--context-policy stop` 超限时返回 `context_limit`，不发送请求；显式使用 `--context-policy trim` 时，按完整旧任务从最早开始裁剪，仍无法满足预算则返回 `context_limit`。裁剪不会摘要或重试请求。该预算独立于 `--max-steps` 的模型调用次数、`--max-retries` 的临时错误重试次数以及本地命令的 `--command-output-limit`。

## 运行统计

使用 `--show-stats` 可在每个任务结束时显示一行中文摘要。摘要中的 Provider 请求次数按真实调用计数并包含临时错误重试；工具调度次数按进入 Dispatcher 的次数计数，工具错误单独统计，不能理解为成功写入或命令运行次数。输入、输出和总 Token 只来自服务端返回的 `ModelResponse.usage`，无 usage 的响应或请求失败会显示未知请求数，不会用字节数估算。上下文显示最后一次请求前的实际 UTF-8 字节数与预算，并保留裁剪任务提示；统计不追加到消息历史，也不改变预算、审批、模型参数或工具结果。

## 经审批的文件修改

需要创建或精确修改 UTF-8 文本文件时，CLI 默认已开放对应工具，也可以显式增加 `--allow-write`；如需关闭写入，使用 `--no-write` 或 `--read-only`：

```powershell
uv run --env-file .env coding-agent "请创建 hello.py，输出 Hello, world!" --workspace . --allow-write --max-steps 8
```

模型提出每一次 `write_file` 或 `edit_file` 调用后，CLI 都会先显示目标路径、待创建目录以及完整新内容或精确差异，然后提示：

```text
批准本次操作？[y/N]:
```

只有在交互终端中输入 `y` 或 `yes` 才批准本次操作。空输入、其他回答、无法读取输入以及管道或重定向的非交互输入均视为拒绝；`Ctrl+C` 会中断整个任务并返回退出码 130。`--allow-write` 和 `--allow-exec` 只负责向模型开放对应工具，不会自动批准任何操作。

- `write_file` 只创建新文件，不覆盖任何已有文件或目录。
- `edit_file` 要求 `old_text` 非空且在原文件中恰好出现一次，重叠位置也计为多次；不执行模糊、正则或隐式多处替换。
- 文件必须是 UTF-8 文本，原内容和结果默认均不超过 65536 字节。
- `.git`、`.local`、`.venv`、真实 `.env`、路径逃逸、符号链接、junction/reparse point 和危险 Windows 特殊路径会被拒绝。
- 审批后会重新核对目标；文件被修改、删除或替换时返回冲突，不套用过期操作。

## 经审批的本地命令

需要运行生成的程序时，同时开放文件修改和本地命令工具：

```powershell
uv run --env-file .env coding-agent "请创建 solution.py，读取两个整数并输出它们的和，然后用输入 7 5 运行验证" --workspace . --allow-write --allow-exec --command-timeout 20 --command-output-limit 65536 --max-steps 8
```

模型调用 `run_command` 时提供结构化 `argv`、工作区内的 `cwd` 和独立 `stdin` 文本。程序不会把参数拼成 Shell 命令，也不会自动解释管道、重定向或 `&` 等字符。`python` 会解析为当前 uv 环境实际使用的 Python 3.12 解释器。审批界面会显示解析后的完整参数、工作目录、stdin、超时和输出上限；只有批准后才启动进程，批准后还会重新核对命令与工作目录。

子进程结束后，工具以 JSON 返回 `stdout`、`stderr`、`exit_code`、`status`、执行耗时和 `truncated`，Agent Loop 再把该工具结果交回模型。`completed` 表示进程以退出码 0 结束，不表示生成的算法已经通过其他测试；非零退出返回 `failed`，超时返回 `timeout`，stdout/stderr 合计超过上限返回 `output_limit`。

- `--command-timeout` 默认 20 秒，可设置为大于 0 且不超过 60 的有限数值。
- `--command-output-limit` 默认 65536 字节，可设置为 1 至 1048576；stdout 和 stderr 共享该预算，达到上限时从读取阶段停止进程并标记截断。
- stdin、stdout 和 stderr 与审批终端输入彼此独立；stdin 写完后关闭，空字符串表示立即发送 EOF。
- 子进程使用最小化环境，不继承 DeepSeek、AWS 等凭据；`cwd` 必须是工作区内不经过符号链接或 reparse point 的已有目录。
- 超时、输出超限或 `Ctrl+C` 时会清理普通进程树。Windows 使用 Job Object，Linux 使用独立进程组。

当前版本不删除文件，也不提供自动审批。经批准的本地程序仍以当前用户权限运行，可以读写该用户有权访问的位置；`cwd` 限制、参数校验和进程清理用于降低误操作风险，不构成操作系统沙箱，也不保证约束刻意脱离进程组或改变权限的恶意程序。

## 连续任务会话

`--chat` 使用一份内存中的权威消息历史。每个新任务创建独立的任务状态，因此 Provider 调用次数和临时错误重试预算都会从零开始；模型请求仍能看到此前正常完成任务的 `system`、`user`、`assistant` 和 `tool` 消息，包括原始 `tool_call_id` 与必要的 `reasoning_content`。系统消息不会在每个任务前重复添加。

正常完成任务的历史会继续计入 `--max-context-bytes`；默认策略下，如果累积历史或工具结果使下一次请求超限，任务以 `context_limit` 停止；使用 `--context-policy trim` 时，会先移除最早的完整任务，仍超限才停止。已完成的工具结果仍保留在本次状态中。`/clear` 清空内存历史后，后续任务只按新的消息和工具 Schema 检查预算。

只有停止原因为 `completed` 的任务会提交到会话历史。`max_steps`、Provider 错误、`length`、`content_filter`、`insufficient_system_resource` 或 `Ctrl+C` 会结束整个会话，不继续读取下一任务，也不会把不完整的工具调用序列用于后续请求。已经完成的工具操作及其真实结果不会回滚或伪造，尚未执行的工具不会继续执行。

`/clear` 清空内存历史，但保留工作区、Provider 配置、工具开关和命令资源限制，不删除任何文件。会话不会写入磁盘，退出进程后不能恢复；当前不提供摘要压缩、保存或中断续跑，历史裁剪仅由显式的 `--context-policy trim` 控制。

## Provider 配置

未通过测试注入 Provider 时，CLI 从环境变量创建 DeepSeek Provider。仓库提供不含凭据的 `.env.example`，程序不会自动搜索或读取 `.env`；需要使用配置文件时，必须由 `uv` 显式加载：

```powershell
if (Test-Path -LiteralPath .env) {
    Write-Host '已存在 .env，保留现有文件。'
}
else {
    Copy-Item -LiteralPath .env.example -Destination .env
}
```

上述复制步骤不会读取或覆盖已有 `.env`。填写本地配置后，可使用下面的脚本安全检查；配置错误只输出变量名和原因并返回非零状态，不打印异常链或配置值：

```powershell
@'
import sys

from coding_agent.config import ProviderConfig
from coding_agent.errors import ProviderConfigurationError

try:
    ProviderConfig.from_env()
except ProviderConfigurationError as error:
    print(f"配置错误: {error}", file=sys.stderr)
    raise SystemExit(1)

print("DeepSeek 配置完整")
'@ | uv run --env-file .env python -
```

也可以只在当前 PowerShell 会话中设置四项环境变量，再直接运行 `uv run coding-agent`。`DEEPSEEK_API_KEY` 不应写入命令历史、Issue、PR、截图或任何受 Git 跟踪的文件；如使用本地未跟踪的 `.env`，应先确认其仍被 `.gitignore` 忽略。需要隐藏输入时可使用 `Read-Host -AsSecureString` 后设置 `$env:DEEPSEEK_API_KEY`。这些环境变量只对当前会话及其子进程有效。

当 `uv --env-file` 加载配置文件时，当前 PowerShell 中已经存在的同名环境变量优先于文件值。`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` 和 `DEEPSEEK_TIMEOUT_SECONDS` 仍由 `ProviderConfig` 统一校验；缺失或无效时会指出变量名，不会输出配置值。

`.env.example` 中的配置示例为：`DEEPSEEK_BASE_URL=https://api.deepseek.com`、`DEEPSEEK_MODEL=deepseek-v4-flash`、`DEEPSEEK_TIMEOUT_SECONDS=60`，`DEEPSEEK_API_KEY` 保持为空。真实 API Key 只应通过未跟踪的 `.env` 或当前会话的隐藏输入提供。

`--max-steps` 限制一个任务的 Provider 调用次数，临时错误重试也计入该上限。`--max-retries` 设置该任务中每次临时错误最多重试次数；`--chat` 中的下一个任务会重新计算这两项预算。

CLI 会以非零退出码报告 `max_steps`、`interrupted`、Provider 错误和非正常模型停止原因；文件修改和本地命令均不会自动批准。工具执行错误会作为结构化结果交回模型，由模型决定是否说明、修正参数或结束任务，不会自动重试副作用命令。

## 可选的真实 DeepSeek 手工验证

自动测试始终使用 `FakeProvider`，不会调用付费 API。需要手工验证时，在未入库的本地环境中设置 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` 和 `DEEPSEEK_TIMEOUT_SECONDS`。真实 Key 不得写入受 Git 跟踪的文件、Shell 历史或截图；允许保存在已确认被忽略的本地 `.env` 中。

然后执行：

```powershell
uv run --env-file .env coding-agent "请读取 docs/architecture.md 并列出 Agent Loop 的停止条件" --workspace . --max-steps 6
```

预期结果是模型先请求 `read_file`，程序在本地读取该文件并将结果回传，随后模型输出最终回答；若模型直接回答，则不会产生工具消息。达到 `max_steps`、网络失败或按下 `Ctrl+C` 时，CLI 应显示对应停止原因并返回非零退出码。

真实写入或命令验证必须由用户明确决定，并增加对应的 `--allow-write` 或 `--allow-exec`。这会调用真实模型并产生 API 费用；在审批提示出现后，应先核对完整预览，再决定是否输入 `y`，不确认时直接按回车即可拒绝。自动测试使用 `FakeProvider`、无害的短 Python 进程和临时目录，不需要真实 API，也不会产生模型费用。
