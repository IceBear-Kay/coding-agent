# coding-agent

`coding-agent` 是一个在本地工作区运行的 Coding Agent：模型读取任务和文件，按需请求本地工具，并根据真实工具结果继续处理。当前支持读取 UTF-8 文本文件、经用户批准创建和精确修改文件、经用户批准执行本地命令，以及内存或磁盘持久化的连续任务会话。

## 环境要求

- Windows PowerShell（以下命令以 PowerShell 为准）。
- Python 3.12。
- [uv](https://docs.astral.sh/uv/)。
- DeepSeek API Key。真实模型请求会产生 API 费用。

## 快速启动

### 1. 获取代码并同步环境

```powershell
git clone --branch develop https://github.com/IceBear-Kay/coding-agent.git
Set-Location .\coding-agent
uv sync --frozen
uv run python --version
uv --version
```

仓库默认分支 `main` 当前尚未包含可运行代码，因此首次克隆时显式检出 `develop`；这不会修改 GitHub 的默认分支。`uv sync --frozen` 按已有 `uv.lock` 同步依赖，不重新解析或修改锁文件。Python 版本应为 `3.12.x`。

### 2. 准备一个普通示例工作区

下面的命令在仓库内创建 `coding-agent-demo`，不会使用仓库的私人 `.local/` 目录；已有的 `notes.md` 不会被覆盖。

```powershell
$demo = Join-Path (Get-Location) 'coding-agent-demo'
New-Item -ItemType Directory -Force -Path $demo | Out-Null
$notes = Join-Path $demo 'notes.md'
if (-not (Test-Path -LiteralPath $notes)) {
@'
# Demo notes

这是一个用于验证 coding-agent 读取能力的示例文件。
'@ | Set-Content -LiteralPath $notes -Encoding utf8
}
```

### 3. 配置 DeepSeek

项目从当前进程环境读取以下配置：

- `DEEPSEEK_API_KEY`：必填的 API Key。
- `DEEPSEEK_BASE_URL`：必填的 HTTP API 地址，模板默认为 `https://api.deepseek.com`。
- `DEEPSEEK_MODEL`：必填的模型标识。
- `DEEPSEEK_TIMEOUT_SECONDS`：必填的 API 请求超时秒数，必须大于 0；模板使用 `60`。

仓库提供不含凭据的 `.env.example`。程序不会自动搜索或读取 `.env`；需要使用配置文件时，必须通过 `uv run --env-file .env` 显式加载。下面的命令只在目标不存在时复制模板，已有 `.env` 会原样保留且不会被读取或覆盖：

```powershell
if (Test-Path -LiteralPath .env) {
    Write-Host '已存在 .env，保留现有文件。'
}
else {
    Copy-Item -LiteralPath .env.example -Destination .env
}
```

随后在本地编辑未跟踪的 `.env`，填写真实 Key。该文件已被 Git 忽略，不能将凭据写入任何受 Git 跟踪的文件。

如果不希望把 Key 写入文件，可在当前 PowerShell 中用隐藏输入设置它，再在同一会话设置其余配置：

```powershell
$secureKey = Read-Host '请输入 DeepSeek API Key（输入时不会显示）' -AsSecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $env:DEEPSEEK_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    $secureKey.Dispose()
}

$env:DEEPSEEK_BASE_URL = 'https://api.deepseek.com'
$env:DEEPSEEK_MODEL = 'deepseek-v4-flash'
$env:DEEPSEEK_TIMEOUT_SECONDS = '60'
```

使用 `.env` 时，通过下面的安全检查确认配置是否完整。脚本会捕获 `ProviderConfigurationError`，只输出不含配置值的错误说明，并以状态码 `1` 退出，不打印异常链：

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

这些 `$env:` 变量只在当前 PowerShell 会话及其启动的子进程中有效；换用新终端后需要重新设置。更换 Flash/Pro 只需修改 `DEEPSEEK_MODEL`，具体标识以 DeepSeek 账户实际可用的模型为准：

```powershell
$env:DEEPSEEK_MODEL = 'deepseek-v4-pro'
```

通过 `uv --env-file` 加载配置文件时，当前 PowerShell 中已经存在的同名环境变量优先于文件值。不要使用 `setx` 保存真实 Key，也不要把 Key 写入命令历史、Issue、PR 或截图。

### 4. 启动 Agent

推荐使用本地 `.env`；后续示例统一采用这种显式加载方式：

```powershell
uv run --env-file .env coding-agent '读取 notes.md 并用中文总结文件内容。' --workspace $demo --max-steps 8
```

如果上一节选择了当前 PowerShell 环境变量方式，则不需要加载文件：

```powershell
uv run coding-agent '读取 notes.md 并用中文总结文件内容。' --workspace $demo --max-steps 8
```

CLI 默认开放 `list_files`、`read_file`、`write_file`、`edit_file` 和 `run_command`。写入、修改和执行仍必须由你逐次审批；可用 `--no-write`、`--no-exec` 或 `--read-only` 明确关闭对应工具。

## 常用操作

只读文件：

```powershell
uv run --env-file .env coding-agent '读取 notes.md，列出其中的主要信息。' --workspace $demo
```

创建或精确修改文件。每一次 `write_file` 或 `edit_file` 都会显示预览并等待确认：

```powershell
uv run --env-file .env coding-agent '创建 greeting.py，输出 Hello。' --workspace $demo --allow-write --max-steps 8
```

生成 Python 程序并运行验证。文件写入和命令执行是两类独立的逐次审批：

```powershell
uv run --env-file .env coding-agent '创建 solution.py，读取两个整数并输出它们的和，然后用输入 7 5 运行验证。' --workspace $demo --allow-write --allow-exec --command-timeout 20 --command-output-limit 65536 --max-steps 8 --max-retries 0
```

省略任务参数后，Agent 默认进入连续聊天；如需只读取一次任务，使用 `--no-chat`：

```powershell
uv run --env-file .env coding-agent --no-chat --workspace $demo
```

也可以显式使用 `--chat` 进入连续聊天，且不要同时提供位置任务：

```powershell
uv run --env-file .env coding-agent --chat --workspace $demo --max-steps 8
```

需要跨进程保留已完成历史时，为聊天指定会话 ID：

```powershell
uv run --env-file .env coding-agent --chat --session demo-chat --session-dir .local/sessions --workspace $demo
```

下次启动使用同一存档目录和 `--resume` 恢复；恢复只加载历史，不会重放历史模型请求或工具调用：

```powershell
uv run --env-file .env coding-agent --chat --resume demo-chat --session-dir .local/sessions --workspace $demo
```

持久聊天启动时会显示会话 ID 和存档路径，不展示完整历史。输入 `/clear` 会保留旧存档并切换到新的空会话 ID；输入 `/exit` 退出。只有正常完成的任务才会写入存档，异常停止或中断任务不会作为可恢复的完整任务保存。

恢复已有会话后，历史中的工具结果可能已经过时；继续任务前应重新读取并核验当前文件。

同一会话 ID 在一个持久会话使用期间由单个进程独占；其他进程会在模型请求和工具执行前收到占用错误。正常退出、异常退出和 `Ctrl+C` 都会释放本进程持有的锁；来源不明的锁文件不会自动删除。

每次输入的非空文本都会启动一个新任务，并继承此前正常完成任务的内存历史。输入 `/clear` 清空内存历史，输入 `/exit` 正常退出；空输入继续等待，等待输入时收到 EOF 也会正常退出。正常工具调用和结果默认显示，可用 `--hide-tool-events` 隐藏；审批、错误、重要警告、最终回答和停止原因始终显示。

持久聊天可使用 `--session ID` 新建或 `--resume ID` 恢复，并可用 `--session-dir PATH` 指定存档目录；省略 `--session-dir` 时默认使用启动目录下的 `.local/sessions`。`--session` 与 `--resume` 互斥，且必须处于聊天模式（显式使用 `--chat` 或省略位置任务进入默认聊天）。普通 `--chat` 不会创建存档。持久会话保存完整历史，不保存裁剪后的请求上下文、Provider 配置、环境变量、API Key、审批状态或进程对象；存档目录受到工具访问保护，不会被 Agent 文件工具读取或写入。

查看当前安装版本的完整参数说明：

```powershell
uv run coding-agent --help
```

`--help` 只验证 CLI 帮助入口并列出参数，不创建 Provider，因此不验证 `.env` 或 DeepSeek 配置。

更完整的工具协议、安全边界和可选真实 DeepSeek 验证说明见 [`docs/cli.md`](docs/cli.md) 与 [`docs/architecture.md`](docs/architecture.md)。

## 参数速查

| 参数 | 默认值 | 作用与限制 |
| --- | --- | --- |
| `task` | 省略时进入聊天 | 提供后执行一次任务；不能与 `--chat` 同时使用。 |
| `--chat` / `--no-chat` | 按是否省略 task 推导 | 显式开启或关闭连续聊天；两者互斥。 |
| `--session ID` | 未启用 | 新建指定 ID 的持久聊天会话；需处于聊天模式（可省略 `--chat`），不能与位置任务同时使用。 |
| `--resume ID` | 未启用 | 恢复指定 ID 的持久聊天会话；需处于聊天模式（可省略 `--chat`），不能与位置任务同时使用。 |
| `--session-dir PATH` | 启动目录下 `.local/sessions`（仅持久模式） | 指定会话 JSON 存档目录；必须与 `--session` 或 `--resume` 一起使用。 |
| `--workspace`, `-w` | `.` | 工作区目录，必须是已有目录；文件工具只能访问工作区内路径。 |
| `--allow-write` / `--no-write` | 默认开启 | 开放或关闭 `write_file` 和 `edit_file`；每个操作仍需审批。 |
| `--allow-exec` / `--no-exec` | 默认开启 | 开放或关闭 `run_command`；每个命令仍需审批。 |
| `--read-only` | 关闭 | 同时关闭写入和执行工具；不能与显式 `--allow-write` 或 `--allow-exec` 同时使用。 |
| `--show-tool-events` / `--hide-tool-events` | 默认显示 | 控制正常工具过程提示；隐藏时审批、错误和最终回答仍显示。 |
| `--command-timeout` | `20.0` 秒 | 本地命令超时；必须是有限数值，且大于 0、不超过 60 秒。 |
| `--command-output-limit` | `65536` 字节 | 本地命令 `stdout` 与 `stderr` 共享上限；范围为 1 至 1048576。 |
| `--max-steps` | `8` | 每个任务的 Provider 调用次数上限，必须为正整数；临时错误重试也计入。不是工具调用次数上限。 |
| `--max-retries` | `2` | 每个任务中单次临时 Provider 错误的最大重试次数，必须为非负整数；设为 `0` 关闭自动重试。 |
| `--max-context-bytes` | `262144` | 每次 Provider 请求前的上下文 UTF-8 字节预算，必须为正整数；超限时按 `--context-policy` 处理。 |
| `--context-policy` | `stop` | 上下文超预算时的策略；`stop` 立即停止，`trim` 按完整旧任务从最早开始裁剪。 |

`--max-steps` 统计模型请求次数；一次模型响应可以包含多个工具调用，这些调用按模型给出的顺序处理。在 `--chat` 模式中，每个新任务都会重新计算 `--max-steps` 和重试预算。写入和执行工具仍按调用逐次审批。`DEEPSEEK_TIMEOUT_SECONDS` 控制 API 网络请求，与 `--command-timeout` 控制的本地子进程超时无关。

`--max-context-bytes` 是软件级输入预算，按内部消息、工具参数、工具结果、`reasoning_content` 和工具 Schema 的紧凑 JSON UTF-8 字节数统计。它不是模型 Token 数、API 请求精确字节数或模型上下文窗口保证。预算超限时，默认 `--context-policy stop` 返回 `context_limit` 且不发送请求；显式使用 `--context-policy trim` 时，才会按完整旧任务从最早开始裁剪，仍无法满足预算则返回 `context_limit`。两种策略都不会摘要或重试该请求。在 `--chat` 中，正常完成任务的历史会继续计入预算；`/clear` 后只计算新历史。

## 审批与限制

- 写入和本地命令执行都必须逐次审批。输入 `y` 或 `yes` 才批准；直接回车、其他输入、EOF 或非交互管道输入都会拒绝。`--chat` 中前一任务的批准不会自动批准后续操作。
- `Ctrl+C` 中断当前任务并返回退出码 `130`；在 `--chat` 模式中会同时结束整个会话。已经完成的工具结果会保留在本次任务状态中，已产生的文件变化不会回滚，未执行的后续工具不会继续执行。
- 正常结束只表示模型不再请求工具，不等于生成的程序通过了独立测试；命令的退出码 `0` 也不自动证明答案正确。
- `--chat` 默认只在当前进程内保存正常完成任务的历史；使用 `--session` 或 `--resume` 可将完整历史保存到 JSON 存档并跨进程恢复。`max_steps`、Provider 错误和非正常模型停止原因会结束整个会话，不继续使用不完整历史。
- `/clear` 在普通聊天中只清空内存消息；在持久聊天中保留旧存档并切换到新的空会话 ID。两种模式都不删除工作区文件，也不改变工具开关和运行参数。
- 会话存档不支持中断续跑、跨设备同步、数据库和自动迁移；当前仍不提供上下文摘要压缩，历史裁剪仅在显式设置 `--context-policy trim` 时按完整旧任务执行，默认策略为 `stop`。
- 会话文件属于私人数据，应放在未被 Git 跟踪的目录中；存档目录不允许 Agent 文件工具直接访问。
- 文件工具只处理受工作区约束的 UTF-8 文本，不提供删除文件操作。
- 本地命令以当前用户权限运行。路径检查、审批、资源预算和进程清理用于降低常见误操作风险，不是操作系统级沙箱，也不能阻止被批准程序主动访问其他资源。
- 自动测试使用 `FakeProvider` 和模拟 HTTP，不调用真实 DeepSeek API。真实 API 手工验证请先确认费用和审批范围。
