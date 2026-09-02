# coding-agent

`coding-agent` 是一个在本地工作区运行的 Coding Agent。它可以读取文件和文本型 PDF/DOCX，按需创建或精确修改文件，并在用户逐次审批后执行本地命令；连续聊天可选择保存在本地会话存档中。

## 当前能力

- 在工作区内列出、读取 UTF-8 文本文件。
- 读取文本型 PDF/DOCX 并返回受资源预算限制的结构化结果。
- 经逐次审批创建或精确修改文件。
- 经逐次审批运行本地命令，并返回退出码、输出和截断状态。
- 在内存或本地 JSON 存档中保留连续聊天历史，支持选择、恢复和重命名。

## 快速开始（Windows PowerShell）

环境要求：Python 3.12、[uv](https://docs.astral.sh/uv/) 和 DeepSeek API Key。

```powershell
git clone --branch develop https://github.com/IceBear-Kay/coding-agent.git
Set-Location .\coding-agent
uv sync --frozen
```

当前可运行代码位于 `develop`；仓库默认分支 `main` 暂不包含运行入口。下面创建一个普通示例工作区，不使用私人 `.local/`：

```powershell
$demo = Join-Path (Get-Location) 'coding-agent-demo'
New-Item -ItemType Directory -Force -Path $demo | Out-Null
$notes = Join-Path $demo 'notes.md'
if (-not (Test-Path -LiteralPath $notes)) {
@'
# Demo notes

这是一个用于验证读取能力的示例文件。
'@ | Set-Content -LiteralPath $notes -Encoding utf8
}
```

### 配置 DeepSeek

复制配置模板时先检查目标；已有 `.env` 会保留，不读取或覆盖：

```powershell
if (-not (Test-Path -LiteralPath .env)) {
    Copy-Item -LiteralPath .env.example -Destination .env
}
```

在本地编辑未被 Git 跟踪的 `.env`，填写以下变量：

```text
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_TIMEOUT_SECONDS=60
```

运行时显式加载该文件：

```powershell
uv run --env-file .env coding-agent --help
```

如果不想把密钥写入文件，可在同一 PowerShell 会话中使用隐藏输入设置 `$env:DEEPSEEK_API_KEY`，再设置其余 `$env:DEEPSEEK_*` 变量。它们只对当前会话及其子进程有效；Flash/Pro 通过修改 `DEEPSEEK_MODEL` 切换。不要使用 `setx` 保存密钥，也不要把密钥写入命令历史、Issue、PR 或截图。

需要检查配置时，可使用只输出安全错误的脚本；`--help` 只检查帮助入口，不验证配置：

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

## 常用操作

只读总结文件：

```powershell
uv run --env-file .env coding-agent '读取 notes.md 并用中文总结。' --workspace $demo --read-only --max-steps 8
```

创建或修改文件（每次操作都会显示预览并等待批准）：

```powershell
uv run --env-file .env coding-agent '创建 greeting.py，输出 Hello。' --workspace $demo --allow-write --max-steps 8
```

生成程序并运行验证（写入和执行分别审批）：

```powershell
uv run --env-file .env coding-agent '创建 solution.py，读取两个整数并输出它们的和，然后用输入 7 5 验证。' --workspace $demo --allow-write --allow-exec --max-steps 8 --max-retries 0
```

连续聊天及持久会话：

```powershell
uv run --env-file .env coding-agent --chat --workspace $demo
uv run --env-file .env coding-agent --chat --session demo-chat --session-dir .local/sessions --workspace $demo
uv run --env-file .env coding-agent --chat --resume demo-chat --session-dir .local/sessions --workspace $demo
```

省略位置任务时默认进入聊天；使用 `--no-chat` 可只读取一次输入，使用 `--no-save` 可关闭持久存档。聊天命令包括 `/help`、`/sessions`、`/rename <标题>`、`/clear` 和 `/exit`。

## 参数速查

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--workspace`, `-w` | `.` | 工作区目录；文件工具只能访问其范围内的路径。 |
| `--allow-write` / `--no-write` | 开启 | 开放或关闭 `write_file`、`edit_file`；每次仍需审批。 |
| `--allow-exec` / `--no-exec` | 开启 | 开放或关闭 `run_command`；每次仍需审批。 |
| `--read-only` | 关闭 | 同时关闭写入和命令执行。 |
| `--max-steps` | `64` | 每个任务的 Provider 调用次数上限，重试也计入。 |
| `--max-retries` | `2` | 临时 Provider 错误的重试次数；`0` 表示不重试。 |
| `--command-timeout` | `20` 秒 | 本地命令超时，必须大于 0 且不超过 60 秒。 |
| `--command-output-limit` | `65536` 字节 | 本地命令 stdout/stderr 共享上限。 |
| `--max-context-bytes` | `8388608` | 每次请求的上下文 UTF-8 字节预算。 |
| `--max-context-tokens` | `524288` | 输入 token 的本地粗估预算。 |
| `--max-output-tokens` | `32768` | 每次模型请求的输出预算。 |
| `--context-policy` | `trim` | 上下文超限时按完整旧任务裁剪，或用 `stop` 立即停止。 |
| `--show-plan` / `--no-show-plan` | 显示 | 控制伴随工具调用的简短行动说明。 |
| `--show-tool-events` / `--hide-tool-events` | 显示 | 控制正常工具进度提示；审批、错误和最终答案仍显示。 |
| `--show-stats` | 关闭 | 显示本次任务的耗时、请求、工具和用量统计。 |

`--max-steps` 限制模型请求次数，不限制一次响应中的工具调用数；`--command-timeout` 只限制本地命令，不影响 API 超时。完整参数和默认值以 `uv run coding-agent --help` 及 [`docs/cli.md`](docs/cli.md) 为准。

## 审批与运行边界

- 写入、修改和本地命令都按调用逐次审批；交互终端输入 `y` 或 `yes` 才批准，回车、其他输入、EOF、管道输入都会拒绝。`Ctrl+C` 返回退出码 `130`。
- 真实终端使用 Rich 显示 Markdown 和等待状态；等待状态会在审批或其他用户输入前停止。非 TTY、重定向或显示失败时安全降级为纯文本，终端控制字符不会被当作格式化指令。
- 文件工具拒绝越界路径、符号链接、junction/reparse point、`.git`、`.local`、`.venv` 和真实 `.env`，并且不提供删除文件操作。
- 本地命令以当前用户权限运行。路径检查、审批、资源预算和进程清理不是操作系统级沙箱，已批准的程序仍可能访问该用户有权访问的其他资源。
- 持久会话保存完整历史，不保存 API Key 或运行时配置；恢复后的工具结果可能过时，涉及当前文件时应重新读取核验。当前不支持跨设备同步、摘要压缩或中断续跑。

自动测试使用 `FakeProvider`、模拟 HTTP 和临时目录，不调用真实 DeepSeek API。更多工具协议、文档读取、会话和安全限制见 [`docs/cli.md`](docs/cli.md) 与 [`docs/architecture.md`](docs/architecture.md)。
