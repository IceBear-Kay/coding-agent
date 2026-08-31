# CLI 使用说明

`coding-agent` 每次运行处理一个任务。默认只提供工作区内的 `list_files` 和 `read_file`；显式使用 `--allow-write` 后，模型才可以申请调用 `write_file` 和 `edit_file`，显式使用 `--allow-exec` 后才可以申请调用 `run_command`。

## 直接运行

以下任务示例假定已按“Provider 配置”一节准备本地 `.env`，并在仓库根目录通过 `uv` 显式加载：

```powershell
uv run --env-file .env coding-agent "请读取 docs/architecture.md 并总结项目架构" --workspace . --max-steps 8
```

也可以省略任务参数，程序会在终端提示输入：

```powershell
uv run --env-file .env coding-agent --workspace .
```

## 参数速查

- `task`：一次任务的文本；省略时从终端读取。
- `--workspace` / `-w`：工作区目录，默认是当前目录。
- `--max-steps`：本次运行允许的 Provider 调用次数，默认 8；临时错误重试也计入预算。
- `--max-retries`：单次临时 Provider 错误最多重试次数，默认 2；设为 0 可关闭自动重试。
- `--allow-write`：开放 `write_file` 和 `edit_file`，每个副作用操作仍需单独审批。
- `--allow-exec`：开放 `run_command`，每个命令仍需单独审批。
- `--command-timeout`：命令超时上限，默认 10 秒，最大 60 秒。
- `--command-output-limit`：stdout 和 stderr 共享的输出字节上限，默认 65536。

查看当前版本的完整参数说明：

```powershell
uv run coding-agent --help
```

`--help` 只检查 CLI 帮助入口，不创建 Provider，也不验证 `.env` 中的配置。

## 经审批的文件修改

需要创建或精确修改 UTF-8 文本文件时，显式增加 `--allow-write`：

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
uv run --env-file .env coding-agent "请创建 solution.py，读取两个整数并输出它们的和，然后用输入 7 5 运行验证" --workspace . --allow-write --allow-exec --command-timeout 10 --command-output-limit 65536 --max-steps 8
```

模型调用 `run_command` 时提供结构化 `argv`、工作区内的 `cwd` 和独立 `stdin` 文本。程序不会把参数拼成 Shell 命令，也不会自动解释管道、重定向或 `&` 等字符。`python` 会解析为当前 uv 环境实际使用的 Python 3.12 解释器。审批界面会显示解析后的完整参数、工作目录、stdin、超时和输出上限；只有批准后才启动进程，批准后还会重新核对命令与工作目录。

子进程结束后，工具以 JSON 返回 `stdout`、`stderr`、`exit_code`、`status`、执行耗时和 `truncated`，Agent Loop 再把该工具结果交回模型。`completed` 表示进程以退出码 0 结束，不表示生成的算法已经通过其他测试；非零退出返回 `failed`，超时返回 `timeout`，stdout/stderr 合计超过上限返回 `output_limit`。

- `--command-timeout` 默认 10 秒，可设置为大于 0 且不超过 60 的有限数值。
- `--command-output-limit` 默认 65536 字节，可设置为 1 至 1048576；stdout 和 stderr 共享该预算，达到上限时从读取阶段停止进程并标记截断。
- stdin、stdout 和 stderr 与审批终端输入彼此独立；stdin 写完后关闭，空字符串表示立即发送 EOF。
- 子进程使用最小化环境，不继承 DeepSeek、AWS 等凭据；`cwd` 必须是工作区内不经过符号链接或 reparse point 的已有目录。
- 超时、输出超限或 `Ctrl+C` 时会清理普通进程树。Windows 使用 Job Object，Linux 使用独立进程组。

当前版本不删除文件，也不提供自动审批。经批准的本地程序仍以当前用户权限运行，可以读写该用户有权访问的位置；`cwd` 限制、参数校验和进程清理用于降低误操作风险，不构成操作系统沙箱，也不保证约束刻意脱离进程组或改变权限的恶意程序。

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

`--max-steps` 限制一次运行的 Provider 调用次数，临时错误重试也计入该上限。`--max-retries` 设置每次临时错误最多重试次数。

CLI 会以非零退出码报告 `max_steps`、`interrupted`、Provider 错误和非正常模型停止原因；文件修改和本地命令均不会自动批准。工具执行错误会作为结构化结果交回模型，由模型决定是否说明、修正参数或结束任务，不会自动重试副作用命令。

## 可选的真实 DeepSeek 手工验证

自动测试始终使用 `FakeProvider`，不会调用付费 API。需要手工验证时，在未入库的本地环境中设置 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` 和 `DEEPSEEK_TIMEOUT_SECONDS`。真实 Key 不得写入受 Git 跟踪的文件、Shell 历史或截图；允许保存在已确认被忽略的本地 `.env` 中。

然后执行：

```powershell
uv run --env-file .env coding-agent "请读取 docs/architecture.md 并列出 Agent Loop 的停止条件" --workspace . --max-steps 6
```

预期结果是模型先请求 `read_file`，程序在本地读取该文件并将结果回传，随后模型输出最终回答；若模型直接回答，则不会产生工具消息。达到 `max_steps`、网络失败或按下 `Ctrl+C` 时，CLI 应显示对应停止原因并返回非零退出码。

真实写入或命令验证必须由用户明确决定，并增加对应的 `--allow-write` 或 `--allow-exec`。这会调用真实模型并产生 API 费用；在审批提示出现后，应先核对完整预览，再决定是否输入 `y`，不确认时直接按回车即可拒绝。自动测试使用 `FakeProvider`、无害的短 Python 进程和临时目录，不需要真实 API，也不会产生模型费用。
