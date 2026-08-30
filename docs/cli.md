# CLI 使用说明

`coding-agent` 每次运行处理一个任务。默认只提供工作区内的 `list_files` 和 `read_file`；显式使用 `--allow-write` 后，模型才可以申请调用 `write_file` 和 `edit_file`。

## 直接运行

在仓库根目录执行下面的命令即可开始一次任务：

```powershell
uv run coding-agent "请读取 docs/architecture.md 并总结项目架构" --workspace . --max-steps 8
```

也可以省略任务参数，程序会在终端提示输入：

```powershell
uv run coding-agent --workspace .
```

## 经审批的文件修改

需要创建或精确修改 UTF-8 文本文件时，显式增加 `--allow-write`：

```powershell
uv run coding-agent "请创建 hello.py，输出 Hello, world!" --workspace . --allow-write --max-steps 8
```

模型提出每一次 `write_file` 或 `edit_file` 调用后，CLI 都会先显示目标路径、待创建目录以及完整新内容或精确差异，然后提示：

```text
批准本次操作？[y/N]:
```

只有输入 `y` 或 `yes` 才批准本次操作。空输入、其他回答和无法读取输入均视为拒绝；`Ctrl+C` 会中断整个任务并返回退出码 130。`--allow-write` 只负责向模型开放工具，不会自动批准任何操作。

- `write_file` 只创建新文件，不覆盖任何已有文件或目录。
- `edit_file` 要求 `old_text` 非空且在原文件中恰好出现一次；不执行模糊、正则或隐式多处替换。
- 文件必须是 UTF-8 文本，原内容和结果默认均不超过 65536 字节。
- `.git`、`.local`、`.venv`、真实 `.env`、路径逃逸、符号链接、junction/reparse point 和危险 Windows 特殊路径会被拒绝。
- 审批后会重新核对目标；文件被修改、删除或替换时返回冲突，不套用过期操作。

当前版本不执行本地命令、不删除文件，也不提供自动审批。路径检查和逐次确认用于降低误操作风险，不等同于操作系统沙箱。

## Provider 配置

未通过测试注入 Provider 时，CLI 从环境变量创建 DeepSeek Provider。运行前在当前终端设置 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` 和 `DEEPSEEK_TIMEOUT_SECONDS`；不要把真实 Key 写入仓库或命令历史。

`--max-steps` 限制一次运行的 Provider 调用次数，临时错误重试也计入该上限。`--max-retries` 设置每次临时错误最多重试次数。

CLI 会以非零退出码报告 `max_steps`、`interrupted`、Provider 错误和非正常模型停止原因；不会自动批准文件修改，也不会执行命令工具。

## 可选的真实 DeepSeek 手工验证

自动测试始终使用 `FakeProvider`，不会调用付费 API。需要手工验证时，在未入库的本地环境中设置 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` 和 `DEEPSEEK_TIMEOUT_SECONDS`，其中 Key 只使用占位符替换，不要写入文件、Shell 历史或截图。

然后执行：

```powershell
uv run coding-agent "请读取 docs/architecture.md 并列出 Agent Loop 的停止条件" --workspace . --max-steps 6
```

预期结果是模型先请求 `read_file`，程序在本地读取该文件并将结果回传，随后模型输出最终回答；若模型直接回答，则不会产生工具消息。达到 `max_steps`、网络失败或按下 `Ctrl+C` 时，CLI 应显示对应停止原因并返回非零退出码。

真实写入验证必须由用户明确决定，并增加 `--allow-write`。在审批提示出现后，应先核对完整预览，再决定是否输入 `y`；不确认时直接按回车即可拒绝。自动测试使用 `FakeProvider` 和临时目录，不需要真实 API，也不会产生模型费用。
