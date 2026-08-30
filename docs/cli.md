# CLI 使用说明

`coding-agent` 每次运行处理一个任务。当前版本只提供工作区内的只读工具：`list_files` 和 `read_file`。

## 直接运行

在仓库根目录执行下面的命令即可开始一次任务：

```powershell
uv run coding-agent "请读取 docs/architecture.md 并总结项目架构" --workspace . --max-steps 8
```

也可以省略任务参数，程序会在终端提示输入：

```powershell
uv run coding-agent --workspace .
```

## Provider 配置

未通过测试注入 Provider 时，CLI 从环境变量创建 DeepSeek Provider。运行前在当前终端设置 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` 和 `DEEPSEEK_TIMEOUT_SECONDS`；不要把真实 Key 写入仓库或命令历史。

`--max-steps` 限制一次运行的 Provider 调用次数，临时错误重试也计入该上限。`--max-retries` 设置每次临时错误最多重试次数。

CLI 会以非零退出码报告 `max_steps`、`interrupted`、Provider 错误和非正常模型停止原因；不会自动执行写文件或命令工具。

## 可选的真实 DeepSeek 手工验证

自动测试始终使用 `FakeProvider`，不会调用付费 API。需要手工验证时，在未入库的本地环境中设置 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` 和 `DEEPSEEK_TIMEOUT_SECONDS`，其中 Key 只使用占位符替换，不要写入文件、Shell 历史或截图。

然后执行：

```powershell
uv run coding-agent "请读取 docs/architecture.md 并列出 Agent Loop 的停止条件" --workspace . --max-steps 6
```

预期结果是模型先请求 `read_file`，程序在本地读取该文件并将结果回传，随后模型输出最终回答；若模型直接回答，则不会产生工具消息。达到 `max_steps`、网络失败或按下 `Ctrl+C` 时，CLI 应显示对应停止原因并返回非零退出码。
