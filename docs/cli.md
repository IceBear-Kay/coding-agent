# CLI 使用说明

`coding-agent` 每次运行处理一个任务。当前版本只提供工作区内的只读工具：`list_files` 和 `read_file`。

## 直接运行

在仓库根目录执行下面的命令即可开始一次任务：

```powershell
uv run coding-agent "请读取 README.md 并总结项目用途" --workspace . --max-steps 8
```

也可以省略任务参数，程序会在终端提示输入：

```powershell
uv run coding-agent --workspace .
```

## Provider 配置

未通过测试注入 Provider 时，CLI 从环境变量创建 DeepSeek Provider。运行前在当前终端设置 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` 和 `DEEPSEEK_TIMEOUT_SECONDS`；不要把真实 Key 写入仓库或命令历史。

`--max-steps` 限制一次运行的 Provider 调用次数，临时错误重试也计入该上限。`--max-retries` 设置每次临时错误最多重试次数。

CLI 会以非零退出码报告 `max_steps`、`interrupted`、Provider 错误和非正常模型停止原因；不会自动执行写文件或命令工具。
