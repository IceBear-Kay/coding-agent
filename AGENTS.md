# AGENTS.md

本文件是仓库内 Coding Agent 的稳定工作规则。具体任务范围与验收标准以当前 GitHub Issue 为准。

## 项目范围

本项目从零实现一个本地 Coding Agent。核心逻辑必须自行编写，包括消息与上下文管理、工具定义和本地执行、模型输出解析、Agent 循环终止与错误处理。

禁止引入 LangChain、LlamaIndex、OpenAI Agents SDK、Claude Agent SDK、AutoGen、CrewAI 等 Agent 框架或 SDK；禁止依赖服务端托管的文件和代码执行工具。

## 语言规范

- Issue 标题使用英文动词开头的简短句子；不添加 `[Feature]:`、`[Bug]:`、`[Task]:` 或 `feat:` 等分类前缀，任务分类由现有 GitHub 标签表达。
- 里程碑名称使用 `M<n> - <英文标题>` 格式，例如 `M4 - Interactive Sessions and Runtime Configuration`。
- Commit 标题和 PR 标题使用英文 Conventional Commits 格式。
- Issue/PR 正文、模板提示、README 和面向用户的公开说明统一使用专业、自然、成熟的中文，章节标题也保持中文一致。
- 文件名、代码、字段 ID、配置键、命令、GitHub 标签和 CI 检查名称保持原样；不要为了翻译而修改这些标识符。
- 英文技术术语的教学性中文释义只用于与用户的交流，不作为项目文档要求。
- 功能 PR 使用 `Refs #<issue-number>` 关联 Issue；合入 `develop` 后，按确认流程评论对应 PR 并手动关闭 Issue。
- 交付前检查 Issue 标题、里程碑名称、PR 标题与正文、模板及状态说明的一致性。

## 工作规则

1. 开始任务前读取当前 Issue、相关代码和测试，不无范围扩张。
2. 优先使用 `rg --files` 和 `rg` 定位内容。
3. 使用 Python 3.12 和 uv；不要创建或提交未约定的依赖管理方式。
4. 所有模型、工具和循环行为都应可在无网络条件下用 Fake/Test Double 测试。
5. 新行为必须包含相应测试；修复缺陷必须包含回归测试。
6. 工作区存在无关修改时保留它们，不覆盖、不删除。
7. 不提交 `.env`、API Key、`.local/`、虚拟环境、缓存或个人文件。
8. 默认不自动合并 PR，不 force push，不改写已推送历史。

## 架构边界

- Agent Loop 只依赖统一的 Provider、State 和 Tool 接口。
- Provider 负责把厂商响应转换为统一内部模型。
- 工具参数必须经过 Schema/模型校验，文件操作必须限制在工作区。
- 工具错误应结构化返回；认证等致命错误应停止循环。
- 模型名称、Base URL、超时和 Key 均来自配置，不在代码中硬编码凭据。

## 质量检查命令

项目基础建立后，提交前运行：

```powershell
uv sync --frozen
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

若命令尚未由当前 Milestone 建立，只执行仓库当前真实存在的检查，不用临时代码伪造通过。

## Git 与交付

- 临时分支从最新 `develop` 创建。
- 功能 PR 目标为 `develop`；发布 PR 为 `develop -> main`。
- Commit 和 PR 标题采用 Conventional Commits。
- 一个 Issue 对应一个分支和一个 PR；一个 PR 可有多个原子 Commit。
- 合并方式使用 Merge Commit，禁止 Squash、Rebase Merge 和 Force Push。
- 完成后报告修改文件、验证命令、PR 链接、风险与未完成项。

详细规范见 `CONTRIBUTING.md`，架构边界见 `docs/architecture.md`。
