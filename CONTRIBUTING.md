# Contributing Guide

本项目采用 Issue 驱动、分支开发、Pull Request（拉取请求）集成和 CI 自动验证。

## Branch model

- `main`：稳定发布版本。
- `develop`：开发集成版本。
- `feature/<issue>-<name>`：新功能。
- `fix/<issue>-<name>`：缺陷修复。
- `chore/<issue>-<name>`：配置、文档和 CI。
- `refactor/<issue>-<name>`：不改变外部行为的重构。

普通任务从最新 `develop` 创建临时分支，PR 目标为 `develop`。发布通过 `develop -> main` 的 PR 完成。

`main` 是仓库默认分支。由于功能 PR 合入的是 `develop`，GitHub 不会通过 `Closes #<issue>` 自动关闭 Issue。PR 合并后，应在关联 Issue 中评论对应 PR 已合入，并手动关闭该 Issue。

## Issue and PR size

一个 Issue 对应一个内聚、可独立验收的能力，以及一个临时分支和一个 PR。一个 PR 内可以有多个原子 Commit 和多次 Push；不要为每个函数创建 PR，也不要把多个无关 Issue 混在一起。

## Standard flow

```powershell
git switch develop
git pull --ff-only origin develop
git switch -c feature/3-deepseek-provider

# Implement and test
git add <explicit-paths>
git commit -m "feat(provider): implement DeepSeek provider"
git push -u origin feature/3-deepseek-provider
```

随后创建目标为 `develop` 的 PR，并在正文中使用 `Refs #<issue>`；合并后按上面的流程手动关闭 Issue。

## Conventional Commits

格式：

```text
<type>(optional-scope): <imperative summary>
```

允许的常用类型：

- `feat`：功能。
- `fix`：缺陷修复。
- `test`：测试。
- `refactor`：重构。
- `docs`：文档。
- `ci`：持续集成。
- `chore`：工程维护。
- `style`：纯格式调整。

Commit 标题使用简洁英文和祈使语气。每个 Commit 只表达一个清晰意图。

## Pull requests

PR 必须说明：

- 做了什么以及为什么。
- 关联的 Issue。
- 如何测试及测试结果。
- 风险、限制和未完成内容。
- 是否改变公开接口、配置或依赖。

合并前必须自行检查完整 Diff，并确保 CI 通过。项目使用 Merge Commit；不使用 Squash Merge 或 Rebase Merge，不 force push，不改写已推送历史。

## Quality gates

Python 基础建立后，本地与 CI 使用一致的命令：

```powershell
uv sync --frozen
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

新功能包含单元或端到端测试；缺陷修复包含能证明问题不再复现的回归测试。

## Security

- API Key 只通过环境变量或未跟踪的配置提供。
- 仅提交 `.env.example`，不得提交真实 `.env`。
- 不在 Issue、PR、Actions 日志、README 或演示视频中暴露凭据。
- 不提交 `.local/`、虚拟环境、缓存、IDE 私人配置或考核私人材料。
- 文件与命令工具必须遵守工作区边界、输出大小和超时限制。

## Definition of Done

- [ ] 实现与当前 Issue 范围一致。
- [ ] 验收条件全部满足。
- [ ] 相关测试与完整质量门禁通过。
- [ ] 文档与行为一致。
- [ ] 没有敏感信息和无关修改。
- [ ] PR 描述完整并关联 Issue。
- [ ] 使用 Merge Commit 合并，随后删除临时分支。

## Release

当 `develop` 达到一个可运行、可解释且测试通过的状态时，创建 `develop -> main` 的发布 PR。合并后按 Semantic Versioning（语义化版本）创建 Tag，例如 `v0.1.0`。截止时间后不再推送任何新提交。
