"""Command-line entry point for coding-agent tasks and in-memory chat sessions."""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import Any

from coding_agent.agent import (
    COMPLETED_STOP_REASON,
    DEFAULT_MAX_CONTEXT_BYTES,
    DEFAULT_MAX_STEPS,
    DEFAULT_SYSTEM_PROMPT,
    AgentEvent,
    AgentLoop,
    AgentRunResult,
)
from coding_agent.approval import ApprovalRequest
from coding_agent.command_tools import (
    DEFAULT_COMMAND_OUTPUT_LIMIT_BYTES,
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    CommandLimits,
)
from coding_agent.config import ProviderConfig
from coding_agent.errors import ProviderError
from coding_agent.file_tools import create_workspace_registry
from coding_agent.models import ToolResult
from coding_agent.provider import ModelProvider, OpenAICompatibleProvider
from coding_agent.session import AgentSession
from coding_agent.tools import Workspace

_STRUCTURED_RESULT_TOOLS = frozenset({"write_file", "edit_file", "run_command"})


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是正整数") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the parser separately so help and argument behavior are testable."""
    parser = argparse.ArgumentParser(
        prog="coding-agent",
        description="在工作区运行 coding-agent 任务。",
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="发送给 agent 的单次任务；省略后默认进入聊天，可用 --no-chat 改为单次输入。",
    )
    chat_group = parser.add_mutually_exclusive_group()
    chat_group.add_argument(
        "--chat",
        dest="chat",
        action="store_true",
        default=None,
        help="进入同一进程内的连续任务会话；不能与位置任务同时使用。",
    )
    chat_group.add_argument(
        "--no-chat",
        dest="chat",
        action="store_false",
        help="关闭连续任务会话；省略位置任务时只读取一次输入。",
    )
    parser.add_argument(
        "--workspace",
        "-w",
        default=".",
        help="工作区目录（默认：当前目录）。",
    )
    write_group = parser.add_mutually_exclusive_group()
    write_group.add_argument(
        "--allow-write",
        dest="allow_write",
        action="store_true",
        default=None,
        help="允许模型申请创建或精确修改文件；每次操作仍需确认。",
    )
    write_group.add_argument(
        "--no-write",
        dest="allow_write",
        action="store_false",
        help="关闭创建和修改文件工具。",
    )
    exec_group = parser.add_mutually_exclusive_group()
    exec_group.add_argument(
        "--allow-exec",
        dest="allow_exec",
        action="store_true",
        default=None,
        help="允许模型申请执行本地命令；每次操作仍需确认。",
    )
    exec_group.add_argument(
        "--no-exec",
        dest="allow_exec",
        action="store_false",
        help="关闭本地命令工具。",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="同时关闭写入和命令执行工具。",
    )
    event_group = parser.add_mutually_exclusive_group()
    event_group.add_argument(
        "--show-tool-events",
        dest="show_tool_events",
        action="store_true",
        default=None,
        help="显示正常工具调用和结果提示（默认）。",
    )
    event_group.add_argument(
        "--hide-tool-events",
        dest="show_tool_events",
        action="store_false",
        help="隐藏正常工具调用和结果提示；审批、错误和最终回答仍显示。",
    )
    parser.add_argument(
        "--command-timeout",
        type=float,
        default=DEFAULT_COMMAND_TIMEOUT_SECONDS,
        help=f"run_command 最大执行时间（默认：{DEFAULT_COMMAND_TIMEOUT_SECONDS} 秒）。",
    )
    parser.add_argument(
        "--command-output-limit",
        type=_positive_int,
        default=DEFAULT_COMMAND_OUTPUT_LIMIT_BYTES,
        help=(
            "run_command stdout/stderr 合计字节上限"
            f"（默认：{DEFAULT_COMMAND_OUTPUT_LIMIT_BYTES}）。"
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        default=DEFAULT_MAX_STEPS,
        help=f"Provider 最大调用次数（包括重试，默认：{DEFAULT_MAX_STEPS}）。",
    )
    parser.add_argument(
        "--max-retries",
        type=_non_negative_int,
        default=2,
        help="临时 Provider 错误的最大重试次数（默认：2）。",
    )
    parser.add_argument(
        "--max-context-bytes",
        type=_positive_int,
        default=DEFAULT_MAX_CONTEXT_BYTES,
        help=f"请求前上下文 UTF-8 字节预算（默认：{DEFAULT_MAX_CONTEXT_BYTES}）。",
    )
    parser.add_argument(
        "--context-policy",
        choices=("stop", "trim"),
        default="stop",
        help="上下文超预算时的处理策略（默认：stop）。",
    )
    return parser


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是非负整数") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("必须是非负整数")
    return parsed


def _default_provider() -> ModelProvider:
    return OpenAICompatibleProvider(ProviderConfig.from_env())


def _report_result(
    result: AgentRunResult,
    output_fn: Callable[[str], Any],
    error_fn: Callable[[str], Any],
) -> int:
    if result.answer is not None:
        output_fn(result.answer)

    if result.state.context_trimmed_tasks:
        output_fn(
            "上下文提示：已移除 "
            f"{result.state.context_trimmed_tasks} 个较早的完整任务以满足字节预算。"
        )

    if result.stop_reason == COMPLETED_STOP_REASON:
        output_fn(f"停止原因: {result.stop_reason}")
        return 0

    error_text = str(result.error) if result.error is not None else ""
    detail = f"：{error_text}" if error_text else ""
    error_fn(f"停止原因: {result.stop_reason}{detail}")
    return 130 if result.stop_reason == "interrupted" else 1


def _report_event(
    event: AgentEvent,
    output_fn: Callable[[str], Any],
    *,
    show_tool_events: bool = True,
) -> None:
    """Render only concise facts from real tool calls and structured results."""
    if event.kind == "tool_call" and event.tool_call is not None:
        if not show_tool_events:
            return
        tool_call = event.tool_call
        details = _tool_call_summary(tool_call.name, tool_call.arguments)
        suffix = f"，{details}" if details else ""
        output_fn(f"工具调用: {tool_call.name} ({tool_call.id}){suffix}")
        return

    if event.kind == "tool_result" and event.tool_result is not None:
        result = event.tool_result
        if not show_tool_events and not _tool_result_requires_notice(event.tool_name, result):
            return
        status, details = _tool_result_summary(event.tool_name, result.content)
        detail_text = f"，{details}" if details else ""
        error_text = "，错误" if result.is_error else ""
        tool_name = f" {event.tool_name}" if event.tool_name else ""
        output_fn(f"工具结果{tool_name}: {status or '已返回'}{error_text}{detail_text}")


def _tool_result_requires_notice(tool_name: str | None, result: ToolResult) -> bool:
    """Keep failures visible while hiding successful routine tool progress."""
    if result.is_error:
        return True
    if tool_name in {"list_files", "read_file"}:
        return False
    status, _ = _tool_result_summary(tool_name, result.content)
    return status not in {"created", "edited", "completed"}


def _tool_call_summary(tool_name: str, arguments: dict[str, Any]) -> str:
    if "path" in arguments and isinstance(arguments["path"], str):
        return f"路径: {arguments['path']}"
    if tool_name == "run_command":
        argv = arguments.get("argv")
        cwd = arguments.get("cwd", ".")
        if isinstance(argv, list):
            return f"argv: {json.dumps(argv, ensure_ascii=False)}，cwd: {cwd}"
    return ""


def _tool_result_summary(tool_name: str | None, content: str) -> tuple[str | None, str | None]:
    if tool_name not in _STRUCTURED_RESULT_TOOLS:
        return None, None

    try:
        payload = json.loads(content)
    except (TypeError, ValueError, RecursionError):
        return None, None
    if not isinstance(payload, dict):
        return None, None

    status = payload.get("status") if isinstance(payload.get("status"), str) else None
    fields: list[str] = []
    for name in ("path", "exit_code", "truncated", "bytes_written", "replacements"):
        value = payload.get(name)
        if value is not None:
            fields.append(f"{name}: {value}")
    return status, "，".join(fields) or None


def _report_interrupt(error_fn: Callable[[str], Any]) -> None:
    with suppress(KeyboardInterrupt):
        error_fn("停止原因: interrupted")


def _prompt_for_approval(
    request: ApprovalRequest,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], Any],
) -> bool:
    output_fn(f"待审批操作: {request.operation}\n{request.preview}")
    if input_fn is input and not _stdin_is_interactive():
        output_fn("审批结果: 已拒绝（非交互输入不能用于审批）")
        return False
    try:
        answer = input_fn("批准本次操作？[y/N]: ")
    except EOFError:
        output_fn("审批结果: 已拒绝（无法读取输入）")
        return False

    approved = answer.strip().casefold() in {"y", "yes"}
    output_fn("审批结果: 已批准" if approved else "审批结果: 已拒绝")
    return approved


def _stdin_is_interactive() -> bool:
    try:
        return sys.stdin.isatty()
    except (AttributeError, OSError):
        return False


def _run_chat(
    session: AgentSession,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], Any],
    error_fn: Callable[[str], Any],
) -> int:
    """Read tasks until an explicit exit, EOF, interrupt, or abnormal result."""
    while True:
        try:
            raw_task = input_fn("任务（/clear 清空历史，/exit 退出）: ")
        except EOFError:
            return 0

        task = raw_task.strip()
        if task == "/exit":
            return 0
        if task == "/clear":
            session.clear()
            output_fn("会话历史已清空")
            continue
        if not task:
            continue

        result = session.run(task)
        exit_code = _report_result(result, output_fn, error_fn)
        if exit_code != 0:
            return exit_code


def main(
    argv: Sequence[str] | None = None,
    *,
    provider: ModelProvider | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], Any] = print,
    error_fn: Callable[[str], Any] | None = None,
) -> int:
    """Run one task or an in-memory chat session and return a process-style exit code.

    ``provider`` is injectable for offline tests; normal CLI use creates the configured
    OpenAI-compatible provider from the environment.
    """
    report_error = error_fn or (lambda message: print(message, file=sys.stderr))

    try:
        args = build_parser().parse_args(argv)
        if args.chat is True and args.task is not None:
            report_error("错误: --chat 不能与位置任务同时使用")
            return 2

        if args.read_only and (args.allow_write is True or args.allow_exec is True):
            report_error("错误: --read-only 不能与 --allow-write 或 --allow-exec 同时使用")
            return 2

        effective_chat = args.chat if args.chat is not None else args.task is None
        effective_allow_write = (
            False if args.read_only else True if args.allow_write is None else args.allow_write
        )
        effective_allow_exec = (
            False if args.read_only else True if args.allow_exec is None else args.allow_exec
        )
        effective_show_tool_events = (
            True if args.show_tool_events is None else args.show_tool_events
        )

        task = args.task
        if task is None and not effective_chat:
            task = input_fn("任务: ").strip()
        if not effective_chat and not task:
            report_error("错误: task 不能为空")
            return 2

        workspace = Workspace(args.workspace)
        command_limits = CommandLimits(
            timeout_seconds=args.command_timeout,
            output_limit_bytes=args.command_output_limit,
        )

        def approve_operation(request: ApprovalRequest) -> bool:
            return _prompt_for_approval(
                request,
                input_fn,
                output_fn,
            )

        registry = create_workspace_registry(
            workspace,
            allow_write=effective_allow_write,
            allow_exec=effective_allow_exec,
            approval_callback=approve_operation
            if (effective_allow_write or effective_allow_exec)
            else None,
            command_limits=command_limits,
        )
        selected_provider = provider if provider is not None else _default_provider()
        loop = AgentLoop(
            selected_provider,
            workspace,
            registry=registry,
            max_steps=args.max_steps,
            max_retries=args.max_retries,
            max_context_bytes=args.max_context_bytes,
            context_policy=args.context_policy,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            event_callback=lambda event: _report_event(
                event,
                output_fn,
                show_tool_events=effective_show_tool_events,
            ),
        )
        if effective_chat:
            return _run_chat(
                AgentSession(loop),
                input_fn,
                output_fn,
                report_error,
            )

        result = loop.run(task)
        return _report_result(result, output_fn, report_error)
    except KeyboardInterrupt:
        _report_interrupt(report_error)
        return 130
    except EOFError:
        report_error("错误: 无法读取 task")
        return 2
    except (ProviderError, OSError, ValueError) as exc:
        report_error(f"错误: {exc}")
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through the console script.
    raise SystemExit(main())
