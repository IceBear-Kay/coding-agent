"""Command-line entry point for one coding-agent task."""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import Any

from coding_agent.agent import (
    COMPLETED_STOP_REASON,
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
from coding_agent.provider import ModelProvider, OpenAICompatibleProvider
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
        description="在工作区运行一次 coding-agent 任务。",
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="发送给 agent 的任务；省略后将交互式输入。",
    )
    parser.add_argument(
        "--workspace",
        "-w",
        default=".",
        help="工作区目录（默认：当前目录）。",
    )
    parser.add_argument(
        "--allow-write",
        action="store_true",
        help="允许模型申请创建或精确修改文件；每次操作仍需确认。",
    )
    parser.add_argument(
        "--allow-exec",
        action="store_true",
        help="允许模型申请执行本地命令；每次操作仍需确认。",
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

    if result.stop_reason == COMPLETED_STOP_REASON:
        output_fn(f"停止原因: {result.stop_reason}")
        return 0

    error_text = str(result.error) if result.error is not None else ""
    detail = f"：{error_text}" if error_text else ""
    error_fn(f"停止原因: {result.stop_reason}{detail}")
    return 130 if result.stop_reason == "interrupted" else 1


def _report_event(event: AgentEvent, output_fn: Callable[[str], Any]) -> None:
    """Render only concise facts from real tool calls and structured results."""
    if event.kind == "tool_call" and event.tool_call is not None:
        tool_call = event.tool_call
        details = _tool_call_summary(tool_call.name, tool_call.arguments)
        suffix = f"，{details}" if details else ""
        output_fn(f"工具调用: {tool_call.name} ({tool_call.id}){suffix}")
        return

    if event.kind == "tool_result" and event.tool_result is not None:
        result = event.tool_result
        status, details = _tool_result_summary(event.tool_name, result.content)
        detail_text = f"，{details}" if details else ""
        error_text = "，错误" if result.is_error else ""
        tool_name = f" {event.tool_name}" if event.tool_name else ""
        output_fn(f"工具结果{tool_name}: {status or '已返回'}{error_text}{detail_text}")


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


def main(
    argv: Sequence[str] | None = None,
    *,
    provider: ModelProvider | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], Any] = print,
    error_fn: Callable[[str], Any] | None = None,
) -> int:
    """Run one task and return a process-style exit code.

    ``provider`` is injectable for offline tests; normal CLI use creates the configured
    OpenAI-compatible provider from the environment.
    """
    report_error = error_fn or (lambda message: print(message, file=sys.stderr))

    try:
        args = build_parser().parse_args(argv)
        task = args.task
        if task is None:
            task = input_fn("任务: ").strip()
        if not task:
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
            allow_write=args.allow_write,
            allow_exec=args.allow_exec,
            approval_callback=approve_operation if (args.allow_write or args.allow_exec) else None,
            command_limits=command_limits,
        )
        selected_provider = provider if provider is not None else _default_provider()
        result = AgentLoop(
            selected_provider,
            workspace,
            registry=registry,
            max_steps=args.max_steps,
            max_retries=args.max_retries,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            event_callback=lambda event: _report_event(event, output_fn),
        ).run(task)
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
