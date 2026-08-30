"""Command-line entry point for one coding-agent task."""

import argparse
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import Any

from coding_agent.agent import COMPLETED_STOP_REASON, DEFAULT_MAX_STEPS, AgentLoop, AgentRunResult
from coding_agent.approval import ApprovalRequest
from coding_agent.config import ProviderConfig
from coding_agent.errors import ProviderError
from coding_agent.file_tools import create_workspace_registry
from coding_agent.provider import ModelProvider, OpenAICompatibleProvider
from coding_agent.tools import Workspace


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
        return 0

    error_text = str(result.error) if result.error is not None else ""
    detail = f"：{error_text}" if error_text else ""
    error_fn(f"停止原因: {result.stop_reason}{detail}")
    return 130 if result.stop_reason == "interrupted" else 1


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

        def approve_operation(request: ApprovalRequest) -> bool:
            return _prompt_for_approval(
                request,
                input_fn,
                output_fn,
            )

        registry = create_workspace_registry(
            workspace,
            allow_write=args.allow_write,
            approval_callback=approve_operation if args.allow_write else None,
        )
        selected_provider = provider if provider is not None else _default_provider()
        result = AgentLoop(
            selected_provider,
            workspace,
            registry=registry,
            max_steps=args.max_steps,
            max_retries=args.max_retries,
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
