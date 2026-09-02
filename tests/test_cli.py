import json
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest

from coding_agent import session_store as session_store_module
from coding_agent.agent import DEFAULT_SYSTEM_PROMPT
from coding_agent.cli import build_parser, main
from coding_agent.context import measure_context_bytes
from coding_agent.models import Message, ModelResponse, ToolCall, Usage
from coding_agent.provider import FakeProvider
from coding_agent.session_store import SessionStore, SessionStoreError
from coding_agent.tools import Workspace, create_read_only_registry


def test_cli_parser_accepts_task_workspace_limits_and_tool_flags() -> None:
    args = build_parser().parse_args(
        [
            "检查项目",
            "--workspace",
            "D:/coding-agent-demo",
            "--allow-write",
            "--allow-exec",
            "--command-timeout",
            "3.5",
            "--command-output-limit",
            "2048",
            "--max-steps",
            "12",
            "--max-retries",
            "0",
        ]
    )

    assert args.task == "检查项目"
    assert args.workspace == "D:/coding-agent-demo"
    assert args.allow_write is True
    assert args.allow_exec is True
    assert args.command_timeout == 3.5
    assert args.command_output_limit == 2048
    assert args.max_steps == 12
    assert args.max_retries == 0


def test_cli_parser_accepts_chat_mode() -> None:
    args = build_parser().parse_args(["--chat"])

    assert args.chat is True
    assert args.task is None


def test_cli_parser_accepts_persistent_session_options() -> None:
    args = build_parser().parse_args(["--chat", "--session", "chat_1", "--session-dir", "sessions"])

    assert args.chat is True
    assert args.session == "chat_1"
    assert args.resume is None
    assert args.session_dir == "sessions"

    resumed = build_parser().parse_args(["--chat", "--resume", "chat_1"])
    assert resumed.resume == "chat_1"


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--session", "chat_1", "任务"], "错误: 持久会话不能与位置任务同时使用"),
        (
            ["--no-chat", "--session", "chat_1"],
            "错误: --session 或 --resume 只能与 --chat 一起使用",
        ),
        (
            ["--session-dir", "sessions"],
            "错误: --session-dir 必须与 --session 或 --resume 一起使用",
        ),
    ],
)
def test_cli_rejects_invalid_persistence_combinations(
    tmp_path: Path, argv: list[str], message: str
) -> None:
    errors: list[str] = []
    exit_code = main(
        [*argv, "--workspace", str(tmp_path)],
        provider=FakeProvider([]),
        error_fn=errors.append,
    )

    assert exit_code == 2
    assert errors == [message]


def test_cli_persistent_session_creates_archive_and_saves_completed_tasks(tmp_path: Path) -> None:
    store_root = tmp_path / "archives"
    provider = FakeProvider(
        [
            ModelResponse(text="第一项完成", finish_reason="stop"),
            ModelResponse(text="第二项完成", finish_reason="stop"),
        ]
    )
    inputs = iter(["第一项任务", "第二项任务", "/exit"])
    output: list[str] = []

    exit_code = main(
        [
            "--chat",
            "--session",
            "chat_1",
            "--session-dir",
            str(store_root),
            "--workspace",
            str(tmp_path),
            "--read-only",
        ],
        provider=provider,
        input_fn=lambda _: next(inputs),
        output_fn=output.append,
    )

    assert exit_code == 0
    assert output[0].startswith("持久会话 ID: chat_1\n存档路径:")
    archive = json.loads((store_root / "chat_1.json").read_text(encoding="utf-8"))
    assert [message["role"] for message in archive["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [message["content"] for message in archive["messages"] if message["role"] == "user"] == [
        "第一项任务",
        "第二项任务",
    ]


def test_cli_persistent_resume_does_not_replay_provider_or_tools(tmp_path: Path) -> None:
    store_root = tmp_path / "archives"
    first_inputs = iter(["保存历史", "/exit"])
    first_provider = FakeProvider([ModelResponse(text="已保存", finish_reason="stop")])
    assert (
        main(
            [
                "--chat",
                "--session",
                "chat_1",
                "--session-dir",
                str(store_root),
                "--workspace",
                str(tmp_path),
                "--read-only",
            ],
            provider=first_provider,
            input_fn=lambda _: next(first_inputs),
        )
        == 0
    )

    resumed_provider = FakeProvider([])
    output: list[str] = []
    exit_code = main(
        [
            "--chat",
            "--resume",
            "chat_1",
            "--session-dir",
            str(store_root),
            "--workspace",
            str(tmp_path),
            "--read-only",
        ],
        provider=resumed_provider,
        input_fn=lambda _: "/exit",
        output_fn=output.append,
    )

    assert exit_code == 0
    assert resumed_provider.requests == []
    assert output[0].startswith("持久会话 ID: chat_1\n存档路径:")
    assert "保存历史" not in output[0]


def test_cli_resume_adds_stale_history_rule_to_provider_system_prompt(tmp_path: Path) -> None:
    store_root = tmp_path / "archives"
    first_inputs = iter(["保存历史", "/exit"])
    assert (
        main(
            [
                "--chat",
                "--session",
                "chat_1",
                "--session-dir",
                str(store_root),
                "--workspace",
                str(tmp_path),
                "--read-only",
            ],
            provider=FakeProvider([ModelResponse(text="已保存", finish_reason="stop")]),
            input_fn=lambda _: next(first_inputs),
        )
        == 0
    )

    provider = FakeProvider([ModelResponse(text="已继续", finish_reason="stop")])
    inputs = iter(["继续任务", "/exit"])
    output: list[str] = []
    assert (
        main(
            [
                "--chat",
                "--resume",
                "chat_1",
                "--session-dir",
                str(store_root),
                "--workspace",
                str(tmp_path),
                "--read-only",
            ],
            provider=provider,
            input_fn=lambda _: next(inputs),
            output_fn=output.append,
        )
        == 0
    )

    system = provider.requests[0][0][0]
    assert system.role == "system"
    assert system.content is not None
    assert system.content.count("历史工具结果可能已过时") == 1
    assert sum(message.role == "system" for message in provider.requests[0][0]) == 1


def test_cli_archive_io_failure_reports_save_error_and_keeps_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_root = tmp_path / "archives"
    calls = 0
    real_fsync = session_store_module.os.fsync

    def fail_during_save(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls > 3:
            raise OSError("simulated archive fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(session_store_module.os, "fsync", fail_during_save)
    output: list[str] = []
    errors: list[str] = []
    exit_code = main(
        [
            "--chat",
            "--session",
            "chat_1",
            "--session-dir",
            str(store_root),
            "--workspace",
            str(tmp_path),
            "--read-only",
        ],
        provider=FakeProvider([ModelResponse(text="真实答案", finish_reason="stop")]),
        input_fn=lambda _: "完成任务",
        output_fn=output.append,
        error_fn=errors.append,
    )

    assert exit_code == 1
    assert "真实答案" in output
    assert errors and errors[0].startswith("停止原因: session_save_error")
    archive = json.loads((store_root / "chat_1.json").read_text(encoding="utf-8"))
    assert archive["messages"] == []


def test_cli_rejects_cross_process_session_occupancy_before_provider_or_tools(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "archives"
    SessionStore(store_root).create("chat_1", tmp_path)
    child_code = (
        "import sys,time; "
        "from coding_agent.session_store import SessionStore; "
        "lease=SessionStore(sys.argv[1]).acquire('chat_1'); "
        "print('ready', flush=True); time.sleep(30)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", child_code, str(store_root)],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        provider = FakeProvider([])
        errors: list[str] = []
        exit_code = main(
            [
                "--chat",
                "--resume",
                "chat_1",
                "--session-dir",
                str(store_root),
                "--workspace",
                str(tmp_path),
                "--read-only",
            ],
            provider=provider,
            error_fn=errors.append,
        )
    finally:
        process.terminate()
        process.wait(timeout=10)

    assert exit_code == 2
    assert provider.requests == []
    assert errors == ["错误: session is already in use"]
    assert not (tmp_path / "unexpected.txt").exists()


def test_cli_persistent_clear_keeps_old_archive_and_switches_id(tmp_path: Path) -> None:
    store_root = tmp_path / "archives"
    provider = FakeProvider(
        [
            ModelResponse(text="旧回答", finish_reason="stop"),
            ModelResponse(text="新回答", finish_reason="stop"),
        ]
    )
    inputs = iter(["旧任务", "/clear", "新任务", "/exit"])
    output: list[str] = []

    exit_code = main(
        [
            "--chat",
            "--session",
            "chat_old",
            "--session-dir",
            str(store_root),
            "--workspace",
            str(tmp_path),
            "--read-only",
        ],
        provider=provider,
        input_fn=lambda _: next(inputs),
        output_fn=output.append,
    )

    assert exit_code == 0
    archives = sorted(store_root.glob("*.json"))
    assert {path.stem for path in archives} >= {"chat_old"}
    assert len(archives) == 2
    old = json.loads((store_root / "chat_old.json").read_text(encoding="utf-8"))
    new_path = next(path for path in archives if path.stem != "chat_old")
    new = json.loads(new_path.read_text(encoding="utf-8"))
    assert "旧任务" in [message.get("content") for message in old["messages"]]
    assert "新任务" in [message.get("content") for message in new["messages"]]
    assert "旧任务" not in [message.get("content") for message in new["messages"]]
    assert any("已切换到新持久会话" in message for message in output)


def test_cli_persistent_clear_failure_keeps_current_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_root = tmp_path / "archives"
    provider = FakeProvider([ModelResponse(text="旧回答", finish_reason="stop")])
    inputs = iter(["旧任务", "/clear", "/exit"])
    errors: list[str] = []

    class FixedUuid:
        hex = "chat_old"

    monkeypatch.setattr("coding_agent.cli.uuid.uuid4", lambda: FixedUuid())
    exit_code = main(
        [
            "--chat",
            "--session",
            "chat_old",
            "--session-dir",
            str(store_root),
            "--workspace",
            str(tmp_path),
            "--read-only",
        ],
        provider=provider,
        input_fn=lambda _: next(inputs),
        error_fn=errors.append,
    )

    assert exit_code == 0
    assert errors and errors[0].startswith("错误: 无法切换持久会话：")
    assert sorted(path.name for path in store_root.glob("*.json")) == ["chat_old.json"]


def test_cli_clear_release_failure_stops_without_follow_up_model_or_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_root = tmp_path / "archives"
    original_acquire = SessionStore.acquire
    leases = []

    def capture_acquire(store: SessionStore, session_id: str):
        lease = original_acquire(store, session_id)
        leases.append(lease)
        if len(leases) == 2:
            original_release = lease.release
            failed = False

            def fail_once() -> None:
                nonlocal failed
                if not failed:
                    failed = True
                    raise SessionStoreError("simulated old lock release failure")
                original_release()

            lease.release = fail_once  # type: ignore[method-assign]
        return lease

    monkeypatch.setattr(SessionStore, "acquire", capture_acquire)
    provider = FakeProvider([ModelResponse(text="旧回答", finish_reason="stop")])
    inputs = iter(["旧任务", "/clear", "不应执行"])
    errors: list[str] = []
    exit_code = main(
        [
            "--chat",
            "--session",
            "chat_old",
            "--session-dir",
            str(store_root),
            "--workspace",
            str(tmp_path),
            "--read-only",
        ],
        provider=provider,
        input_fn=lambda _: next(inputs),
        error_fn=errors.append,
    )

    assert exit_code == 2
    assert len(provider.requests) == 1
    assert errors and any("旧持久会话锁未能释放" in message for message in errors)
    assert not (tmp_path / "unexpected.txt").exists()


def test_cli_startup_output_failure_releases_session_lock(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "archives"

    def fail_output(_: str) -> None:
        raise RuntimeError("simulated output failure")

    with pytest.raises(RuntimeError, match="simulated output failure"):
        main(
            [
                "--chat",
                "--session",
                "chat_1",
                "--session-dir",
                str(store_root),
                "--workspace",
                str(tmp_path),
                "--read-only",
            ],
            provider=FakeProvider([]),
            output_fn=fail_output,
        )

    lease = SessionStore(store_root).acquire("chat_1")
    lease.release()


def test_cli_memory_chat_does_not_create_session_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    inputs = iter(["普通聊天", "/exit"])
    exit_code = main(
        ["--chat", "--workspace", str(tmp_path), "--read-only"],
        provider=FakeProvider([ModelResponse(text="完成", finish_reason="stop")]),
        input_fn=lambda _: next(inputs),
    )

    assert exit_code == 0
    assert not (tmp_path / ".local" / "sessions").exists()


def test_cli_parser_preserves_unspecified_defaults_for_mode_and_permissions() -> None:
    args = build_parser().parse_args([])

    assert args.chat is None
    assert args.allow_write is None
    assert args.allow_exec is None
    assert args.read_only is False
    assert args.show_tool_events is None
    assert args.show_stats is False
    assert args.command_timeout == 20.0


def test_cli_show_stats_reports_task_diagnostics_without_changing_default_output(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                text="Done",
                finish_reason="stop",
                usage=Usage(input_tokens=5, output_tokens=2, total_tokens=7),
            )
        ]
    )
    output: list[str] = []

    exit_code = main(
        ["Done task", "--workspace", str(tmp_path), "--read-only", "--show-stats"],
        provider=provider,
        output_fn=output.append,
    )

    assert exit_code == 0
    assert output[0] == "Done"
    assert output[1].startswith("运行统计: ")
    assert "Provider 请求: 1 次" in output[1]
    assert "输入 Token: 5" in output[1]
    assert output[2] == "停止原因: completed"

    default_output: list[str] = []
    exit_code = main(
        ["Done task", "--workspace", str(tmp_path), "--read-only"],
        provider=FakeProvider([ModelResponse(text="Done", finish_reason="stop")]),
        output_fn=default_output.append,
    )

    assert exit_code == 0
    assert default_output == ["Done", "停止原因: completed"]


def test_cli_parser_accepts_explicit_disable_switches() -> None:
    args = build_parser().parse_args(
        ["--no-chat", "--no-write", "--no-exec", "--read-only", "--hide-tool-events"]
    )

    assert args.chat is False
    assert args.allow_write is False
    assert args.allow_exec is False
    assert args.read_only is True
    assert args.show_tool_events is False


@pytest.mark.parametrize(
    "flags",
    [
        ("--chat", "--no-chat"),
        ("--allow-write", "--no-write"),
        ("--allow-exec", "--no-exec"),
        ("--show-tool-events", "--hide-tool-events"),
    ],
)
def test_cli_parser_rejects_opposite_switches(flags: tuple[str, str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(list(flags))

    assert exc_info.value.code == 2


def test_cli_parser_accepts_context_budget() -> None:
    args = build_parser().parse_args(["--max-context-bytes", "4096"])

    assert args.max_context_bytes == 4096


def test_cli_parser_context_policy_defaults_to_trim_and_accepts_stop() -> None:
    assert build_parser().parse_args([]).context_policy == "trim"
    assert build_parser().parse_args(["--context-policy", "trim"]).context_policy == "trim"

    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--context-policy", "invalid"])

    assert exc_info.value.code == 2


def test_cli_parser_exposes_runtime_budget_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.max_steps == 64
    assert args.max_context_tokens == 524_288
    assert args.max_output_tokens == 32_768
    assert args.max_context_bytes == 8_388_608


@pytest.mark.parametrize("model", ["deepseek-v4-flash", "deepseek-v4-pro"])
def test_cli_applies_v4_output_default_override_and_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", model)
    monkeypatch.setenv("DEEPSEEK_TIMEOUT_SECONDS", "30")
    providers: list[FakeProvider] = []

    def create_provider(_config: object) -> FakeProvider:
        provider = FakeProvider([ModelResponse(text="ok", finish_reason="stop")])
        providers.append(provider)
        return provider

    monkeypatch.setattr("coding_agent.cli.OpenAICompatibleProvider", create_provider)

    assert (
        main(
            ["hello", "--workspace", str(tmp_path)],
            provider=None,
        )
        == 0
    )
    assert providers[-1].max_tokens == [32_768]

    assert (
        main(
            ["hello", "--workspace", str(tmp_path), "--max-output-tokens", "65536"],
            provider=None,
        )
        == 0
    )
    assert providers[-1].max_tokens == [65_536]

    errors: list[str] = []
    provider_count = len(providers)
    assert (
        main(
            ["hello", "--workspace", str(tmp_path), "--max-output-tokens", "384001"],
            provider=None,
            error_fn=errors.append,
        )
        == 2
    )
    assert errors and "max-output-tokens" in errors[0]
    assert len(providers) == provider_count


def test_cli_rejects_non_positive_context_budget_before_provider_call(tmp_path: Path) -> None:
    provider = FakeProvider([])
    errors: list[str] = []

    with pytest.raises(SystemExit) as exc_info:
        main(
            ["Inspect", "--workspace", str(tmp_path), "--max-context-bytes", "0"],
            provider=provider,
            error_fn=errors.append,
        )

    assert exc_info.value.code == 2
    assert provider.requests == []


def test_cli_reports_context_limit_without_exposing_task_content(tmp_path: Path) -> None:
    provider = FakeProvider([])
    errors: list[str] = []
    task = "private task content that must not be echoed"

    exit_code = main(
        [task, "--workspace", str(tmp_path), "--max-context-bytes", "1"],
        provider=provider,
        error_fn=errors.append,
    )

    assert exit_code == 1
    assert len(provider.requests) == 0
    assert errors[0].startswith("停止原因: context_limit")
    assert "context input exceeds byte budget" in errors[0]
    assert task not in errors[0]


def test_cli_chat_rejects_positional_task_without_provider_call(tmp_path: Path) -> None:
    provider = FakeProvider([])
    errors: list[str] = []

    exit_code = main(
        ["--chat", "already supplied", "--workspace", str(tmp_path)],
        provider=provider,
        error_fn=errors.append,
    )

    assert exit_code == 2
    assert errors == ["错误: --chat 不能与位置任务同时使用"]
    assert provider.requests == []


@pytest.mark.parametrize(
    "flags",
    [
        ("--read-only", "--allow-write"),
        ("--allow-write", "--read-only"),
        ("--read-only", "--allow-exec"),
        ("--allow-exec", "--read-only"),
    ],
)
def test_cli_rejects_read_only_enable_conflicts_before_provider_call(
    tmp_path: Path,
    flags: tuple[str, str],
) -> None:
    provider = FakeProvider([])
    errors: list[str] = []

    exit_code = main(
        ["Inspect", "--workspace", str(tmp_path), *flags],
        provider=provider,
        error_fn=errors.append,
    )

    assert exit_code == 2
    assert errors == ["错误: --read-only 不能与 --allow-write 或 --allow-exec 同时使用"]
    assert provider.requests == []


def test_cli_defaults_to_chat_when_task_is_omitted(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ModelResponse(text="第一轮回答", finish_reason="stop"),
            ModelResponse(text="第二轮回答", finish_reason="stop"),
        ]
    )
    inputs = iter(["第一项任务", "第二项任务", "/exit"])

    exit_code = main(
        ["--workspace", str(tmp_path)],
        provider=provider,
        input_fn=lambda _: next(inputs),
    )

    assert exit_code == 0
    assert len(provider.requests) == 2
    assert provider.requests[0][0][-1].content == "第一项任务"
    assert provider.requests[1][0][-1].content == "第二项任务"


def test_cli_positional_task_defaults_to_single_run_without_reading_input(
    tmp_path: Path,
) -> None:
    provider = FakeProvider([ModelResponse(text="完成", finish_reason="stop")])

    def unexpected_input(_: str) -> str:
        raise AssertionError("single-task mode must not read another task")

    exit_code = main(
        ["检查工作区", "--workspace", str(tmp_path)],
        provider=provider,
        input_fn=unexpected_input,
    )

    assert exit_code == 0
    assert len(provider.requests) == 1
    assert provider.requests[0][0][-1].content == "检查工作区"


def test_cli_chat_preserves_history_and_resets_each_task_budget(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ModelResponse(text="第一轮回答", finish_reason="stop"),
            ModelResponse(text="第二轮回答", finish_reason="stop"),
        ]
    )
    inputs = iter(["第一项任务", "第二项任务", "/exit"])
    output: list[str] = []

    exit_code = main(
        ["--chat", "--workspace", str(tmp_path), "--max-steps", "1"],
        provider=provider,
        input_fn=lambda _: next(inputs),
        output_fn=output.append,
    )

    assert exit_code == 0
    assert output == [
        "第一轮回答",
        "停止原因: completed",
        "第二轮回答",
        "停止原因: completed",
    ]
    assert provider.requests[0][0][-1].content == "第一项任务"
    assert [message.role for message in provider.requests[1][0]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert sum(message.role == "system" for message in provider.requests[1][0]) == 1


def test_cli_chat_clear_discards_history_without_calling_provider(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ModelResponse(text="旧回答", finish_reason="stop"),
            ModelResponse(text="新回答", finish_reason="stop"),
        ]
    )
    inputs = iter(["旧任务", "", "/clear", "新任务", "/exit"])
    output: list[str] = []

    exit_code = main(
        ["--chat", "--workspace", str(tmp_path)],
        provider=provider,
        input_fn=lambda _: next(inputs),
        output_fn=output.append,
    )

    assert exit_code == 0
    assert output == [
        "旧回答",
        "停止原因: completed",
        "会话历史已清空",
        "新回答",
        "停止原因: completed",
    ]
    assert [message.role for message in provider.requests[1][0]] == ["system", "user"]
    assert provider.requests[1][0][-1].content == "新任务"


def test_cli_chat_clear_allows_new_task_after_history_budget_would_be_exceeded(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        [
            ModelResponse(text="旧回答", finish_reason="stop"),
            ModelResponse(text="新回答", finish_reason="stop"),
        ]
    )
    workspace = Workspace(tmp_path)
    registry = create_read_only_registry(workspace)
    system = Message(role="system", content=DEFAULT_SYSTEM_PROMPT)
    old_task = "旧任务"
    new_task = "新任务"
    budget = measure_context_bytes(
        [system, Message(role="user", content=new_task)],
        registry.schemas(),
    )
    accumulated = measure_context_bytes(
        [
            system,
            Message(role="user", content=old_task),
            Message(role="assistant", content="旧回答"),
            Message(role="user", content=new_task),
        ],
        registry.schemas(),
    )
    assert accumulated > budget
    inputs = iter([old_task, "/clear", new_task, "/exit"])
    output: list[str] = []

    exit_code = main(
        [
            "--chat",
            "--workspace",
            str(tmp_path),
            "--read-only",
            "--max-steps",
            "1",
            "--max-context-bytes",
            str(budget),
        ],
        provider=provider,
        input_fn=lambda _: next(inputs),
        output_fn=output.append,
    )

    assert exit_code == 0
    assert len(provider.requests) == 2
    assert [message.role for message in provider.requests[1][0]] == ["system", "user"]
    assert provider.requests[1][0][-1].content == new_task
    assert output == [
        "旧回答",
        "停止原因: completed",
        "会话历史已清空",
        "新回答",
        "停止原因: completed",
    ]


def test_cli_trim_policy_reports_context_diagnostic_without_message_content(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path)
    registry = create_read_only_registry(workspace)
    budget = measure_context_bytes(
        [
            Message(role="system", content=DEFAULT_SYSTEM_PROMPT),
            Message(role="user", content="new"),
        ],
        registry.schemas(),
    )
    provider = FakeProvider(
        [
            ModelResponse(text="first answer", finish_reason="stop"),
            ModelResponse(text="second answer", finish_reason="stop"),
        ]
    )
    inputs = iter(["old", "new", "/exit"])
    output: list[str] = []

    exit_code = main(
        [
            "--chat",
            "--workspace",
            str(tmp_path),
            "--read-only",
            "--max-steps",
            "1",
            "--max-context-bytes",
            str(budget),
            "--context-policy",
            "trim",
            "--hide-tool-events",
        ],
        provider=provider,
        input_fn=lambda _: next(inputs),
        output_fn=output.append,
    )

    assert exit_code == 0
    diagnostics = [message for message in output if message.startswith("上下文提示：")]
    assert diagnostics == [
        "上下文提示：已移除 1 个较早的完整任务，仅影响本次请求上下文；完整历史仍保留。"
    ]
    assert "old" not in diagnostics[0]
    assert "first answer" not in diagnostics[0]


def test_cli_trim_policy_reports_when_remaining_context_still_exceeds_budget(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path)
    registry = create_read_only_registry(workspace)
    budget = measure_context_bytes(
        [
            Message(role="system", content=DEFAULT_SYSTEM_PROMPT),
            Message(role="user", content="x"),
        ],
        registry.schemas(),
    )
    provider = FakeProvider([ModelResponse(text="old done", finish_reason="stop")])
    inputs = iter(["x", "current task"])
    output: list[str] = []
    errors: list[str] = []

    exit_code = main(
        [
            "--chat",
            "--workspace",
            str(tmp_path),
            "--read-only",
            "--max-context-bytes",
            str(budget),
            "--context-policy",
            "trim",
        ],
        provider=provider,
        input_fn=lambda _: next(inputs),
        output_fn=output.append,
        error_fn=errors.append,
    )

    assert exit_code == 1
    assert len(provider.requests) == 1
    diagnostics = [message for message in output if message.startswith("上下文提示：")]
    assert diagnostics == [
        "上下文提示：已移除 1 个较早的完整任务，仅影响本次请求上下文；"
        "完整历史仍保留；裁剪后仍超出字节预算，未发送请求。"
    ]
    assert errors[0].startswith("停止原因: context_limit")


def test_cli_chat_eof_and_empty_input_do_not_call_provider(tmp_path: Path) -> None:
    provider = FakeProvider([])
    prompts: list[str] = []

    def end_input(prompt: str) -> str:
        prompts.append(prompt)
        raise EOFError

    exit_code = main(
        ["--chat", "--workspace", str(tmp_path)],
        provider=provider,
        input_fn=end_input,
    )

    assert exit_code == 0
    assert len(prompts) == 1
    assert provider.requests == []


def test_cli_chat_abnormal_task_stops_session_without_consuming_next_task(
    tmp_path: Path,
) -> None:
    provider = FakeProvider([ModelResponse(text="截断内容", finish_reason="length")])
    inputs = iter(["第一个任务", "不应执行的第二个任务"])
    errors: list[str] = []

    exit_code = main(
        ["--chat", "--workspace", str(tmp_path)],
        provider=provider,
        input_fn=lambda _: next(inputs),
        error_fn=errors.append,
    )

    assert exit_code == 1
    assert errors == ["停止原因: length"]
    assert len(provider.requests) == 1


def test_cli_chat_keyboard_interrupt_while_waiting_returns_130(tmp_path: Path) -> None:
    def interrupt(_: str) -> str:
        raise KeyboardInterrupt

    errors: list[str] = []
    exit_code = main(
        ["--chat", "--workspace", str(tmp_path)],
        provider=FakeProvider([]),
        input_fn=interrupt,
        error_fn=errors.append,
    )

    assert exit_code == 130
    assert errors == ["停止原因: interrupted"]


def test_cli_does_not_parse_read_file_json_as_execution_status(tmp_path: Path) -> None:
    file_content = '{"status":"created","path":"never-created.py","exit_code":0}'
    (tmp_path / "note.txt").write_text(file_content, encoding="utf-8")
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_read_json",
                        name="read_file",
                        arguments={"path": "note.txt"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="已读取文件。", finish_reason="stop"),
        ]
    )
    output: list[str] = []

    exit_code = main(
        ["读取 note.txt", "--workspace", str(tmp_path), "--read-only"],
        provider=provider,
        output_fn=output.append,
    )

    assert exit_code == 0
    assert output[1] == "工具结果 read_file: 已返回"
    assert "never-created.py" not in output[1]
    assert provider.requests[1][0][-1].content == file_content


def test_cli_handles_large_integer_read_file_content_without_aborting(tmp_path: Path) -> None:
    file_content = "9" * 5_000
    (tmp_path / "numbers.txt").write_text(file_content, encoding="utf-8")
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_read_large",
                        name="read_file",
                        arguments={"path": "numbers.txt"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="已读取长数字文本。", finish_reason="stop"),
        ]
    )

    output: list[str] = []
    exit_code = main(
        ["读取 numbers.txt", "--workspace", str(tmp_path), "--read-only"],
        provider=provider,
        output_fn=output.append,
    )

    assert exit_code == 0
    assert output[-2:] == ["已读取长数字文本。", "停止原因: completed"]
    assert provider.requests[1][0][-1].content == file_content


def test_cli_runs_injected_provider_and_passes_workspace_and_task(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Project contents", encoding="utf-8")
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_readme",
                        name="read_file",
                        arguments={"path": "README.md"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="Project contents", finish_reason="stop"),
        ]
    )
    output: list[str] = []
    errors: list[str] = []

    exit_code = main(
        [
            "Read README.md",
            "--workspace",
            str(tmp_path),
            "--read-only",
            "--max-steps",
            "3",
        ],
        provider=provider,
        output_fn=output.append,
        error_fn=errors.append,
    )

    assert exit_code == 0
    assert output[0].startswith("工具调用: read_file (call_readme)")
    assert output[1] == "工具结果 read_file: 已返回"
    assert output[2:] == ["Project contents", "停止原因: completed"]
    assert errors == []
    assert provider.requests[0][0][0].role == "system"
    assert provider.requests[0][0][1].content == "Read README.md"


def test_cli_hiding_tool_events_preserves_execution_and_history(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("Project contents", encoding="utf-8")

    def run(flags: tuple[str, ...]) -> tuple[FakeProvider, list[str], int]:
        provider = FakeProvider(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(
                            id="call_hidden_read",
                            name="read_file",
                            arguments={"path": "note.txt"},
                        )
                    ],
                    finish_reason="tool_calls",
                ),
                ModelResponse(text="文件已读取。", finish_reason="stop"),
            ]
        )
        output: list[str] = []
        exit_code = main(
            ["读取 note.txt", "--workspace", str(tmp_path), "--read-only", *flags],
            provider=provider,
            output_fn=output.append,
        )
        return provider, output, exit_code

    shown_provider, shown_output, shown_exit_code = run(())
    hidden_provider, hidden_output, hidden_exit_code = run(("--hide-tool-events",))
    explicit_provider, explicit_output, explicit_exit_code = run(("--show-tool-events",))

    assert shown_exit_code == hidden_exit_code == explicit_exit_code == 0
    assert shown_provider.requests == hidden_provider.requests == explicit_provider.requests
    assert shown_output == explicit_output
    assert shown_output[0].startswith("工具调用: read_file")
    assert shown_output[1] == "工具结果 read_file: 已返回"
    assert hidden_output == ["文件已读取。", "停止原因: completed"]


def test_cli_hiding_tool_events_keeps_approval_and_errors_visible(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_hidden_write",
                        name="write_file",
                        arguments={"path": "denied.txt", "content": "content"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="写入已拒绝。", finish_reason="stop"),
        ]
    )
    output: list[str] = []

    exit_code = main(
        ["Create denied.txt", "--workspace", str(tmp_path), "--hide-tool-events"],
        provider=provider,
        input_fn=lambda _: "",
        output_fn=output.append,
    )

    assert exit_code == 0
    assert not (tmp_path / "denied.txt").exists()
    assert not any(message.startswith("工具调用:") for message in output)
    assert any(message.startswith("待审批操作:") for message in output)
    assert any("审批结果: 已拒绝" in message for message in output)
    assert any(
        message.startswith("工具结果 write_file:") and "错误" in message for message in output
    )
    assert output[-2:] == ["写入已拒绝。", "停止原因: completed"]


def test_cli_prompts_for_task_when_argument_is_omitted(tmp_path: Path) -> None:
    provider = FakeProvider([ModelResponse(text="Done", finish_reason="stop")])
    prompts: list[str] = []
    output: list[str] = []

    exit_code = main(
        ["--workspace", str(tmp_path), "--no-chat", "--read-only"],
        provider=provider,
        input_fn=lambda prompt: prompts.append(prompt) or "Inspect the workspace",
        output_fn=output.append,
    )

    assert exit_code == 0
    assert prompts == ["任务: "]
    assert output == ["Done", "停止原因: completed"]


def test_cli_reports_max_steps_without_requesting_another_response(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_list",
                        name="list_files",
                        arguments={},
                    )
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    errors: list[str] = []

    exit_code = main(
        [
            "Inspect files",
            "--workspace",
            str(tmp_path),
            "--read-only",
            "--max-steps",
            "1",
        ],
        provider=provider,
        error_fn=errors.append,
    )

    assert exit_code == 1
    assert errors == ["停止原因: max_steps"]
    assert len(provider.requests) == 1


def test_cli_rejects_empty_interactive_task(tmp_path: Path) -> None:
    errors: list[str] = []

    exit_code = main(
        ["--workspace", str(tmp_path), "--no-chat", "--read-only"],
        provider=FakeProvider([]),
        input_fn=lambda _: "  ",
        error_fn=errors.append,
    )

    assert exit_code == 2
    assert errors == ["错误: task 不能为空"]


def test_cli_handles_keyboard_interrupt_during_input(tmp_path: Path) -> None:
    errors: list[str] = []

    def interrupt(_: str) -> str:
        raise KeyboardInterrupt

    exit_code = main(
        ["--workspace", str(tmp_path), "--no-chat", "--read-only"],
        provider=FakeProvider([]),
        input_fn=interrupt,
        error_fn=errors.append,
    )

    assert exit_code == 130
    assert errors == ["停止原因: interrupted"]


def test_cli_handles_keyboard_interrupt_during_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors: list[str] = []

    def interrupt_workspace(_: str | Path) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("coding_agent.cli.Workspace", interrupt_workspace)

    exit_code = main(
        ["Inspect files", "--workspace", str(tmp_path), "--read-only"],
        provider=FakeProvider([]),
        error_fn=errors.append,
    )

    assert exit_code == 130
    assert errors == ["停止原因: interrupted"]


def test_cli_handles_keyboard_interrupt_during_output(tmp_path: Path) -> None:
    provider = FakeProvider([ModelResponse(text="Done", finish_reason="stop")])
    errors: list[str] = []

    def interrupt_output(_: str) -> None:
        raise KeyboardInterrupt

    exit_code = main(
        ["Finish the task", "--workspace", str(tmp_path), "--read-only"],
        provider=provider,
        output_fn=interrupt_output,
        error_fn=errors.append,
    )

    assert exit_code == 130
    assert errors == ["停止原因: interrupted"]


@pytest.mark.parametrize(
    ("flags", "expected_names"),
    [
        (
            (),
            ["list_files", "read_file", "read_document", "write_file", "edit_file", "run_command"],
        ),
        (("--no-write",), ["list_files", "read_file", "read_document", "run_command"]),
        (("--no-exec",), ["list_files", "read_file", "read_document", "write_file", "edit_file"]),
        (("--read-only",), ["list_files", "read_file", "read_document"]),
    ],
)
def test_cli_permission_switches_control_exposed_tool_schemas(
    tmp_path: Path,
    flags: tuple[str, ...],
    expected_names: list[str],
) -> None:
    provider = FakeProvider([ModelResponse(text="完成", finish_reason="stop")])

    exit_code = main(
        ["Inspect", "--workspace", str(tmp_path), *flags],
        provider=provider,
    )

    assert exit_code == 0
    assert [schema["function"]["name"] for schema in provider.requests[0][1]] == expected_names


def test_cli_read_only_tools_reject_model_requested_write(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_write",
                        name="write_file",
                        arguments={"path": "created.txt", "content": "content"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="Write was unavailable.", finish_reason="stop"),
        ]
    )
    approval_prompts: list[str] = []

    exit_code = main(
        [
            "Create a file",
            "--workspace",
            str(tmp_path),
            "--read-only",
            "--max-steps",
            "3",
        ],
        provider=provider,
        input_fn=lambda prompt: approval_prompts.append(prompt) or "y",
    )

    assert exit_code == 0
    assert approval_prompts == []
    assert not (tmp_path / "created.txt").exists()
    assert [schema["function"]["name"] for schema in provider.requests[0][1]] == [
        "list_files",
        "read_file",
        "read_document",
    ]
    tool_message = next(message for message in provider.requests[1][0] if message.role == "tool")
    assert tool_message.tool_call_id == "call_write"
    assert "Unknown tool" in tool_message.content


def test_cli_executes_approved_command_with_configured_limits(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_run",
                        name="run_command",
                        arguments={
                            "argv": ["python", "-c", "print('command output')"],
                        },
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="命令已完成。", finish_reason="stop"),
        ]
    )
    output: list[str] = []

    exit_code = main(
        [
            "Run a short Python command",
            "--workspace",
            str(tmp_path),
            "--allow-exec",
            "--no-write",
            "--command-timeout",
            "2",
            "--command-output-limit",
            "1024",
            "--max-steps",
            "3",
        ],
        provider=provider,
        input_fn=lambda _: "y",
        output_fn=output.append,
    )

    assert exit_code == 0
    assert [schema["function"]["name"] for schema in provider.requests[0][1]] == [
        "list_files",
        "read_file",
        "read_document",
        "run_command",
    ]
    tool_message = next(message for message in provider.requests[1][0] if message.role == "tool")
    payload = json.loads(tool_message.content)
    assert payload["status"] == "completed"
    assert payload["exit_code"] == 0
    assert payload["stdout"].strip() == "command output"
    assert any("Timeout seconds: 2.0" in message for message in output)
    assert any("Combined output limit bytes: 1024" in message for message in output)
    assert output[-2:] == ["命令已完成。", "停止原因: completed"]


def test_cli_does_not_expose_run_command_without_allow_exec(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_run",
                        name="run_command",
                        arguments={"argv": ["python", "-c", "print('not run')"]},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="命令不可用。", finish_reason="stop"),
        ]
    )

    exit_code = main(
        ["Run a command", "--workspace", str(tmp_path), "--no-write", "--no-exec"],
        provider=provider,
        input_fn=lambda _: "y",
    )

    assert exit_code == 0
    assert [schema["function"]["name"] for schema in provider.requests[0][1]] == [
        "list_files",
        "read_file",
        "read_document",
    ]
    tool_message = next(message for message in provider.requests[1][0] if message.role == "tool")
    assert "Unknown tool" in tool_message.content


def test_cli_rejects_invalid_command_limits_before_provider_call(tmp_path: Path) -> None:
    provider = FakeProvider([])
    errors: list[str] = []

    exit_code = main(
        [
            "Run a command",
            "--workspace",
            str(tmp_path),
            "--allow-exec",
            "--no-write",
            "--command-timeout",
            "61",
        ],
        provider=provider,
        error_fn=errors.append,
    )

    assert exit_code == 2
    assert provider.requests == []
    assert "timeout_seconds" in errors[0]


def test_cli_default_input_rejects_redirected_command_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_run",
                        name="run_command",
                        arguments={
                            "argv": [
                                "python",
                                "-c",
                                "open('not-created.txt', 'w').write('unexpected')",
                            ]
                        },
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="命令已拒绝。", finish_reason="stop"),
        ]
    )
    redirected_input = StringIO("y\n")
    output: list[str] = []
    monkeypatch.setattr(sys, "stdin", redirected_input)

    exit_code = main(
        ["Run a command", "--workspace", str(tmp_path), "--allow-exec"],
        provider=provider,
        output_fn=output.append,
    )

    assert exit_code == 0
    assert redirected_input.tell() == 0
    assert not (tmp_path / "not-created.txt").exists()
    assert "审批结果: 已拒绝（非交互输入不能用于审批）" in output
    tool_message = next(message for message in provider.requests[1][0] if message.role == "tool")
    assert json.loads(tool_message.content)["status"] == "denied"


def test_cli_defaults_to_enabled_tools_but_still_requires_approval(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_default_write",
                        name="write_file",
                        arguments={"path": "not-created.txt", "content": "content"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="写入已拒绝。", finish_reason="stop"),
        ]
    )
    output: list[str] = []

    exit_code = main(
        ["Create a file", "--workspace", str(tmp_path)],
        provider=provider,
        input_fn=lambda _: "",
        output_fn=output.append,
    )

    assert exit_code == 0
    assert not (tmp_path / "not-created.txt").exists()
    assert any("审批结果: 已拒绝" in message for message in output)
    assert [schema["function"]["name"] for schema in provider.requests[0][1]] == [
        "list_files",
        "read_file",
        "read_document",
        "write_file",
        "edit_file",
        "run_command",
    ]


def test_cli_approves_write_and_edit_as_separate_operations(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_write",
                        name="write_file",
                        arguments={"path": "program.py", "content": "value = 1\n"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_edit",
                        name="edit_file",
                        arguments={
                            "path": "program.py",
                            "old_text": "value = 1",
                            "new_text": "value = 2",
                        },
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="文件已创建并修改。", finish_reason="stop"),
        ]
    )
    answers = iter(["y", "yes"])
    prompts: list[str] = []
    output: list[str] = []

    exit_code = main(
        [
            "Create and update program.py",
            "--workspace",
            str(tmp_path),
            "--allow-write",
            "--no-exec",
            "--max-steps",
            "4",
        ],
        provider=provider,
        input_fn=lambda prompt: prompts.append(prompt) or next(answers),
        output_fn=output.append,
    )

    assert exit_code == 0
    assert prompts == ["批准本次操作？[y/N]: ", "批准本次操作？[y/N]: "]
    assert (tmp_path / "program.py").read_text(encoding="utf-8") == "value = 2\n"
    assert [schema["function"]["name"] for schema in provider.requests[0][1]] == [
        "list_files",
        "read_file",
        "read_document",
        "write_file",
        "edit_file",
    ]
    first_tool_message = next(
        message for message in provider.requests[1][0] if message.tool_call_id == "call_write"
    )
    second_tool_message = next(
        message for message in provider.requests[2][0] if message.tool_call_id == "call_edit"
    )
    assert json.loads(first_tool_message.content)["status"] == "created"
    assert json.loads(second_tool_message.content)["status"] == "edited"
    assert any("Path: program.py" in message for message in output)
    assert output[-2:] == ["文件已创建并修改。", "停止原因: completed"]


def test_cli_default_input_rejects_redirected_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_write",
                        name="write_file",
                        arguments={"path": "redirected.txt", "content": "content"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="操作未执行。", finish_reason="stop"),
        ]
    )
    redirected_input = StringIO("y\n")
    output: list[str] = []
    monkeypatch.setattr(sys, "stdin", redirected_input)

    exit_code = main(
        [
            "Create redirected.txt",
            "--workspace",
            str(tmp_path),
            "--allow-write",
            "--no-exec",
        ],
        provider=provider,
        output_fn=output.append,
    )

    assert exit_code == 0
    assert redirected_input.tell() == 0
    assert not (tmp_path / "redirected.txt").exists()
    assert "审批结果: 已拒绝（非交互输入不能用于审批）" in output
    tool_message = next(message for message in provider.requests[1][0] if message.role == "tool")
    assert json.loads(tool_message.content)["status"] == "denied"


@pytest.mark.parametrize("answer", ["", "n"])
def test_cli_rejected_or_empty_approval_does_not_write_file(
    tmp_path: Path,
    answer: str,
) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_write",
                        name="write_file",
                        arguments={"path": "denied.txt", "content": "content"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="操作未执行。", finish_reason="stop"),
        ]
    )

    exit_code = main(
        [
            "Create denied.txt",
            "--workspace",
            str(tmp_path),
            "--allow-write",
            "--no-exec",
        ],
        provider=provider,
        input_fn=lambda _: answer,
    )

    assert exit_code == 0
    assert not (tmp_path / "denied.txt").exists()
    tool_message = next(message for message in provider.requests[1][0] if message.role == "tool")
    assert json.loads(tool_message.content)["status"] == "denied"


def test_cli_eof_during_approval_denies_operation(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_write",
                        name="write_file",
                        arguments={"path": "denied.txt", "content": "content"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="操作未执行。", finish_reason="stop"),
        ]
    )
    output: list[str] = []

    def end_input(_: str) -> str:
        raise EOFError

    exit_code = main(
        [
            "Create denied.txt",
            "--workspace",
            str(tmp_path),
            "--allow-write",
            "--no-exec",
        ],
        provider=provider,
        input_fn=end_input,
        output_fn=output.append,
    )

    assert exit_code == 0
    assert not (tmp_path / "denied.txt").exists()
    assert "审批结果: 已拒绝（无法读取输入）" in output
    tool_message = next(message for message in provider.requests[1][0] if message.role == "tool")
    assert json.loads(tool_message.content)["status"] == "denied"


def test_cli_keyboard_interrupt_during_approval_stops_without_writing(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_write",
                        name="write_file",
                        arguments={"path": "interrupted.txt", "content": "content"},
                    )
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    errors: list[str] = []

    def interrupt(_: str) -> str:
        raise KeyboardInterrupt

    exit_code = main(
        [
            "Create interrupted.txt",
            "--workspace",
            str(tmp_path),
            "--allow-write",
            "--no-exec",
        ],
        provider=provider,
        input_fn=interrupt,
        error_fn=errors.append,
    )

    assert exit_code == 130
    assert errors == ["停止原因: interrupted"]
    assert not (tmp_path / "interrupted.txt").exists()
    assert len(provider.requests) == 1


def test_cli_keyboard_interrupt_during_edit_cleans_up_and_returns_130(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_edit",
                        name="edit_file",
                        arguments={
                            "path": "notes.txt",
                            "old_text": "before",
                            "new_text": "after",
                        },
                    )
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    errors: list[str] = []

    def interrupt_fsync(_: int) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("coding_agent.file_tools.os.fsync", interrupt_fsync)

    exit_code = main(
        [
            "Edit notes.txt",
            "--workspace",
            str(tmp_path),
            "--allow-write",
            "--no-exec",
        ],
        provider=provider,
        input_fn=lambda _: "y",
        error_fn=errors.append,
    )

    assert exit_code == 130
    assert errors == ["停止原因: interrupted"]
    assert target.read_text(encoding="utf-8") == "before"
    assert list(tmp_path.glob(".notes.txt.*.tmp")) == []
    assert len(provider.requests) == 1
