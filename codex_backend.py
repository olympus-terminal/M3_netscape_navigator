"""Transformation routing for a local API or the isolated Codex CLI backend."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import tempfile
from contextlib import suppress
from pathlib import Path

from config import (
    CODEX_BIN, CODEX_IGNORE_USER_CONFIG, CODEX_MODEL,
    LLM_BACKEND, LLM_MODEL, LLM_DEPLOYMENT,
)

log = logging.getLogger(__name__)

_DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "shell_snapshot",
    "apps",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "computer_use",
    "image_generation",
    "plugins",
    "remote_plugin",
    "tool_suggest",
    "workspace_dependencies",
    "skill_mcp_dependency_install",
    "auth_elicitation",
    "tool_call_mcp_elicitation",
    "hooks",
    "multi_agent",
)


class CodexExecError(RuntimeError):
    """Raised when a Codex subprocess does not produce a valid final result."""


def _safe_diagnostic(text: str, limit: int = 4000) -> str:
    """Trim subprocess diagnostics and redact common inline credential forms."""
    text = text[-limit:]
    patterns = (
        r"(?i)(authorization\s*:\s*bearer\s+)[^\s]+",
        r'''(?i)(["']?(?:api[_-]?key|access[_-]?token|auth[_-]?token|token)["']?\s*[=:]\s*["']?)[^"'\s,}]+''',
    )
    for pattern in patterns:
        text = re.sub(pattern, r"\1[REDACTED]", text)
    return text.strip()


async def _terminate_process_group(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return

    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        return

    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
        return
    except asyncio.TimeoutError:
        pass

    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        return
    await proc.wait()


def _validate_events(stdout: str) -> None:
    events = []
    try:
        for line in stdout.splitlines():
            if line.strip():
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise CodexExecError("Codex emitted a non-object JSONL telemetry event")
                events.append(event)
    except json.JSONDecodeError as exc:
        raise CodexExecError("Codex emitted invalid JSONL telemetry") from exc

    failed = [event for event in events if event.get("type") in {"error", "turn.failed"}]
    if failed:
        messages = [str(event.get("message") or event.get("error") or event.get("type")) for event in failed]
        detail = _safe_diagnostic("; ".join(messages[-3:]))
        raise CodexExecError(f"Codex reported a failed turn: {detail}")
    for event in events:
        event_type = event.get("type")
        if isinstance(event_type, str) and event_type.startswith("item."):
            item = event.get("item")
            item_type = item.get("type") if isinstance(item, dict) else None
            if item_type not in {"agent_message", "reasoning"}:
                raise CodexExecError(
                    f"Codex used a disallowed tool/action item: {item_type or 'unknown'}"
                )
    if not any(event.get("type") == "turn.completed" for event in events):
        raise CodexExecError("Codex exited without a turn.completed event")


async def run_codex(
    instruction: str,
    payload: str,
    *,
    timeout_seconds: int,
    output_schema: Path | None = None,
) -> str:
    """Run one isolated Codex transform and return its final agent message.

    ``instruction`` is trusted application text passed as the CLI prompt.
    ``payload`` may contain untrusted scraped content and is sent through stdin,
    which Codex presents as a separate input block.
    """
    if LLM_BACKEND == "openai_compatible":
        from compatible_backend import ModelError, transform
        try:
            return await transform(
                instruction, payload, timeout_seconds=timeout_seconds,
                output_schema=output_schema,
            )
        except ModelError as exc:
            raise CodexExecError(str(exc)) from exc
    if not instruction.strip():
        raise ValueError("Codex instruction must not be empty")
    if timeout_seconds <= 0:
        raise ValueError("Codex timeout must be positive")
    if output_schema is not None:
        output_schema = output_schema.resolve()
        if not output_schema.is_file():
            raise CodexExecError(f"Codex output schema not found: {output_schema}")

    with tempfile.TemporaryDirectory(prefix="ai-digest-codex-") as temp_dir:
        run_dir = Path(temp_dir)
        final_path = run_dir / "final-message.txt"
        command = [
            CODEX_BIN,
            "exec",
            "--ephemeral",
            "--strict-config",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--skip-git-repo-check",
            "-C",
            str(run_dir),
            "-c",
            'approval_policy="never"',
            "-c",
            'web_search="disabled"',
        ]
        if CODEX_IGNORE_USER_CONFIG:
            command.append("--ignore-user-config")
        for feature in _DISABLED_FEATURES:
            command.extend(("--disable", feature))
        command.extend(("--json", "-o", str(final_path)))
        if CODEX_MODEL and CODEX_MODEL.lower() != "default":
            command.extend(("--model", CODEX_MODEL))
        if output_schema is not None:
            command.extend(("--output-schema", str(output_schema)))
        command.append(instruction)

        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(run_dir),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=(os.name == "posix"),
            )
        except FileNotFoundError as exc:
            raise CodexExecError(f"Codex executable not found: {CODEX_BIN}") from exc

        communicate = asyncio.create_task(proc.communicate(payload.encode("utf-8")))
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                asyncio.shield(communicate), timeout=timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            await _terminate_process_group(proc)
            stdout_bytes, stderr_bytes = await communicate
            detail = _safe_diagnostic(stderr_bytes.decode("utf-8", errors="replace"))
            raise CodexExecError(
                f"Codex timed out after {timeout_seconds}s"
                + (f": {detail}" if detail else "")
            ) from exc
        except BaseException:
            await _terminate_process_group(proc)
            if not communicate.done():
                communicate.cancel()
            with suppress(asyncio.CancelledError):
                await communicate
            raise

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            detail = _safe_diagnostic(
                "\n".join(part for part in (stderr, stdout) if part.strip())
            )
            raise CodexExecError(
                f"Codex exited with status {proc.returncode}"
                + (f": {detail}" if detail else "")
            )

        _validate_events(stdout)
        if not final_path.is_file():
            raise CodexExecError("Codex produced no final-message file")
        final = final_path.read_text(encoding="utf-8").strip()
        if not final:
            raise CodexExecError("Codex returned an empty final message")
        return final


async def preflight_codex(timeout_seconds: int = 15) -> tuple[str, str]:
    """Check the configured model endpoint or CLI and saved authentication."""

    if LLM_BACKEND == "openai_compatible":
        from compatible_backend import ModelError, preflight
        try:
            return await preflight(timeout_seconds)
        except ModelError as exc:
            raise CodexExecError(str(exc)) from exc

    async def run_check(*args: str) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                CODEX_BIN,
                *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=(os.name == "posix"),
            )
        except FileNotFoundError as exc:
            raise CodexExecError(f"Codex executable not found: {CODEX_BIN}") from exc
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            await _terminate_process_group(proc)
            raise CodexExecError(f"Codex {' '.join(args)} timed out") from exc
        detail = (stdout + stderr).decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            raise CodexExecError(
                f"Codex {' '.join(args)} failed with status {proc.returncode}: "
                f"{_safe_diagnostic(detail)}"
            )
        return detail

    version = await run_check("--version")
    auth = await run_check("login", "status")
    log.info("Codex preflight passed: %s", version)
    return version, auth


def backend_identity() -> dict:
    """Configured identity; local calls also verify the response model name."""
    return {
        "backend": LLM_BACKEND,
        "model": LLM_MODEL if LLM_BACKEND == "openai_compatible" else CODEX_MODEL or "cli_default",
        "deployment": LLM_DEPLOYMENT or None,
    }
