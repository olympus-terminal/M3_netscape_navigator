#!/usr/bin/env python3
"""
AI Research Agent — persistent daemon that crawls, judges, and narrates
bleeding-edge AI research into daily audio digests.

Architecture:
  [Crawlers] --async--> [Queue] --> [Codex Judge] --> [Codex Writer] --> [TTS]

  Crawlers run concurrently across domains with human-paced delays.
  The LLM stays saturated — always has the next item ready from the queue.
  Once enough content is judged, the script writer composes a narration
  and TTS synthesizes the MP3.

Usage:
  python agent.py                # run one crawl+judge+digest cycle
  python agent.py --daemon       # run continuously, one cycle per day
  python agent.py --install-service  # install/update automatic user startup
  python agent.py --crawl-only   # crawl and judge, skip TTS
  python agent.py --digest-only  # compose and synthesize from last judged set
"""

import argparse
import asyncio
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from codex_backend import CodexExecError, preflight_codex
from config import (
    BASE_DIR,
    CODEX_JUDGE_WORKERS,
    LOG_FILE,
    OUTPUT_DIR,
    RAW_DIR,
    STATE_FILE,
    TARGET_WORD_COUNT,
    TTS_ENABLED,
    WORDS_PER_MINUTE,
)
from crawlers import run_all_crawlers
from digest import (
    compose_digest_script,
    preflight_tts,
    save_digest_metadata,
    synthesize_audio,
    write_digest_source_notes,
)
from judge import judge_loop
from queue_manager import ContentQueue

# ─── Logging ─────────────────────────────────────────────────────────────────

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)-5s] %(name)-12s: %(message)s"
    datefmt = "%H:%M:%S"

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)

    file_handler = logging.FileHandler(LOG_FILE, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logging.getLogger().addHandler(file_handler)


log = logging.getLogger("agent")

_DAILY_INTERVAL = timedelta(days=1)
_RETRY_INTERVAL = timedelta(hours=1)
_SCHEDULE_RECHECK_SECONDS = 60.0


# ─── Startup installation ───────────────────────────────────────────────────

_SERVICE_NAME = "ai-research-agent.service"
_DESKTOP_AUTOSTART_NAME = "ai-research-agent.desktop"
_SYSTEMCTL_TIMEOUT_SECONDS = 180


def _absolute_lexical_path(path: str | Path) -> Path:
    """Return an absolute path without resolving virtualenv symlinks."""
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _startup_config_home(config_home: str | Path | None = None) -> Path:
    if config_home is not None:
        return _absolute_lexical_path(config_home)
    configured_home = os.environ.get("XDG_CONFIG_HOME")
    if configured_home and Path(configured_home).is_absolute():
        return _absolute_lexical_path(configured_home)
    return _absolute_lexical_path(Path.home() / ".config")


def _systemd_quote(value: str | Path, *, exec_argument: bool = False) -> str:
    """Quote one literal path for a systemd unit without shell semantics."""
    text = os.fspath(value)
    if any(character in text for character in ("\0", "\n", "\r")):
        raise ValueError("systemd paths cannot contain NUL or line breaks")
    text = text.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    if exec_argument:
        text = text.replace("$", "$$")
    return f'"{text}"'


def _systemd_path_value(value: str | Path) -> str:
    """Encode a literal absolute path for path-only systemd settings."""
    text = os.fspath(value)
    if any(character in text for character in ("\0", "\n", "\r")):
        raise ValueError("systemd paths cannot contain NUL or line breaks")

    encoded = []
    for character in text:
        if character == "\\":
            encoded.append("\\\\")
        elif character == "%":
            encoded.append("%%")
        elif character == " ":
            encoded.append("\\x20")
        elif character == "\t":
            encoded.append("\\t")
        elif character == '"':
            encoded.append("\\x22")
        elif character == "'":
            encoded.append("\\x27")
        else:
            encoded.append(character)
    return "".join(encoded)


def _desktop_quoted_argument(value: str) -> str:
    """Encode one quoted Exec argument through both desktop escaping layers."""
    text = value
    if any(character in text for character in ("\0", "\n", "\r")):
        raise ValueError("desktop Exec arguments cannot contain NUL or line breaks")
    exec_escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("`", "\\`")
        .replace("$", "\\$")
    )
    escaped = exec_escaped.replace("\\", "\\\\").replace("%", "%%")
    return f'"{escaped}"'


def _render_startup_unit(
    project_dir: str | Path,
    python_executable: str | Path,
) -> str:
    project_path = _absolute_lexical_path(project_dir)
    python_path = _absolute_lexical_path(python_executable)
    agent_path = project_path / "agent.py"
    return (
        "[Unit]\n"
        "Description=Daily Codex AI research audio agent\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n"
        "StartLimitIntervalSec=0\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"WorkingDirectory={_systemd_path_value(project_path)}\n"
        f"ExecStart={_systemd_quote(python_path, exec_argument=True)} "
        f"{_systemd_quote(agent_path, exec_argument=True)} --daemon\n"
        "Environment=PYTHONUNBUFFERED=1\n"
        "Restart=always\n"
        "RestartSec=300\n"
        "KillSignal=SIGTERM\n"
        "KillMode=mixed\n"
        "TimeoutStopSec=120\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def _render_desktop_autostart(systemctl_path: str | Path) -> str:
    executable = shlex.quote(str(_absolute_lexical_path(systemctl_path)))
    command = (
        f"{executable} --user daemon-reload && "
        f"{executable} --user start {_SERVICE_NAME}"
    )
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=AI Research Agent\n"
        "Comment=Start the daily research digest after the encrypted home is unlocked\n"
        f"Exec=/bin/sh -c {_desktop_quoted_argument(command)}\n"
        "Terminal=false\n"
        "X-GNOME-Autostart-enabled=true\n"
    )


def _write_text_if_changed_atomic(path: Path, content: str) -> bool:
    """Atomically publish UTF-8 text with mode 0644; return whether it changed."""
    try:
        content_matches = path.read_text(encoding="utf-8") == content
        mode_is_safe = (
            not path.is_symlink() and (path.stat().st_mode & 0o777) == 0o644
        )
        if content_matches and mode_is_safe:
            return False
    except FileNotFoundError:
        pass

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return True


def _required_executable(name: str, supplied_path: str | Path | None) -> Path:
    discovered = (
        os.fspath(supplied_path) if supplied_path is not None else shutil.which(name)
    )
    if discovered is None:
        raise FileNotFoundError(f"{name} was not found on PATH")
    executable = _absolute_lexical_path(discovered)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"{name} is not executable: {executable}")
    return executable


def _validate_startup_unit(unit_content: str, systemd_analyze: Path):
    """Ask the installed systemd parser to validate a staged unit."""
    with tempfile.TemporaryDirectory(prefix="ai_research_agent_unit_") as temp:
        staged_unit = Path(temp) / _SERVICE_NAME
        staged_unit.write_text(unit_content, encoding="utf-8")
        staged_unit.chmod(0o644)
        subprocess.run(
            [str(systemd_analyze), "--user", "verify", str(staged_unit)],
            check=True,
            timeout=_SYSTEMCTL_TIMEOUT_SECONDS,
        )


def install_startup_service(
    *,
    project_dir: str | Path | None = None,
    python_executable: str | Path | None = None,
    config_home: str | Path | None = None,
    systemctl_path: str | Path | None = None,
    systemd_analyze_path: str | Path | None = None,
) -> tuple[Path, Path, bool]:
    """Install or refresh the user service and encrypted-home login fallback."""
    project_path = _absolute_lexical_path(
        project_dir if project_dir is not None else Path(__file__).resolve().parent
    )
    python_path = _absolute_lexical_path(
        python_executable if python_executable is not None else sys.executable
    )
    agent_path = project_path / "agent.py"
    if not agent_path.is_file():
        raise FileNotFoundError(f"Agent entry point not found: {agent_path}")
    if not python_path.is_file():
        raise FileNotFoundError(f"Python interpreter not found: {python_path}")
    if not os.access(python_path, os.X_OK):
        raise PermissionError(f"Python interpreter is not executable: {python_path}")

    systemctl = _required_executable("systemctl", systemctl_path)
    systemd_analyze = _required_executable("systemd-analyze", systemd_analyze_path)

    config_root = _startup_config_home(config_home)

    service_path = config_root / "systemd" / "user" / _SERVICE_NAME
    desktop_path = config_root / "autostart" / _DESKTOP_AUTOSTART_NAME
    unit_content = _render_startup_unit(project_path, python_path)
    desktop_content = _render_desktop_autostart(systemctl)
    _validate_startup_unit(unit_content, systemd_analyze)

    # Publish the fallback first. The service unit is the primary startup
    # contract and is written last so a fallback-write failure cannot replace it.
    desktop_changed = _write_text_if_changed_atomic(
        desktop_path,
        desktop_content,
    )
    service_changed = _write_text_if_changed_atomic(
        service_path,
        unit_content,
    )

    def systemctl_run(*arguments: str):
        subprocess.run(
            [str(systemctl), "--user", *arguments],
            check=True,
            timeout=_SYSTEMCTL_TIMEOUT_SECONDS,
        )

    systemctl_run("daemon-reload")
    systemctl_run("enable", _SERVICE_NAME)
    systemctl_run("restart" if service_changed else "start", _SERVICE_NAME)
    systemctl_run("is-enabled", "--quiet", _SERVICE_NAME)
    systemctl_run("is-active", "--quiet", _SERVICE_NAME)

    return service_path, desktop_path, service_changed or desktop_changed


# ─── State persistence ──────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"seen_urls": [], "last_run": None, "runs": 0}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _local_now() -> datetime:
    """Return an aware local timestamp for persisted scheduling state."""
    return datetime.now().astimezone()


def _parse_state_time(value, local_tz) -> datetime | None:
    """Parse persisted ISO time, treating legacy naive values as local time."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=local_tz)
    return parsed


def _scheduled_run_from_state(state: dict, now: datetime) -> datetime | None:
    """Return the earliest safe next cycle time represented by daemon state."""
    if now.tzinfo is None:
        raise ValueError("now must include a timezone")

    persisted = _parse_state_time(state.get("next_run_at"), now.tzinfo)
    last_run = _parse_state_time(state.get("last_run"), now.tzinfo)
    derived = None
    if last_run is not None:
        interval = _RETRY_INTERVAL if state.get("pending_digest") else _DAILY_INTERVAL
        derived = last_run + interval

    candidates = [candidate for candidate in (persisted, derived) if candidate is not None]
    if not candidates:
        return None

    # run_cycle writes last_run before the daemon records its next_run_at.
    # Taking the later value closes that small restart window: a stale, elapsed
    # next_run_at cannot cause a duplicate crawl after a completed cycle.
    return max(candidates)


def _next_run_time(cycle_success: bool, now: datetime, state: dict) -> datetime:
    """Return the next attempt time without letting cycle duration shift the daily slot."""
    if now.tzinfo is None:
        raise ValueError("now must include a timezone")
    if not cycle_success:
        return now + _RETRY_INTERVAL

    anchor = _parse_state_time(state.get("last_run"), now.tzinfo) or now
    next_run_at = anchor + _DAILY_INTERVAL
    while next_run_at <= now:
        next_run_at += _DAILY_INTERVAL
    return next_run_at


def _persist_next_run(cycle_success: bool, now: datetime | None = None) -> datetime:
    """Persist and return the next daemon attempt time."""
    now = now or _local_now()
    state = load_state()
    next_run_at = _next_run_time(cycle_success, now, state)
    state["next_run_at"] = next_run_at.isoformat()
    save_state(state)
    return next_run_at


async def _wait_until(target: datetime, shutdown: asyncio.Event):
    """Wait for a wall-clock deadline while remaining responsive to suspend and signals."""
    if target.tzinfo is None:
        raise ValueError("target must include a timezone")

    while not shutdown.is_set():
        remaining = (target - _local_now()).total_seconds()
        if remaining <= 0:
            return
        try:
            await asyncio.wait_for(
                shutdown.wait(),
                timeout=min(remaining, _SCHEDULE_RECHECK_SECONDS),
            )
        except asyncio.TimeoutError:
            pass


def _record_tts_outcome(state: dict, audio_result) -> bool:
    """Persist whether a digest should be retried; return cycle success."""
    if audio_result is None:
        state["pending_digest"] = True
        return False

    state["pending_digest"] = False
    if audio_result.approved:
        state.pop("pending_review", None)
    else:
        # A complete review artifact needs human attention, not another costly
        # writer/TTS run over the same judged set. Preserve its exact paths and
        # continue on the normal daily cadence.
        state["pending_review"] = {
            "recorded_at": _local_now().isoformat(),
            "qc_status": audio_result.qc_status,
            "audio": str(audio_result.path.resolve()),
            "sources": str(audio_result.notes_path.resolve()),
            "manifest": str(audio_result.manifest_path.resolve()),
            "screening_report": str(audio_result.screening_report_path.resolve()),
        }
    return True


# ─── Core cycle ─────────────────────────────────────────────────────────────

async def _run_crawlers_with_judge_supervision(
    queue: ContentQueue,
    seen: set,
    judge_tasks: list[asyncio.Task],
):
    """Run crawlers while failing promptly if a judge worker terminates."""
    crawl_task = asyncio.create_task(run_all_crawlers(queue, seen))
    try:
        done, _ = await asyncio.wait(
            [crawl_task, *judge_tasks],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if crawl_task in done:
            await crawl_task
            return

        for task in (task for task in done if task in judge_tasks):
            if task.cancelled():
                raise RuntimeError("Codex judge worker was cancelled during crawling")
            exception = task.exception()
            if exception is not None:
                raise RuntimeError("Codex judge worker crashed during crawling") from exception
        raise RuntimeError("Codex judge worker exited during crawling")
    finally:
        if not crawl_task.done():
            crawl_task.cancel()
            await asyncio.gather(crawl_task, return_exceptions=True)


async def _drain_judges(
    queue: ContentQueue,
    judge_tasks: list[asyncio.Task],
    stop_event: asyncio.Event,
):
    """Wait for queue completion while surfacing worker crashes immediately."""
    join_task = asyncio.create_task(queue.join_pending())
    try:
        done, _ = await asyncio.wait(
            [join_task, *judge_tasks],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if join_task in done:
            await join_task
            stop_event.set()
            await asyncio.gather(*judge_tasks)
            return

        completed_workers = [task for task in done if task in judge_tasks]
        for task in completed_workers:
            if task.cancelled():
                raise RuntimeError("Codex judge worker was cancelled before the queue drained")
            exception = task.exception()
            if exception is not None:
                raise RuntimeError("Codex judge worker crashed") from exception
        raise RuntimeError("Codex judge worker exited before the queue drained")
    finally:
        if not join_task.done():
            join_task.cancel()
            await asyncio.gather(join_task, return_exceptions=True)


async def _cancel_judges(judge_tasks: list[asyncio.Task], stop_event: asyncio.Event):
    stop_event.set()
    for task in judge_tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*judge_tasks, return_exceptions=True)


async def run_cycle(crawl: bool = True, digest: bool = True, verbose: bool = False):
    cycle_started_at = _local_now()
    try:
        await preflight_codex()
    except CodexExecError as exc:
        raise RuntimeError(f"Codex preflight failed: {exc}") from exc
    if digest and TTS_ENABLED:
        try:
            tts_runtime = await asyncio.to_thread(preflight_tts)
            log.info(
                "TTS preflight passed: Coqui %s, Torch %s, CUDA=%s, voice=%s",
                tts_runtime.get("tts"),
                tts_runtime.get("torch"),
                tts_runtime.get("cuda_available"),
                tts_runtime.get("voice"),
            )
        except Exception as exc:
            raise RuntimeError(f"TTS preflight failed: {exc}") from exc

    state = load_state()
    seen = set(state.get("seen_urls", []))
    queue = ContentQueue()
    stop_event = asyncio.Event()
    cycle_success = True

    log.info(f"=== Cycle start === (seen: {len(seen)} URLs)")

    if crawl:
        # Start bounded judge workers in the background. Each worker batches
        # items into one Codex invocation to avoid paying CLI context overhead
        # for every individual paper.
        judge_tasks = [
            asyncio.create_task(judge_loop(queue, stop_event))
            for _ in range(max(1, CODEX_JUDGE_WORKERS))
        ]

        try:
            # run all crawlers — they feed the queue concurrently
            log.info("Starting crawlers...")
            await _run_crawlers_with_judge_supervision(queue, seen, judge_tasks)

            # Wait for all in-flight calls and bounded retries. Queue.join
            # tracks work after get(); _drain_judges also watches workers so a
            # crash cannot leave join waiting forever.
            log.info(
                "Crawlers done. Waiting for judge to finish (%s pending)...",
                queue.pending_count(),
            )
            await _drain_judges(queue, judge_tasks, stop_event)
        except BaseException:
            await _cancel_judges(judge_tasks, stop_event)
            raise

        log.info(
            f"Judging complete: {queue.stats['kept']}/{queue.stats['judged']} kept "
            f"({queue.stats['skipped']} skipped, {queue.stats['failed']} failed, "
            f"{queue.stats['retried']} retries)"
        )

        # save raw judged items
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        kept_items = await queue.drain_kept()
        failed_items = await queue.drain_failed()
        for item in failed_items:
            # A temporary Codex failure must not permanently mark the URL seen.
            seen.discard(item.url)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = RAW_DIR / f"judged_{stamp}.json"
        raw_path.write_text(json.dumps(
            [
                {
                    "title": item.title,
                    "url": item.url,
                    "source": item.source,
                    "type": item.content_type.value,
                    "authors": item.authors,
                    "published": item.published,
                    "scores": {
                        "technical_depth": item.technical_depth,
                        "novelty": item.novelty,
                        "relevance": item.relevance,
                    },
                    "summary": item.summary,
                    "key_contributions": item.key_contributions,
                    "reason": item.judge_reason,
                    "judge_attempts": item.judge_attempts,
                    "abstract": item.abstract,
                    "metadata": item.metadata,
                }
                for item in kept_items
            ],
            indent=2,
        ))
        log.info(f"Saved {len(kept_items)} judged items to {raw_path}")
    else:
        # load last judged set
        raw_files = sorted(RAW_DIR.glob("judged_*.json"))
        if not raw_files:
            log.error("No previous judged items found for --digest-only")
            return
        raw_path = raw_files[-1]
        log.info(f"Loading previous judged items from {raw_path}")
        raw_data = json.loads(raw_path.read_text())

        from queue_manager import ContentItem, ContentType
        kept_items = []
        for d in raw_data:
            item = ContentItem(
                url=d["url"],
                title=d["title"],
                source=d["source"],
                content_type=ContentType(d["type"]),
                raw_text="",
                abstract=d.get("abstract", ""),
                authors=d.get("authors", ""),
                published=d.get("published", ""),
                metadata=d.get("metadata", {}),
            )
            scores = d.get("scores", {})
            item.technical_depth = scores.get("technical_depth")
            item.novelty = scores.get("novelty")
            item.relevance = scores.get("relevance")
            item.summary = d.get("summary", "")
            item.key_contributions = d.get("key_contributions", "")
            item.judge_reason = d.get("reason", "")
            item.judge_attempts = d.get("judge_attempts", 0)
            item.verdict = "keep"
            kept_items.append(item)

    if digest and kept_items:
        # sort by combined score (highest first)
        kept_items.sort(
            key=lambda x: (x.technical_depth or 0) + (x.novelty or 0) + (x.relevance or 0),
            reverse=True,
        )

        word_estimate = sum(len((item.summary or "").split()) * 3 for item in kept_items)
        log.info(
            f"Composing digest from {len(kept_items)} items "
            f"(est. {word_estimate} words -> {word_estimate / WORDS_PER_MINUTE:.0f} min)"
        )

        script_path = await compose_digest_script(kept_items)
        if script_path:
            source_notes_path = write_digest_source_notes(
                kept_items,
                script_path.with_suffix(".md"),
            )
            log.info("Starting TTS synthesis (this will take a while)...")
            audio_result = synthesize_audio(script_path, source_notes_path)
            try:
                save_digest_metadata(
                    kept_items,
                    script_path,
                    source_notes_path,
                    audio_result,
                )
            except Exception:
                # A completed audio/notes bundle is the delivery commit. Keep it
                # and advance state even if secondary JSON metadata cannot be
                # recorded, avoiding an expensive duplicate retry.
                log.error("Could not save digest metadata", exc_info=True)

            if TTS_ENABLED:
                cycle_success = _record_tts_outcome(state, audio_result)
                if audio_result and audio_result.approved:
                    log.info(
                        "=== DIGEST READY: %s (sources: %s) ===",
                        audio_result.path,
                        audio_result.notes_path,
                    )
                elif audio_result:
                    log.error(
                        "TTS needs review (%s). Audio=%s sources=%s manifest=%s. "
                        "Artifact preserved without automatic regeneration.",
                        audio_result.qc_status,
                        audio_result.path,
                        audio_result.notes_path,
                        audio_result.manifest_path,
                    )
                else:
                    log.error(
                        "TTS failed QC. Script and sources saved for digest-only retry: %s, %s",
                        script_path,
                        source_notes_path,
                    )
            else:
                state["pending_digest"] = False
                state.pop("pending_review", None)
                log.info(
                    "TTS is disabled. Script and sources ready at: %s, %s",
                    script_path,
                    source_notes_path,
                )
        else:
            log.error("Failed to compose digest script")
            cycle_success = False
            state["pending_digest"] = True
    elif digest:
        log.warning("No items passed filtering — no digest to compose")
        state["pending_digest"] = False

    # update state
    state["seen_urls"] = list(seen)
    state["last_run"] = cycle_started_at.isoformat()
    # Any previously scheduled time belongs to the cycle that just finished.
    # Removing it here makes a restart derive the correct daily/hourly delay
    # from this last_run until daemon_loop persists the new exact schedule.
    state.pop("next_run_at", None)
    state["runs"] = state.get("runs", 0) + 1
    save_state(state)

    log.info(f"=== Cycle complete === (total seen: {len(seen)})")
    return cycle_success


# ─── Daemon mode ─────────────────────────────────────────────────────────────

async def daemon_loop(verbose: bool = False):
    log.info("Agent daemon started — will run one cycle per day")

    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()

    def handle_signal(sig, _):
        log.info(f"Received {signal.Signals(sig).name}, shutting down gracefully...")
        # Python may transparently restart the selector syscall after this
        # synchronous signal handler returns. Scheduling thread-safely writes
        # to the loop's wakeup pipe so a long daily sleep ends immediately.
        loop.call_soon_threadsafe(shutdown.set)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    now = _local_now()
    scheduled_run = _scheduled_run_from_state(load_state(), now)
    if scheduled_run is not None:
        remaining = max(0.0, (scheduled_run - now).total_seconds())
        if remaining:
            log.info(
                "Next cycle scheduled for %s (in %.2f hours). Sleeping...",
                scheduled_run.isoformat(timespec="seconds"),
                remaining / 3600,
            )
            await _wait_until(scheduled_run, shutdown)

    while not shutdown.is_set():
        cycle_success = False
        cycle_started_at = _local_now()
        try:
            pending_digest = bool(load_state().get("pending_digest", False))
            if pending_digest:
                log.info("Retrying the pending digest before crawling new content")
            cycle_success = await run_cycle(
                crawl=not pending_digest,
                digest=True,
                verbose=verbose,
            )
        except Exception as e:
            log.error(f"Cycle failed: {e}", exc_info=True)

        # Persist the delay before sleeping so a service restart honors the
        # remaining daily/retry interval instead of immediately starting over.
        now = _local_now()
        next_run_at = _next_run_time(
            cycle_success,
            now,
            {"last_run": cycle_started_at.isoformat()},
        )
        try:
            next_run_at = _persist_next_run(cycle_success, now)
        except Exception:
            log.error(
                "Could not persist next cycle time; restart safety is unavailable",
                exc_info=True,
            )
        remaining = max(0.0, (next_run_at - _local_now()).total_seconds())
        log.info(
            "Next cycle scheduled for %s (in %.2f hours). Sleeping...",
            next_run_at.isoformat(timespec="seconds"),
            remaining / 3600,
        )
        await _wait_until(next_run_at, shutdown)

    log.info("Agent daemon stopped")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Research Agent — crawl, judge, narrate bleeding-edge AI research"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--daemon", action="store_true", help="Run continuously (one cycle/day)")
    mode.add_argument("--crawl-only", action="store_true", help="Crawl and judge, skip TTS")
    mode.add_argument("--digest-only", action="store_true", help="Compose + TTS from last judged set")
    mode.add_argument(
        "--install-service",
        action="store_true",
        help="Install/update automatic startup for the current repository and Python",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()

    if args.install_service:
        service_path, desktop_path, changed = install_startup_service()
        status = "updated" if changed else "already current"
        print(f"Startup service {status}: {service_path}")
        print(f"Encrypted-home login fallback: {desktop_path}")
        print("The agent is enabled and running.")
        return

    setup_logging(args.verbose)

    # ensure output dirs exist
    for d in [OUTPUT_DIR, RAW_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    if args.daemon:
        asyncio.run(daemon_loop(verbose=args.verbose))
    elif args.crawl_only:
        success = asyncio.run(run_cycle(crawl=True, digest=False, verbose=args.verbose))
        if not success:
            sys.exit(1)
    elif args.digest_only:
        success = asyncio.run(run_cycle(crawl=False, digest=True, verbose=args.verbose))
        if not success:
            sys.exit(1)
    else:
        success = asyncio.run(run_cycle(verbose=args.verbose))
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
