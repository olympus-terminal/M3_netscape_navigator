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
  python agent.py --crawl-only   # crawl and judge, skip TTS
  python agent.py --digest-only  # compose and synthesize from last judged set
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
from datetime import datetime
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
from digest import compose_digest_script, preflight_tts, save_digest_metadata, synthesize_audio
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


# ─── State persistence ──────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"seen_urls": [], "last_run": None, "runs": 0}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


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
            "recorded_at": datetime.now().isoformat(),
            "qc_status": audio_result.qc_status,
            "audio": str(audio_result.path.resolve()),
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
            log.info("Starting TTS synthesis (this will take a while)...")
            audio_result = synthesize_audio(script_path)
            save_digest_metadata(kept_items, script_path, audio_result)

            if TTS_ENABLED:
                cycle_success = _record_tts_outcome(state, audio_result)
                if audio_result and audio_result.approved:
                    log.info("=== DIGEST READY: %s ===", audio_result.path)
                elif audio_result:
                    log.error(
                        "TTS needs review (%s). Audio=%s manifest=%s. "
                        "Artifact preserved without automatic regeneration.",
                        audio_result.qc_status,
                        audio_result.path,
                        audio_result.manifest_path,
                    )
                else:
                    log.error(
                        "TTS failed QC. Script saved for digest-only retry: %s",
                        script_path,
                    )
            else:
                state["pending_digest"] = False
                state.pop("pending_review", None)
                log.info("TTS is disabled. Script ready at: %s", script_path)
        else:
            log.error("Failed to compose digest script")
            cycle_success = False
            state["pending_digest"] = True
    elif digest:
        log.warning("No items passed filtering — no digest to compose")
        state["pending_digest"] = False

    # update state
    state["seen_urls"] = list(seen)
    state["last_run"] = datetime.now().isoformat()
    state["runs"] = state.get("runs", 0) + 1
    save_state(state)

    log.info(f"=== Cycle complete === (total seen: {len(seen)})")
    return cycle_success


# ─── Daemon mode ─────────────────────────────────────────────────────────────

async def daemon_loop(verbose: bool = False):
    log.info("Agent daemon started — will run one cycle per day")

    shutdown = asyncio.Event()

    def handle_signal(sig, _):
        log.info(f"Received {signal.Signals(sig).name}, shutting down gracefully...")
        shutdown.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while not shutdown.is_set():
        cycle_success = False
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

        # sleep until next cycle (with periodic wake to check shutdown)
        next_run = 24 * 3600 if cycle_success else 3600
        log.info("Next cycle in %.0f hours. Sleeping...", next_run / 3600)
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=next_run)
        except asyncio.TimeoutError:
            pass

    log.info("Agent daemon stopped")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Research Agent — crawl, judge, narrate bleeding-edge AI research"
    )
    parser.add_argument("--daemon", action="store_true", help="Run continuously (one cycle/day)")
    parser.add_argument("--crawl-only", action="store_true", help="Crawl and judge, skip TTS")
    parser.add_argument("--digest-only", action="store_true", help="Compose + TTS from last judged set")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()

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
