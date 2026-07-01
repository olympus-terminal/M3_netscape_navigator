#!/usr/bin/env python3
"""
AI Research Agent — persistent daemon that crawls, judges, and narrates
bleeding-edge AI research into daily audio digests.

Architecture:
  [Crawlers] --async--> [Queue] --serial--> [LLM Judge] --> [Script Writer] --> [TTS]

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

from config import (
    BASE_DIR,
    LOG_FILE,
    OUTPUT_DIR,
    RAW_DIR,
    STATE_FILE,
    TARGET_WORD_COUNT,
    WORDS_PER_MINUTE,
)
from crawlers import run_all_crawlers
from digest import compose_digest_script, save_digest_metadata, synthesize_audio
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


# ─── Core cycle ─────────────────────────────────────────────────────────────

async def run_cycle(crawl: bool = True, digest: bool = True, verbose: bool = False):
    state = load_state()
    seen = set(state.get("seen_urls", []))
    queue = ContentQueue()
    stop_event = asyncio.Event()

    log.info(f"=== Cycle start === (seen: {len(seen)} URLs)")

    if crawl:
        # start the judge in the background — it will process items as they arrive
        judge_task = asyncio.create_task(judge_loop(queue, stop_event))

        # run all crawlers — they feed the queue concurrently
        log.info("Starting crawlers...")
        await run_all_crawlers(queue, seen)

        # wait for the judge to finish processing remaining items
        log.info(f"Crawlers done. Waiting for judge to finish ({queue.pending_count()} pending)...")
        while queue.pending_count() > 0:
            await asyncio.sleep(5)

        # give judge a moment to finish the last item
        await asyncio.sleep(10)
        stop_event.set()
        await judge_task

        log.info(
            f"Judging complete: {queue.stats['kept']}/{queue.stats['judged']} kept "
            f"({queue.stats['skipped']} skipped)"
        )

        # save raw judged items
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        kept_items = await queue.drain_kept()

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
            audio_path = synthesize_audio(script_path)
            save_digest_metadata(kept_items, script_path, audio_path)

            if audio_path:
                log.info(f"=== DIGEST READY: {audio_path} ===")
            else:
                log.warning(f"TTS failed. Script saved at: {script_path}")
        else:
            log.error("Failed to compose digest script")
    elif digest:
        log.warning("No items passed filtering — no digest to compose")

    # update state
    state["seen_urls"] = list(seen)
    state["last_run"] = datetime.now().isoformat()
    state["runs"] = state.get("runs", 0) + 1
    save_state(state)

    log.info(f"=== Cycle complete === (total seen: {len(seen)})")


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
        try:
            await run_cycle(crawl=True, digest=True, verbose=verbose)
        except Exception as e:
            log.error(f"Cycle failed: {e}", exc_info=True)

        # sleep until next cycle (with periodic wake to check shutdown)
        next_run = 24 * 3600  # 24 hours
        log.info(f"Next cycle in {next_run / 3600:.0f} hours. Sleeping...")
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
        asyncio.run(run_cycle(crawl=True, digest=False, verbose=args.verbose))
    elif args.digest_only:
        asyncio.run(run_cycle(crawl=False, digest=True, verbose=args.verbose))
    else:
        asyncio.run(run_cycle(verbose=args.verbose))


if __name__ == "__main__":
    main()
