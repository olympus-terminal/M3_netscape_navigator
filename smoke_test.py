#!/usr/bin/env python3
"""Preflight Codex locally; opt into paid inference and network source checks."""

import argparse
import asyncio
import logging
import xml.etree.ElementTree as ET

import aiohttp

from codex_backend import preflight_codex
from digest import preflight_tts
from judge import _call_codex_judge
from queue_manager import ContentItem, ContentType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s")
log = logging.getLogger("smoke")


async def test_codex_preflight():
    log.info("Testing Codex CLI and saved authentication...")
    version, _ = await preflight_codex()
    log.info("Codex ready: %s", version)


async def test_live_codex_judge():
    log.info("Running one live Codex judge batch...")
    item = ContentItem(
        url="https://example.test/research",
        title="Sparse recurrent attention for long-context inference",
        source="smoke_test",
        content_type=ContentType.PAPER,
        raw_text=(
            "We introduce a recurrent sparse-attention method that retains selected key-value "
            "states across segments. On a controlled long-context benchmark it reduces cache "
            "memory by sixty percent at matched accuracy. This is synthetic smoke-test content, "
            "not a claim about a real paper."
        ),
        abstract="Synthetic input used only to verify the Codex structured-output contract.",
    )
    result = await _call_codex_judge([item])
    if result is None:
        raise RuntimeError("Live Codex judge returned no valid result")
    log.info("Codex judge contract passed: %s", result[0]["verdict"])


async def test_live_sources():
    log.info("Testing arXiv and Hugging Face source APIs...")
    arxiv_url = (
        "https://export.arxiv.org/api/query"
        "?search_query=cat:cs.AI&sortBy=submittedDate&sortOrder=descending&max_results=1"
    )
    async with aiohttp.ClientSession() as session:
        async with session.get(arxiv_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            response.raise_for_status()
            root = ET.fromstring(await response.text())
            entries = root.findall("{http://www.w3.org/2005/Atom}entry")
            if not entries:
                raise RuntimeError("arXiv returned no entries")
        async with session.get(
            "https://huggingface.co/api/daily_papers",
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            response.raise_for_status()
            papers = await response.json()
            if not isinstance(papers, list):
                raise RuntimeError("Hugging Face returned an unexpected payload")
    log.info("Source APIs ready: arXiv=%s, Hugging Face=%s", len(entries), len(papers))


async def main(live_codex: bool, live_sources: bool, check_tts: bool):
    log.info("=== AI Research Agent — Preflight ===")
    await test_codex_preflight()
    if live_codex:
        await test_live_codex_judge()
    if live_sources:
        await test_live_sources()
    if check_tts:
        runtime = await asyncio.to_thread(preflight_tts)
        if not runtime.get("enabled"):
            raise RuntimeError("TTS is not configured")
        log.info(
            "TTS ready: Coqui=%s Torch=%s CUDA=%s voice=%s",
            runtime.get("tts"),
            runtime.get("torch"),
            runtime.get("cuda_available"),
            runtime.get("voice"),
        )
    log.info("=== Requested checks passed ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-codex",
        action="store_true",
        help="Make one real Codex inference call (uses plan/API quota)",
    )
    parser.add_argument(
        "--live-sources",
        action="store_true",
        help="Query the live arXiv and Hugging Face APIs",
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Verify the configured screened Coqui runtime without synthesis",
    )
    args = parser.parse_args()
    asyncio.run(main(args.live_codex, args.live_sources, args.tts))
