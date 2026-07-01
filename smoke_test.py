#!/usr/bin/env python3
"""Quick smoke test — fetch a few items, judge one with M3, verify the pipeline."""

import asyncio
import json
import logging
import sys

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s")
log = logging.getLogger("smoke")

from config import M3_BASE_URL, M3_MODEL
from queue_manager import ContentItem, ContentQueue, ContentType
from judge import _call_m3, _format_for_judge


async def test_m3_health():
    log.info("1. Testing M3 health...")
    async with aiohttp.ClientSession() as session:
        async with session.get("http://192.168.0.127:8080/health", timeout=aiohttp.ClientTimeout(total=10)) as resp:
            data = await resp.json()
            assert data.get("status") == "ok", f"M3 unhealthy: {data}"
            log.info("   M3 is healthy")


async def test_arxiv_fetch():
    log.info("2. Testing arxiv fetch...")
    import xml.etree.ElementTree as ET
    url = "https://export.arxiv.org/api/query?search_query=cat:cs.AI&sortBy=submittedDate&sortOrder=descending&max_results=3"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            text = await resp.text()
            root = ET.fromstring(text)
            ns = {"a": "http://www.w3.org/2005/Atom"}
            entries = root.findall("a:entry", ns)
            log.info(f"   Got {len(entries)} papers from arxiv")
            if entries:
                title = entries[0].find("a:title", ns).text.strip().replace("\n", " ")
                abstract = entries[0].find("a:summary", ns).text.strip().replace("\n", " ")
                log.info(f"   First paper: {title[:80]}...")
                return ContentItem(
                    url=entries[0].find("a:id", ns).text.strip(),
                    title=title,
                    source="arxiv",
                    content_type=ContentType.PAPER,
                    raw_text=abstract,
                    abstract=abstract,
                    authors="test",
                )
    return None


async def test_m3_judge(item: ContentItem):
    log.info("3. Testing M3 judge on paper...")
    prompt = _format_for_judge(item)
    result = await _call_m3(prompt)
    if result:
        log.info(f"   M3 verdict: {result.get('verdict')}")
        log.info(f"   Scores: depth={result.get('technical_depth')} novelty={result.get('novelty')} relevance={result.get('relevance')}")
        log.info(f"   Reason: {result.get('reason')}")
        if result.get("summary"):
            log.info(f"   Summary: {result['summary'][:200]}...")
        return result
    else:
        log.error("   M3 judge returned None")
        return None


async def test_hf_api():
    log.info("4. Testing HuggingFace papers API...")
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://huggingface.co/api/daily_papers",
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            data = await resp.json()
            log.info(f"   Got {len(data)} papers from HuggingFace")
            if data:
                title = data[0].get("paper", {}).get("title", "?")
                log.info(f"   Top paper: {title[:80]}...")


async def main():
    log.info("=== AI Research Agent Smoke Test ===\n")

    await test_m3_health()
    item = await test_arxiv_fetch()
    if item:
        await test_m3_judge(item)
    await test_hf_api()

    log.info("\n=== All smoke tests passed ===")


if __name__ == "__main__":
    asyncio.run(main())
