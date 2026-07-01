"""M3 judge — scores and summarizes content from the queue. Stays saturated."""

import asyncio
import json
import logging

import aiohttp

from config import (
    FOCUS_AREAS,
    M3_BASE_URL,
    M3_MAX_TOKENS,
    M3_MODEL,
    M3_TEMPERATURE,
    NOVELTY_THRESHOLD,
    RELEVANCE_THRESHOLD,
    TECHNICAL_DEPTH_THRESHOLD,
)
from queue_manager import ContentItem, ContentQueue

log = logging.getLogger(__name__)

FOCUS_LIST = "\n".join(f"  - {f}" for f in FOCUS_AREAS)

JUDGE_SYSTEM_PROMPT = f"""You are a senior AI researcher filtering content for a daily technical briefing.
Your audience is a PhD-level scientist who wants ONLY bleeding-edge, technically substantive content.

REJECT:
- Pop-science rewrites of known work
- Incremental improvements without novel insight
- Marketing / product announcements without technical detail
- Tutorials or explainers of established methods
- Papers that are purely empirical without methodological contribution (unless results are truly surprising)

ACCEPT:
- Novel architectures, training methods, or theoretical insights
- Surprising empirical results that challenge conventional wisdom
- New open-source models/tools with genuine technical innovation
- Significant benchmark improvements with clear methodological contribution
- Work that opens new research directions

Focus areas (but don't limit to these):
{FOCUS_LIST}

For each piece of content, respond with ONLY a JSON object (no markdown, no explanation):
{{
  "technical_depth": <1-10>,
  "novelty": <1-10>,
  "relevance": <1-10>,
  "verdict": "keep" or "skip",
  "reason": "<one sentence why>",
  "key_contributions": "<2-3 bullet points of what's actually new, if keeping>",
  "summary": "<3-5 sentence technical summary for the briefing, if keeping>"
}}"""


async def _call_m3(text: str) -> dict | None:
    payload = {
        "model": M3_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "max_tokens": M3_MAX_TOKENS,
        "temperature": M3_TEMPERATURE,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{M3_BASE_URL}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    log.warning(f"M3 returned {resp.status}: {await resp.text()}")
                    return None
                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
                # strip markdown fences if present
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]
                    if content.endswith("```"):
                        content = content[:-3]
                return json.loads(content.strip())
    except json.JSONDecodeError as e:
        log.warning(f"M3 returned non-JSON: {e}")
        return None
    except Exception as e:
        log.warning(f"M3 call failed: {e}")
        return None


def _format_for_judge(item: ContentItem) -> str:
    parts = [
        f"SOURCE: {item.source}",
        f"TITLE: {item.title}",
    ]
    if item.authors:
        parts.append(f"AUTHORS: {item.authors}")
    if item.published:
        parts.append(f"DATE: {item.published}")
    parts.append(f"TYPE: {item.content_type.value}")
    if item.abstract and item.abstract != item.raw_text:
        parts.append(f"ABSTRACT: {item.abstract}")
    text = item.raw_text[:6000]  # cap input size
    parts.append(f"CONTENT:\n{text}")
    return "\n\n".join(parts)


async def judge_loop(queue: ContentQueue, stop_event: asyncio.Event):
    """Pull items from the pending queue and judge them via M3. Runs until stopped."""
    log.info("M3 judge started — waiting for content")

    while not stop_event.is_set():
        try:
            item = await asyncio.wait_for(queue.get_pending(), timeout=30)
        except asyncio.TimeoutError:
            continue

        prompt = _format_for_judge(item)
        log.info(f"Judging: {item.title[:80]}")

        result = await _call_m3(prompt)
        if result is None:
            log.warning(f"M3 failed to judge: {item.title[:80]}, re-queuing")
            await queue.submit(item)
            await asyncio.sleep(10)
            continue

        item.technical_depth = result.get("technical_depth", 0)
        item.novelty = result.get("novelty", 0)
        item.relevance = result.get("relevance", 0)
        item.verdict = result.get("verdict", "skip")
        item.summary = result.get("summary", "")
        item.key_contributions = result.get("key_contributions", "")

        # enforce thresholds even if M3 said "keep"
        if (
            item.technical_depth < TECHNICAL_DEPTH_THRESHOLD
            or item.novelty < NOVELTY_THRESHOLD
            or item.relevance < RELEVANCE_THRESHOLD
        ):
            item.verdict = "skip"

        await queue.submit_judged(item)

    log.info("M3 judge stopped")
