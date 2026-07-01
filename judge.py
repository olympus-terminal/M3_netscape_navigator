"""LLM judge — scores and summarizes content from the queue. Stays saturated."""

import asyncio
import json
import logging

import aiohttp

from config import (
    FOCUS_AREAS,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
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


async def _call_llm(text: str) -> dict | None:
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY and LLM_API_KEY != "none":
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{LLM_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    log.warning(f"LLM returned {resp.status}: {await resp.text()}")
                    return None
                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]
                    if content.endswith("```"):
                        content = content[:-3]
                return json.loads(content.strip())
    except json.JSONDecodeError as e:
        log.warning(f"LLM returned non-JSON: {e}")
        return None
    except Exception as e:
        log.warning(f"LLM call failed: {e}")
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
    text = item.raw_text[:6000]
    parts.append(f"CONTENT:\n{text}")
    return "\n\n".join(parts)


async def judge_loop(queue: ContentQueue, stop_event: asyncio.Event):
    """Pull items from the pending queue and judge them via the LLM. Runs until stopped."""
    log.info("Judge started — waiting for content")

    while not stop_event.is_set():
        try:
            item = await asyncio.wait_for(queue.get_pending(), timeout=30)
        except asyncio.TimeoutError:
            continue

        prompt = _format_for_judge(item)
        log.info(f"Judging: {item.title[:80]}")

        result = await _call_llm(prompt)
        if result is None:
            log.warning(f"LLM failed to judge: {item.title[:80]}, re-queuing")
            await queue.submit(item)
            await asyncio.sleep(10)
            continue

        item.technical_depth = result.get("technical_depth", 0)
        item.novelty = result.get("novelty", 0)
        item.relevance = result.get("relevance", 0)
        item.verdict = result.get("verdict", "skip")
        item.summary = result.get("summary", "")
        item.key_contributions = result.get("key_contributions", "")

        if (
            item.technical_depth < TECHNICAL_DEPTH_THRESHOLD
            or item.novelty < NOVELTY_THRESHOLD
            or item.relevance < RELEVANCE_THRESHOLD
        ):
            item.verdict = "skip"

        await queue.submit_judged(item)

    log.info("Judge stopped")
