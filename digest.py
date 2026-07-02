"""Compose a narration script from judged content, then synthesize to MP3."""

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

import aiohttp

from config import (
    AUDIO_DIR,
    DIGEST_DIR,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    SCRIPT_DIR_OUT,
    TARGET_WORD_COUNT,
    TTS_CONDA_PYTHON,
    TTS_ENABLED,
    TTS_FORMAT,
    TTS_SCRIPT,
    TTS_SCRIPT_DIR,
    TTS_VOICE,
    WORDS_PER_MINUTE,
)
from queue_manager import ContentItem

log = logging.getLogger(__name__)

SCRIPT_WRITER_SYSTEM = """You are writing a narration script for a technical AI research briefing.
The listener is a PhD-level scientist who wants dense, precise, technical content.

Style guidelines:
- Speak like a knowledgeable colleague, not a news anchor
- Use precise technical language — don't simplify for a general audience
- Explain WHY something matters, not just WHAT it is
- Compare to prior work when relevant
- Include specific numbers: parameter counts, benchmark scores, speedups
- Group related items naturally — don't just list papers sequentially
- Use transitions that connect ideas: "Building on that theme...", "In a different direction..."
- No filler, no hype words ("groundbreaking", "revolutionary", "game-changing")
- Pronounce abbreviations naturally: "RL" becomes "R.L.", "LLM" becomes "L.L.M."
- For paper titles, just say the key idea, not the full formal title
- End with a brief synthesis: what themes emerged, what directions seem most promising

The script will be read by a text-to-speech system, so:
- Write out numbers: "three point five billion" not "3.5B"
- Spell out ambiguous abbreviations on first use
- Avoid markdown formatting, bullet points, or headers — just flowing prose
- Use periods for pauses, not commas in long sequences"""


async def _call_llm_scriptwriter(content: str) -> str | None:
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY and LLM_API_KEY != "none":
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SCRIPT_WRITER_SYSTEM},
            {"role": "user", "content": content},
        ],
        "max_tokens": 131072,
        "temperature": 0.5,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{LLM_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=1800),
            ) as resp:
                if resp.status != 200:
                    log.warning(f"LLM scriptwriter returned {resp.status}")
                    return None
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
    except Exception as e:
        log.warning(f"LLM scriptwriter call failed: {e}")
        return None


def _prepare_batch_prompt(items: list[ContentItem], batch_num: int, total_batches: int) -> str:
    header = (
        f"Write segment {batch_num} of {total_batches} for today's AI research briefing.\n"
        f"Target length: about {TARGET_WORD_COUNT // total_batches} words.\n\n"
    )
    if batch_num == 1:
        header += (
            "Begin with a brief opening: today's date and a one-sentence preview of the themes covered.\n\n"
        )
    if batch_num == total_batches:
        header += "End with a synthesis: what themes emerged and what directions seem most promising.\n\n"

    header += "Here are the items to cover in this segment:\n\n"

    pieces = []
    for i, item in enumerate(items, 1):
        piece = f"--- Item {i} ---\n"
        piece += f"Title: {item.title}\n"
        if item.authors:
            piece += f"Authors: {item.authors}\n"
        piece += f"Source: {item.source}\n"
        piece += f"Key contributions: {item.key_contributions}\n"
        piece += f"Technical summary: {item.summary}\n"
        piece += f"URL: {item.url}\n"
        pieces.append(piece)

    return header + "\n".join(pieces)


async def compose_digest_script(items: list[ContentItem]) -> Path | None:
    """Take judged items and produce a narration script via the LLM."""
    if not items:
        log.warning("No items to compose digest from")
        return None

    SCRIPT_DIR_OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = SCRIPT_DIR_OUT / f"digest_{stamp}.txt"

    batch_size = max(5, len(items) // 6)
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    total_batches = len(batches)

    log.info(f"Composing digest: {len(items)} items in {total_batches} batches")

    segments = []
    for i, batch in enumerate(batches, 1):
        log.info(f"Writing segment {i}/{total_batches} ({len(batch)} items)")
        prompt = _prepare_batch_prompt(batch, i, total_batches)
        segment = await _call_llm_scriptwriter(prompt)
        if segment:
            segments.append(segment)
        else:
            log.warning(f"Failed to generate segment {i}")

    if not segments:
        log.error("Failed to generate any script segments")
        return None

    full_script = "\n\n".join(segments)
    word_count = len(full_script.split())
    script_path.write_text(full_script)

    log.info(
        f"Digest script written: {script_path} "
        f"({word_count} words, ~{word_count / WORDS_PER_MINUTE:.0f} min)"
    )

    return script_path


def synthesize_audio(script_path: Path) -> Path | None:
    """Run the TTS pipeline on the digest script to produce an MP3."""
    if not TTS_ENABLED:
        log.info("TTS not configured — skipping audio synthesis. Script saved as text.")
        return None

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PATH"] = f"{TTS_SCRIPT_DIR}:{env.get('PATH', '')}"

    cmd = [
        TTS_CONDA_PYTHON,
        TTS_SCRIPT,
        "--multi", str(script_path),
        "--headers", "AI Research Briefing.",
        "--voice", TTS_VOICE,
        "--format", TTS_FORMAT,
        "--screen",
    ]

    log.info(f"Starting TTS synthesis...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=7200,
            cwd=str(AUDIO_DIR),
        )

        if result.returncode != 0:
            log.error(f"TTS failed (exit {result.returncode}):\n{result.stderr[-2000:]}")
            return None

        # TTS pipeline may output to various locations and naming patterns
        mp3_files = sorted(AUDIO_DIR.rglob("*.mp3"), key=lambda p: p.stat().st_mtime)
        if mp3_files:
            output = mp3_files[-1]
            size_mb = output.stat().st_size / (1024 * 1024)
            log.info(f"TTS complete: {output} ({size_mb:.1f} MB)")
            return output

        log.error("TTS completed but no output MP3 found")
        return None

    except subprocess.TimeoutExpired:
        log.error("TTS synthesis timed out after 2 hours")
        return None
    except Exception as e:
        log.error(f"TTS synthesis error: {e}")
        return None


def save_digest_metadata(items: list[ContentItem], script_path: Path, audio_path: Path | None):
    """Save metadata about what went into this digest."""
    DIGEST_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_path = DIGEST_DIR / f"digest_{stamp}.json"

    meta = {
        "timestamp": datetime.now().isoformat(),
        "item_count": len(items),
        "script_path": str(script_path),
        "audio_path": str(audio_path) if audio_path else None,
        "items": [
            {
                "title": item.title,
                "url": item.url,
                "source": item.source,
                "scores": {
                    "technical_depth": item.technical_depth,
                    "novelty": item.novelty,
                    "relevance": item.relevance,
                },
                "summary": item.summary,
            }
            for item in items
        ],
    }

    meta_path.write_text(json.dumps(meta, indent=2))
    log.info(f"Digest metadata saved: {meta_path}")
