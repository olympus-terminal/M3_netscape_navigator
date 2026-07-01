"""Thread-safe queue bridging crawlers and the M3 judge."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)


class ContentType(Enum):
    PAPER = "paper"
    BLOG = "blog"
    REPO = "repo"
    DISCUSSION = "discussion"
    VIDEO = "video"


@dataclass
class ContentItem:
    url: str
    title: str
    source: str  # e.g. "arxiv", "huggingface_papers"
    content_type: ContentType
    raw_text: str  # extracted text content
    abstract: str = ""
    authors: str = ""
    published: str = ""
    fetched_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)

    # filled by M3 judge
    technical_depth: Optional[int] = None
    novelty: Optional[int] = None
    relevance: Optional[int] = None
    summary: Optional[str] = None
    key_contributions: Optional[str] = None
    verdict: Optional[str] = None  # "keep" or "skip"


class ContentQueue:
    def __init__(self, maxsize: int = 200):
        self.pending = asyncio.Queue(maxsize=maxsize)
        self.judged_keep = asyncio.Queue()
        self.stats = {"enqueued": 0, "judged": 0, "kept": 0, "skipped": 0}

    async def submit(self, item: ContentItem):
        await self.pending.put(item)
        self.stats["enqueued"] += 1
        log.info(f"Queued: {item.title[:80]} ({item.source})")

    async def get_pending(self) -> ContentItem:
        return await self.pending.get()

    async def submit_judged(self, item: ContentItem):
        self.stats["judged"] += 1
        if item.verdict == "keep":
            await self.judged_keep.put(item)
            self.stats["kept"] += 1
            log.info(f"KEEP [{item.technical_depth}/{item.novelty}/{item.relevance}]: {item.title[:80]}")
        else:
            self.stats["skipped"] += 1
            log.debug(f"SKIP [{item.technical_depth}/{item.novelty}/{item.relevance}]: {item.title[:80]}")

    def pending_count(self) -> int:
        return self.pending.qsize()

    def kept_count(self) -> int:
        return self.judged_keep.qsize()

    async def drain_kept(self) -> list[ContentItem]:
        items = []
        while not self.judged_keep.empty():
            items.append(await self.judged_keep.get())
        return items
