"""Async queue bridging crawlers and Codex judge workers."""

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

    # filled by the Codex judge
    technical_depth: Optional[int] = None
    novelty: Optional[int] = None
    relevance: Optional[int] = None
    summary: Optional[str] = None
    key_contributions: Optional[str] = None
    judge_reason: Optional[str] = None
    judge_attempts: int = 0
    verdict: Optional[str] = None  # "keep" or "skip"


class ContentQueue:
    def __init__(self):
        # The crawl is already bounded per source. Keeping the internal queue
        # unbounded avoids a single judge worker deadlocking when it needs to
        # requeue a failed batch while crawlers have filled a bounded queue.
        self.pending = asyncio.Queue()
        self.judged_keep = asyncio.Queue()
        self.judged_failed = asyncio.Queue()
        self.stats = {
            "enqueued": 0,
            "retried": 0,
            "judged": 0,
            "kept": 0,
            "skipped": 0,
            "failed": 0,
        }

    async def submit(self, item: ContentItem):
        await self.pending.put(item)
        self.stats["enqueued"] += 1
        log.info(f"Queued: {item.title[:80]} ({item.source})")

    async def get_pending(self) -> ContentItem:
        return await self.pending.get()

    async def retry(self, item: ContentItem):
        """Return an item to the pending queue without counting it as new content."""
        await self.pending.put(item)
        self.stats["retried"] += 1
        log.warning(
            "Retrying judge attempt %s: %s",
            item.judge_attempts + 1,
            item.title[:80],
        )

    def task_done(self):
        self.pending.task_done()

    async def join_pending(self):
        """Wait until every queued item, including retries, has finished judging."""
        await self.pending.join()

    async def submit_judged(self, item: ContentItem):
        self.stats["judged"] += 1
        if item.verdict == "keep":
            await self.judged_keep.put(item)
            self.stats["kept"] += 1
            log.info(f"KEEP [{item.technical_depth}/{item.novelty}/{item.relevance}]: {item.title[:80]}")
        else:
            self.stats["skipped"] += 1
            log.debug(f"SKIP [{item.technical_depth}/{item.novelty}/{item.relevance}]: {item.title[:80]}")

    async def submit_failed(self, item: ContentItem):
        """Record an item whose bounded Codex attempts were exhausted."""
        item.verdict = "skip"
        self.stats["judged"] += 1
        self.stats["skipped"] += 1
        self.stats["failed"] += 1
        await self.judged_failed.put(item)
        log.error("FAILED after %s judge attempts: %s", item.judge_attempts, item.title[:80])

    def pending_count(self) -> int:
        return self.pending.qsize()

    def kept_count(self) -> int:
        return self.judged_keep.qsize()

    async def drain_kept(self) -> list[ContentItem]:
        items = []
        while not self.judged_keep.empty():
            items.append(await self.judged_keep.get())
        return items

    async def drain_failed(self) -> list[ContentItem]:
        items = []
        while not self.judged_failed.empty():
            items.append(await self.judged_failed.get())
        return items
