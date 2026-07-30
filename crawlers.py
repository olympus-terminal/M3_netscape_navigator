"""Async crawlers for each source. Human-paced with per-domain jitter."""

import asyncio
import logging
import random
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlparse

import aiohttp

from config import DOMAIN_DELAYS, MAX_PAGES_PER_SOURCE_PER_RUN, SOURCES
from queue_manager import ContentItem, ContentQueue, ContentType

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
}

_ARXIV_NAMESPACES = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


async def _delay_for_domain(domain: str):
    lo, hi = DOMAIN_DELAYS.get(domain, DOMAIN_DELAYS["default"])
    wait = random.uniform(lo, hi)
    log.debug(f"Waiting {wait:.0f}s before next fetch from {domain}")
    await asyncio.sleep(wait)


async def _fetch(session: aiohttp.ClientSession, url: str, as_text: bool = True) -> Optional[str]:
    domain = urlparse(url).netloc
    await _delay_for_domain(domain)
    for attempt in range(3):
        try:
            async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 429:
                    wait = 60 * (attempt + 1)
                    log.warning(f"Rate limited (429) by {domain}, backing off {wait}s")
                    await asyncio.sleep(wait)
                    continue
                if resp.status != 200:
                    log.warning(f"HTTP {resp.status} for {url}")
                    return None
                return await resp.text() if as_text else await resp.read()
        except Exception as e:
            log.warning(f"Fetch error for {url}: {e}")
            return None
    log.warning(f"Gave up after 3 rate-limit retries for {url}")
    return None


def _strip_html(html: str) -> str:
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _arxiv_author_metadata(entry: ET.Element) -> tuple[str, list[dict[str, str]]]:
    """Return author names and any author-affiliation pairs supplied by arXiv."""
    names = []
    author_affiliations = []
    for author in entry.findall("a:author", _ARXIV_NAMESPACES):
        name_element = author.find("a:name", _ARXIV_NAMESPACES)
        if name_element is None or not name_element.text:
            continue
        name = re.sub(r"\s+", " ", name_element.text).strip()
        if not name:
            continue
        names.append(name)

        affiliation_element = author.find("arxiv:affiliation", _ARXIV_NAMESPACES)
        if affiliation_element is None or not affiliation_element.text:
            continue
        affiliation = re.sub(r"\s+", " ", affiliation_element.text).strip()
        if affiliation:
            author_affiliations.append(
                {"author": name, "affiliation": affiliation}
            )
    return ", ".join(names), author_affiliations


def _semantic_scholar_author_names(author_records) -> str:
    """Join the author names returned inline by Semantic Scholar paper search."""
    if not isinstance(author_records, list):
        return ""
    names = []
    for author in author_records:
        if not isinstance(author, dict):
            continue
        name = author.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return ", ".join(names)


# ─── arxiv ───────────────────────────────────────────────────────────────────

async def crawl_arxiv(queue: ContentQueue, seen: set, session: aiohttp.ClientSession):
    cfg = SOURCES["arxiv"]
    if not cfg["enabled"]:
        return
    log.info("Starting arxiv crawl")
    count = 0

    for cat in cfg["categories"]:
        if count >= MAX_PAGES_PER_SOURCE_PER_RUN:
            break
        url = (
            f"{cfg['base_url']}?search_query=cat:{cat}"
            f"&sortBy=submittedDate&sortOrder=descending"
            f"&max_results={cfg['max_results_per_query']}"
        )
        xml_text = await _fetch(session, url)
        if not xml_text:
            continue

        root = ET.fromstring(xml_text)
        ns = _ARXIV_NAMESPACES

        for entry in root.findall("a:entry", ns):
            if count >= MAX_PAGES_PER_SOURCE_PER_RUN:
                break
            paper_id = entry.find("a:id", ns).text.strip()
            if paper_id in seen:
                continue

            title = entry.find("a:title", ns).text.strip().replace("\n", " ")
            abstract = entry.find("a:summary", ns).text.strip().replace("\n", " ")
            authors, author_affiliations = _arxiv_author_metadata(entry)
            published = entry.find("a:published", ns).text[:10]
            metadata = {"category": cat}
            if author_affiliations:
                metadata["author_affiliations"] = author_affiliations

            seen.add(paper_id)
            await queue.submit(ContentItem(
                url=paper_id,
                title=title,
                source="arxiv",
                content_type=ContentType.PAPER,
                raw_text=abstract,
                abstract=abstract,
                authors=authors,
                published=published,
                metadata=metadata,
            ))
            count += 1

    log.info(f"arxiv: queued {count} papers")


# ─── HuggingFace daily papers ───────────────────────────────────────────────

async def crawl_huggingface_papers(queue: ContentQueue, seen: set, session: aiohttp.ClientSession):
    cfg = SOURCES["huggingface_papers"]
    if not cfg["enabled"]:
        return
    log.info("Starting HuggingFace papers crawl")

    api_url = "https://huggingface.co/api/daily_papers"
    text = await _fetch(session, api_url)
    if not text:
        return

    import json
    try:
        papers = json.loads(text)
    except json.JSONDecodeError:
        log.warning("Failed to parse HuggingFace papers API response")
        return

    count = 0
    for paper in papers:
        if count >= MAX_PAGES_PER_SOURCE_PER_RUN:
            break
        paper_data = paper.get("paper", {})
        pid = paper_data.get("id", "")
        url = f"https://huggingface.co/papers/{pid}"
        if url in seen:
            continue

        title = paper_data.get("title", "Untitled")
        abstract = paper_data.get("summary", "")

        seen.add(url)
        await queue.submit(ContentItem(
            url=url,
            title=title,
            source="huggingface_papers",
            content_type=ContentType.PAPER,
            raw_text=abstract,
            abstract=abstract,
            authors=", ".join(a.get("name", "") for a in paper_data.get("authors", [])),
            published=paper_data.get("publishedAt", "")[:10],
            metadata={"upvotes": paper.get("numUpvotes", 0)},
        ))
        count += 1

    log.info(f"HuggingFace papers: queued {count}")


# ─── GitHub trending ────────────────────────────────────────────────────────

async def crawl_github_trending(queue: ContentQueue, seen: set, session: aiohttp.ClientSession):
    cfg = SOURCES["github_trending"]
    if not cfg["enabled"]:
        return
    log.info("Starting GitHub trending crawl")
    count = 0

    for lang in cfg["languages"]:
        if count >= MAX_PAGES_PER_SOURCE_PER_RUN:
            break
        url = f"https://github.com/trending/{lang}?since={cfg['since']}"
        html = await _fetch(session, url)
        if not html:
            continue

        for match in re.finditer(r'class="h3 lh-condensed"[^>]*>\s*<a href="(/[^"]+)"', html):
            if count >= MAX_PAGES_PER_SOURCE_PER_RUN:
                break
            repo_path = match.group(1).strip()
            repo_url = f"https://github.com{repo_path}"
            if repo_url in seen:
                continue

            repo_name = repo_path.strip("/")
            desc_match = re.search(
                rf'href="{re.escape(repo_path)}".*?<p[^>]*>\s*(.*?)\s*</p>',
                html, re.DOTALL
            )
            desc = _strip_html(desc_match.group(1)) if desc_match else ""

            # fetch README for more context
            readme_url = f"https://raw.githubusercontent.com{repo_path}/main/README.md"
            readme = await _fetch(session, readme_url)
            if not readme:
                readme_url = f"https://raw.githubusercontent.com{repo_path}/master/README.md"
                readme = await _fetch(session, readme_url)

            raw_text = f"{desc}\n\n{readme[:3000] if readme else ''}"

            seen.add(repo_url)
            await queue.submit(ContentItem(
                url=repo_url,
                title=repo_name,
                source="github_trending",
                content_type=ContentType.REPO,
                raw_text=raw_text.strip(),
                abstract=desc,
                metadata={"language": lang},
            ))
            count += 1

    log.info(f"GitHub trending: queued {count}")


# ─── Papers With Code ───────────────────────────────────────────────────────

async def crawl_papers_with_code(queue: ContentQueue, seen: set, session: aiohttp.ClientSession):
    cfg = SOURCES["papers_with_code"]
    if not cfg["enabled"]:
        return
    log.info("Starting Papers With Code crawl")

    # PwC merged into HuggingFace — use HF trending endpoint filtered for repos with code
    api_url = "https://huggingface.co/api/daily_papers"
    text = await _fetch(session, api_url)
    if not text:
        return

    import json
    try:
        papers = json.loads(text)
    except json.JSONDecodeError:
        log.warning("Failed to parse PwC/HF API response")
        return

    count = 0
    for paper in papers:
        if count >= MAX_PAGES_PER_SOURCE_PER_RUN:
            break
        paper_data = paper.get("paper", {})
        pid = paper_data.get("id", "")
        url = f"https://huggingface.co/papers/{pid}"
        if url in seen:
            continue

        title = paper_data.get("title", "Untitled")
        abstract = paper_data.get("summary", "")
        if not abstract:
            continue

        seen.add(url)
        await queue.submit(ContentItem(
            url=url,
            title=title,
            source="papers_with_code",
            content_type=ContentType.PAPER,
            raw_text=abstract,
            abstract=abstract,
            authors=", ".join(a.get("name", "") for a in paper_data.get("authors", [])),
            published=paper_data.get("publishedAt", "")[:10],
        ))
        count += 1

    log.info(f"Papers With Code: queued {count}")


# ─── Lab blogs (RSS) ────────────────────────────────────────────────────────

async def crawl_lab_blogs(queue: ContentQueue, seen: set, session: aiohttp.ClientSession):
    cfg = SOURCES["lab_blogs"]
    if not cfg["enabled"]:
        return
    log.info("Starting lab blogs crawl")
    count = 0
    cutoff = datetime.now() - timedelta(days=3)

    for feed_url in cfg["feeds"]:
        if count >= MAX_PAGES_PER_SOURCE_PER_RUN:
            break

        xml_text = await _fetch(session, feed_url)
        if not xml_text:
            continue

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            log.warning(f"Failed to parse RSS feed: {feed_url}")
            continue

        # handle both RSS and Atom feeds
        items = root.findall(".//item") or root.findall(
            ".//{http://www.w3.org/2005/Atom}entry"
        )

        for item in items[:10]:
            if count >= MAX_PAGES_PER_SOURCE_PER_RUN:
                break

            # RSS
            link_el = item.find("link")
            title_el = item.find("title")
            desc_el = item.find("description")

            if link_el is None:
                # Atom
                link_el = item.find("{http://www.w3.org/2005/Atom}link")
                title_el = item.find("{http://www.w3.org/2005/Atom}title")
                desc_el = item.find("{http://www.w3.org/2005/Atom}summary") or \
                          item.find("{http://www.w3.org/2005/Atom}content")

            if link_el is None:
                continue

            link = link_el.text if link_el.text else link_el.get("href", "")
            if not link or link in seen:
                continue

            title = title_el.text if title_el is not None and title_el.text else "Untitled"
            desc = _strip_html(desc_el.text) if desc_el is not None and desc_el.text else ""

            # fetch full page for more content
            page = await _fetch(session, link)
            if page:
                raw_text = f"{desc}\n\n{_strip_html(page)[:4000]}"
            else:
                raw_text = desc

            seen.add(link)
            domain = urlparse(feed_url).netloc
            await queue.submit(ContentItem(
                url=link,
                title=title,
                source=f"blog:{domain}",
                content_type=ContentType.BLOG,
                raw_text=raw_text.strip(),
                abstract=desc[:500],
            ))
            count += 1

    log.info(f"Lab blogs: queued {count}")


# ─── Semantic Scholar (trending AI papers) ───────────────────────────────────

async def crawl_semantic_scholar(queue: ContentQueue, seen: set, session: aiohttp.ClientSession):
    cfg = SOURCES["semantic_scholar"]
    if not cfg["enabled"]:
        return
    log.info("Starting Semantic Scholar crawl")

    search_queries = [
        "large language model",
        "transformer architecture",
        "reinforcement learning from human feedback",
        "vision language model",
        "neural architecture search",
    ]

    count = 0
    for query in search_queries:
        if count >= MAX_PAGES_PER_SOURCE_PER_RUN:
            break
        url = (
            f"{cfg['base_url']}/paper/search"
            f"?query={query}&limit=20&fields={cfg['fields']}"
            f"&year=2025-2026&sort=publicationDate:desc"
        )
        text = await _fetch(session, url)
        if not text:
            continue

        import json
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue

        for paper in data.get("data", []):
            if count >= MAX_PAGES_PER_SOURCE_PER_RUN:
                break
            paper_url = paper.get("url", "")
            if not paper_url or paper_url in seen:
                continue

            title = paper.get("title", "Untitled")
            abstract = paper.get("abstract", "")
            if not abstract:
                continue

            seen.add(paper_url)
            await queue.submit(ContentItem(
                url=paper_url,
                title=title,
                source="semantic_scholar",
                content_type=ContentType.PAPER,
                raw_text=abstract,
                abstract=abstract,
                authors=_semantic_scholar_author_names(paper.get("authors")),
                metadata={
                    "citations": paper.get("citationCount", 0),
                    "has_pdf": bool(paper.get("openAccessPdf")),
                },
            ))
            count += 1

    log.info(f"Semantic Scholar: queued {count}")


# ─── Run all crawlers concurrently ──────────────────────────────────────────

ALL_CRAWLERS = [
    crawl_arxiv,
    crawl_huggingface_papers,
    crawl_github_trending,
    crawl_papers_with_code,
    crawl_lab_blogs,
    crawl_semantic_scholar,
]


async def run_all_crawlers(queue: ContentQueue, seen: set):
    async with aiohttp.ClientSession() as session:
        tasks = [crawler(queue, seen, session) for crawler in ALL_CRAWLERS]
        await asyncio.gather(*tasks, return_exceptions=True)
    log.info(f"All crawlers done. Queue depth: {queue.pending_count()}")
