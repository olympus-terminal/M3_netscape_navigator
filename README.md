# M3_netscape_navigator

Autonomous agent that crawls bleeding-edge AI research, filters it through a local LLM, and produces daily audio digests via text-to-speech.

Designed for researchers who want technically dense, PhD-level briefings — not pop-science summaries. The agent uses an LLM as an editorial filter, scoring each piece on technical depth, novelty, and relevance, then composing a narrated script from the survivors.

## Architecture

```
[Crawlers] ──async──▶ [Queue] ──serial──▶ [LLM Judge] ──▶ [Script Writer] ──▶ [TTS] ──▶ MP3
```

- **Crawlers** fetch from multiple sources concurrently, with human-paced delays per domain (2–5 min jitter) to stay under rate limits and avoid bot detection.
- **Queue** keeps the LLM saturated — while it judges paper N, papers N+1 through N+5 are already fetched and waiting. No idle inference time.
- **Judge** scores each piece 1–10 on technical depth, novelty, and relevance. Rejects pop-science rewrites, marketing announcements, and incremental results.
- **Script writer** composes a spoken narration in dense technical prose — no filler, no hype words, numbers spelled out for TTS.
- **TTS** synthesizes the final MP3 (optional; works without it, producing text scripts only).

## Sources

| Source | What it finds |
|--------|--------------|
| [arxiv](https://arxiv.org) | cs.AI, cs.LG, cs.CL, cs.CV, cs.MA, stat.ML |
| [HuggingFace Daily Papers](https://huggingface.co/papers) | Community-curated trending papers |
| [GitHub Trending](https://github.com/trending) | New ML repos gaining traction (Python, Rust, C++) |
| [Papers With Code](https://paperswithcode.com) | SOTA benchmark results |
| [Semantic Scholar](https://www.semanticscholar.org) | Recent high-impact papers by topic |
| Lab blogs (RSS) | Google Research, Meta AI, OpenAI, Anthropic, DeepMind |

## Requirements

- Python 3.11+
- `aiohttp` (see `requirements.txt`)
- Any OpenAI-compatible LLM API (local or remote)
- TTS pipeline (optional — for audio output)

## Setup

```bash
git clone https://github.com/olympus-terminal/M3_netscape_navigator.git
cd M3_netscape_navigator
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your LLM endpoint and optional TTS paths
```

## Configuration

All user-specific settings are in environment variables (`.env` file or exported directly). See `.env.example` for the full list.

**Key settings in `config.py`:**

| Setting | Default | Description |
|---------|---------|-------------|
| `FOCUS_AREAS` | 12 topics | Research areas the LLM prioritizes |
| `TECHNICAL_DEPTH_THRESHOLD` | 6 | Minimum score (1–10) to keep |
| `NOVELTY_THRESHOLD` | 5 | Minimum novelty score |
| `RELEVANCE_THRESHOLD` | 5 | Minimum relevance score |
| `TARGET_AUDIO_HOURS` | 3.0 | Target digest length |
| `DOMAIN_DELAYS` | 60–300s | Per-domain fetch intervals |

Raise the thresholds for more selective output; lower them for more volume.

## Usage

```bash
# Full cycle: crawl → judge → compose script → synthesize audio
python agent.py

# Crawl and judge only (no TTS) — good for testing
python agent.py --crawl-only -v

# Compose + synthesize from the last judged set
python agent.py --digest-only

# Persistent daemon — one cycle per day
python agent.py --daemon

# Smoke test — verify LLM and source connectivity
python smoke_test.py
```

## Output

```
output/
├── raw/        # Judged items as JSON (scores, summaries, verdicts)
├── scripts/    # Narration scripts (plain text)
├── audio/      # Synthesized MP3 digests
└── digests/    # Per-digest metadata (items, scores, paths)
```

`state.json` in the project root tracks previously seen URLs across runs.

## How it works

A full cycle takes several hours by design. The crawlers deliberately pace themselves to mimic human browsing — one page every 2–5 minutes per domain, with random jitter. This is intentional: it avoids rate limiting, respects site ToS, and makes the traffic indistinguishable from a person reading papers.

While the crawlers are slow, the LLM never waits. Multiple crawlers feed a shared async queue, and the LLM pulls items as fast as it can judge them. The bottleneck is always the intentional fetch delay, not inference speed.

After judging, the agent sorts kept items by combined score, batches them into segments, and asks the LLM to compose a narration script — dense technical prose suitable for TTS. If a TTS pipeline is configured, it synthesizes the script into an MP3.

## License

MIT
