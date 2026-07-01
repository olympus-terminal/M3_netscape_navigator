# ai-research-agent

Persistent autonomous agent that crawls bleeding-edge AI research, filters it through a local LLM (M3 on DGX Spark), and produces daily ~3-hour audio digests via TTS.

## Architecture

```
[Crawlers] --async--> [Queue] --serial--> [M3 Judge] --> [Script Writer] --> [TTS] --> MP3
```

- **Crawlers** run concurrently across domains with human-paced delays (2-5 min jitter per domain)
- **M3 stays saturated** — always has the next item ready from the queue, no idle GPU time
- **Judge** scores each piece on technical depth, novelty, and relevance (1-10); rejects pop-sci and rehashes
- **Script writer** composes a spoken narration via M3 — dense, technical, no filler
- **TTS** synthesizes via gpu-tts-toolkit (Coqui VITS, voice p246)

## Sources

- arxiv (cs.AI, cs.LG, cs.CL, cs.CV, cs.MA, stat.ML)
- HuggingFace daily papers
- GitHub trending (Python, Rust, C++)
- Papers With Code
- Semantic Scholar
- Lab blogs via RSS (Google Research, Meta AI, OpenAI, Anthropic, DeepMind)

## Requirements

- Python 3.11+
- `aiohttp` (`pip install -r requirements.txt`)
- M3 running on DGX Spark (OpenAI-compatible API at `http://192.168.0.127:8080/v1`)
- `tts-app` conda env with gpu-tts-toolkit for audio synthesis

## Usage

```bash
# One full cycle: crawl + judge + compose script + synthesize audio
python agent.py

# Crawl and judge only (no TTS) — good for testing
python agent.py --crawl-only -v

# Compose + synthesize from last judged set (skip crawling)
python agent.py --digest-only

# Persistent daemon (one cycle per day)
python agent.py --daemon
```

## Configuration

Edit `config.py` to tune:

- `FOCUS_AREAS` — what topics M3 looks for
- `TECHNICAL_DEPTH_THRESHOLD` / `NOVELTY_THRESHOLD` / `RELEVANCE_THRESHOLD` — raise for more selective, lower for more volume
- `DOMAIN_DELAYS` — per-domain fetch intervals (human-pacing)
- `SOURCES` — enable/disable individual sources
- `TARGET_AUDIO_HOURS` — target digest length (default 3.0)

## Output

```
output/
  raw/        # judged items as JSON (what M3 kept and why)
  scripts/    # narration scripts (plain text)
  audio/      # synthesized MP3 digests
  digests/    # metadata (what went into each digest)
```
