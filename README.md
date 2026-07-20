# Codex AI Research Audio Agent

Autonomous agent that crawls current AI research, filters it through isolated
`codex exec` workers, writes a technical narration script, and optionally turns
the script into a screened MP3 digest.

The two Codex roles are deliberately narrow:

```text
[Crawlers] ──async──▶ [Queue] ──batches──▶ [Codex Judge]
                                           │
                                           ▼
                                   [Codex Script Writer]
                                           │
                                           ▼
                              [Screened Coqui TTS, p246]
```

- The **judge** scores batches of papers for technical depth, novelty, and
  relevance using a strict JSON schema.
- The **writer** turns only accepted work into flowing, TTS-ready narration.
- Every Codex invocation is ephemeral, non-interactive, read-only, tool-disabled,
  and isolated in a temporary directory. Scraped text is sent through stdin as
  untrusted data rather than included in the trusted instruction.
- The local TTS path uses the published `gpu-tts-toolkit` with speaker `p246`,
  pre/post-synthesis screening, and MP3 verification.

## Sources

- arXiv: `cs.AI`, `cs.LG`, `cs.CL`, `cs.CV`, `cs.MA`, and `stat.ML`
- Hugging Face Daily Papers
- GitHub Trending
- Papers With Code / Hugging Face paper metadata
- Semantic Scholar
- Major AI lab RSS feeds

## Requirements

- Python 3.11+
- An installed Codex CLI authenticated with `codex login`
- `aiohttp` and `python-dotenv` from `requirements.txt`
- Optional audio: the companion
  [`gpu-tts-toolkit`](https://github.com/olympus-terminal/gpu-tts-toolkit), a
  Python environment containing Coqui TTS, CUDA when available, FFmpeg, and
  `ffprobe`

## Setup

```bash
git clone https://github.com/olympus-terminal/M3_netscape_navigator.git
cd M3_netscape_navigator
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
codex login status
```

For local audio, clone the companion toolkit next to this repository and set
its absolute paths in `.env`:

```bash
git clone https://github.com/olympus-terminal/gpu-tts-toolkit.git ../gpu-tts-toolkit
git -C ../gpu-tts-toolkit checkout 1652e425a7ce0066dd0fba1fd540863a7fcbbbab
```

The pinned toolkit revision keeps chunk files until post-synthesis QC,
verifies that every planned chunk exists and decodes, then allows the toolkit
to clean intermediates. The agent fails its TTS preflight if that contract is
missing. `patches/gpu-tts-screening.patch` is retained as an auditable backport
for toolkit revision `7bdcde21871eb816c62432c7bb41e22336dc66b6`; apply that
zero-context patch with `git apply --unidiff-zero`.

```dotenv
TTS_PYTHON=/path/to/miniconda3/envs/tts-app/bin/python
TTS_SCRIPT=/path/to/gpu-tts-toolkit/media_to_tts.py
TTS_VOICE=p246
TTS_FORMAT=mp3
```

The public toolkit's current `media_to_tts.py` path is Coqui-based. Its generic
SLURM example is not a ready Jubail/Qwen3 backend, so this setup intentionally
uses local Coqui rather than submitting an unverified cluster job.

## Configuration

Important environment settings:

| Setting | Default | Purpose |
|---|---:|---|
| `CODEX_BIN` | detected from `PATH` | Absolute Codex CLI path is recommended for cron/systemd |
| `CODEX_MODEL` | empty | Empty uses the CLI's current default model |
| `CODEX_JUDGE_BATCH_SIZE` | `12` | Papers per structured Codex judge call |
| `CODEX_JUDGE_WORKERS` | `1` | Concurrent judge processes; keep low to control quota |
| `CODEX_MAX_JUDGE_ATTEMPTS` | `2` | Bounded attempts before a URL is left for the next run |
| `MAX_PAGES_PER_SOURCE_PER_RUN` | `20` | Daily candidate cap per source |
| `TARGET_AUDIO_HOURS` | `3.0` | Target narration length |
| `MAX_NARRATION_WORDS_PER_ITEM` | `500` | Prevents sparse days from forcing padded, overlong segments |
| `TTS_VOICE` | `p246` | Coqui VCTK speaker used for narration |

One fresh `codex exec` session has meaningful fixed context overhead. Keep
batching enabled; do not revert to one invocation per paper unless quota use is
acceptable.

## Usage

```bash
# Local preflight only; no model inference or source-network call
.venv/bin/python smoke_test.py

# Explicit opt-in live checks
.venv/bin/python smoke_test.py --live-codex
.venv/bin/python smoke_test.py --live-sources
.venv/bin/python smoke_test.py --tts

# Full cycle: crawl, judge, compose, and synthesize
.venv/bin/python agent.py

# Crawl and judge without composing or TTS
.venv/bin/python agent.py --crawl-only -v

# Compose and synthesize from the most recent judged set
.venv/bin/python agent.py --digest-only

# One cycle per day
.venv/bin/python agent.py --daemon
```

## Output

```text
output/
├── raw/        # Accepted items, scores, summaries, and judge provenance
├── scripts/    # Plain-text narration scripts
├── audio/      # Synthesized MP3 directories and QC reports
└── digests/    # Per-digest metadata
```

`state.json` tracks previously seen URLs. Items whose Codex attempts fail are
removed from the seen set so a later run can retry them.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The deterministic suite uses a fake Codex executable and does not consume model
quota. Live inference remains opt-in through `smoke_test.py --live-codex`.

## License

MIT
