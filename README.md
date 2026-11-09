# AI Research Audio Agent

Autonomous agent that crawls current AI research, filters it through isolated
`codex exec` workers or a local model API, writes a technical narration script, and optionally turns
the script into a screened MP3 digest.

The two language-model roles are deliberately narrow:

```text
[Crawlers] ──async──▶ [Queue] ──batches──▶ [Judge]
                                           │
                                           ▼
                                   [Script Writer]
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
git -C ../gpu-tts-toolkit checkout 0e83937d613a561fe1625091caeb36889b2f9138
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

Judging and writing can also use an OpenAI-compatible model through a recording
proxy. Set `LLM_BACKEND=openai_compatible`, `LLM_BASE_URL` to the proxy's `/v1`
URL, and `LLM_MODEL` to its exact advertised model ID. The client uses tool-free
chat completions with JSON schemas and checks the returned model, completion
status, and the existing judge/writer contracts. It rejects truncated responses
and tool calls. Endpoint failure follows the existing bounded retry policy;
it never silently falls back to a cloud model.

For MiMo Flash MXFP4 on a two-GPU llama.cpp server, start with one server slot,
32,768 context tokens, one judge worker, batches of six, and a 1,800-second judge
timeout. `LLM_MAX_TOKENS` defaults to 16,384 and `LLM_ENABLE_THINKING` to false.
Set `LLM_DEPLOYMENT` to distinguish serving stacks and quantizations in digest
metadata. The recording proxy retains requests, responses, and token usage;
each digest also records the configured language-model identity. Codex remains
available with `LLM_BACKEND=codex`. These settings do not change speech synthesis
or the user's global Codex configuration.

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

### Optional Spark Qwen3 narration

With a provisioned, versioned narration bundle, set `SPARK_TTS_HOST` to its SSH
alias, `SPARK_TTS_ROOT` to the remote bundle directory, and `SPARK_TTS_PYTHON`
to the Qwen3 environment's Python. Voice references, transcripts, and the
`voices.json` catalog stay outside Git. Inspect or use the installed profiles:

```bash
python3 spark_tts.py --list-voices
python3 spark_tts.py --preflight --voice zv
python3 spark_tts.py narration.md --voice zv
# Choose another configured node and voice for a single narration:
python3 spark_tts.py narration.md --host tts-node --voice ps
# Explicitly split a longer narration between two nodes:
python3 spark_tts.py narration.md --hosts tts-node-a tts-node-b --voice ps
```

`--hosts` is an opt-in command for standalone narrations. The daily digest still
uses its existing single-node/local workflow. The coordinator cleans the text
with the canonical pipeline, splits near the word midpoint at a paragraph or
sentence boundary, and runs both parts concurrently with the same voice and
model revision. It joins their MP3s in source order without re-encoding, checks
the combined duration and decodes the whole result. Both parts' text/ASR reports
remain available, with their offsets recorded in `parallel_manifest.json`. Any
review flag marks the combined file `REVIEW_NEEDED`; a failed part prevents
publication of a combined recording. Logs and completed parts are retained.

The standalone command returns the local audio path and screening status. It
keeps the spoken text, text screen, ASR report, transcript, hashes, and runtime
manifest beside the audio. Reference-only entries without confirmed transcripts
are preserved but cannot be selected for synthesis.

To select this route for future daily digests, set `TTS_BACKEND=spark_qwen3`
and `SPARK_TTS_VOICE` in `.env`, then reload the daemon after its current cycle.
`TTS_BACKEND=local_coqui` remains the default. There is no automatic voice or
engine substitution on failure. The remote worker runs one narration per node,
checks at least 16 GiB of available memory, and uses the canonical pipeline's
text screening, chunk-duration guard, and ASR comparison. Review flags are
retained in the delivered filename and metadata. `asr_clean` describes this
route's screening scope; the older Coqui route retains `structural_clean`.

The remote bundle contains `spark_tts_worker.py`, the canonical
`media_to_tts.py` and `verify_tts_audio.py`, an offline snapshot launcher,
`voices.json`, and `bundle.json` with the pinned model revision and file hashes.
Each narration copies the pipeline into a unique run directory. Its model and
Whisper caches must already be available locally on the node.

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

## Automatic startup

Install or refresh the per-user startup service from the current checkout:

```bash
.venv/bin/python agent.py --install-service
```

The command records the current repository and virtual-environment paths,
enables and starts the service, and installs a desktop-session fallback for an
encrypted home directory. The service starts again with the user session after
a reboot and restarts after an unexpected process failure. If the repository
is moved or the virtual environment is recreated elsewhere, rerun the same
command from the new location; no separate deployment copy is needed.

## Output

Finished audio is placed directly in the repository root with a local-time
timestamp and UTC offset. A same-named Markdown file accompanies every audio
artifact with the covered studies, clickable original sources, source-listed
authors, any source-provided affiliations, selection scores, and assistant
summaries:

```text
ai_research_digest_2026-07-23_11-30-00+0400.mp3
ai_research_digest_2026-07-23_11-30-00+0400.md
ai_research_digest_2026-07-23_11-30-00+0400_REVIEW_NEEDED.mp3
ai_research_digest_2026-07-23_11-30-00+0400_REVIEW_NEEDED.md
```

Files carrying `REVIEW_NEEDED` are complete and structurally verified, but have
a screening flag that still requires inspection. They are never silently called
clean. A historical file carrying `MANUALLY_ACCEPTED` has an explicit acceptance
record while its original machine report remains unchanged. Internal evidence
remains under `output/`:

```text
output/
├── raw/        # Accepted items, scores, summaries, and judge provenance
├── scripts/    # Plain-text narration scripts and internal Markdown source notes
├── audio/      # TTS working files, QC reports, and manifests
└── digests/    # Per-digest metadata
```

`state.json` tracks previously seen URLs. Items whose Codex attempts fail are
removed from the seen set so a later run can retry them. A synthesis failure is
retried after one hour; a complete `REVIEW_NEEDED` artifact is retained for
human inspection without regenerating the same multi-hour digest.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The deterministic suite uses a fake Codex executable and does not consume model
quota. Live inference remains opt-in through `smoke_test.py --live-codex`.

## License

MIT
