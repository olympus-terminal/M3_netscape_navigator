"""Configuration for the AI research agent.

All user-specific paths and endpoints are read from environment variables,
with sensible defaults for local development. Copy .env.example to .env
and edit to match your setup, or export the variables directly.
"""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _env_positive_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


# === Codex agents ===
# Saved `codex login` authentication is reused. CODEX_MODEL is intentionally
# empty by default so the installed CLI can select its current default model.
CODEX_BIN = os.environ.get("CODEX_BIN", shutil.which("codex") or "codex")
CODEX_MODEL = os.environ.get("CODEX_MODEL", "").strip()
CODEX_IGNORE_USER_CONFIG = _env_bool("CODEX_IGNORE_USER_CONFIG", True)
CODEX_JUDGE_TIMEOUT_SECONDS = _env_positive_int("CODEX_JUDGE_TIMEOUT_SECONDS", 900)
CODEX_WRITER_TIMEOUT_SECONDS = _env_positive_int("CODEX_WRITER_TIMEOUT_SECONDS", 1800)
CODEX_JUDGE_BATCH_SIZE = _env_positive_int("CODEX_JUDGE_BATCH_SIZE", 12)
CODEX_JUDGE_BATCH_WAIT_SECONDS = _env_positive_float("CODEX_JUDGE_BATCH_WAIT_SECONDS", 5)
CODEX_JUDGE_WORKERS = _env_positive_int("CODEX_JUDGE_WORKERS", 1)
CODEX_MAX_JUDGE_ATTEMPTS = _env_positive_int("CODEX_MAX_JUDGE_ATTEMPTS", 2)
CODEX_MAX_WRITER_ATTEMPTS = _env_positive_int("CODEX_MAX_WRITER_ATTEMPTS", 2)
CODEX_RETRY_DELAY_SECONDS = _env_positive_float("CODEX_RETRY_DELAY_SECONDS", 5)

# === TTS (optional — omit to skip audio synthesis) ===
TTS_CONDA_PYTHON = os.environ.get("TTS_PYTHON", shutil.which("python3") or "python3")
TTS_SCRIPT = os.environ.get("TTS_SCRIPT", "")
TTS_SCRIPT_DIR = str(Path(TTS_SCRIPT).parent) if TTS_SCRIPT else ""
TTS_VOICE = os.environ.get("TTS_VOICE", "p246")
TTS_FORMAT = os.environ.get("TTS_FORMAT", "mp3")
TTS_ENABLED = bool(TTS_SCRIPT and Path(TTS_SCRIPT).exists())

# === yt-dlp (optional) ===
YTDLP_BIN = os.environ.get("YTDLP_BIN", shutil.which("yt-dlp") or "yt-dlp")

# === Output ===
PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(os.environ.get("AGENT_DATA_DIR", PROJECT_DIR))
OUTPUT_DIR = BASE_DIR / "output"
# Finished, user-facing audio belongs at the repository root. The nested
# output tree is reserved for scripts, reports, manifests, and other internals.
DELIVERY_DIR = PROJECT_DIR
DIGEST_DIR = OUTPUT_DIR / "digests"
AUDIO_DIR = OUTPUT_DIR / "audio"
SCRIPT_DIR_OUT = OUTPUT_DIR / "scripts"
RAW_DIR = OUTPUT_DIR / "raw"
STATE_FILE = BASE_DIR / "state.json"
LOG_FILE = BASE_DIR / "agent.log"

# === Crawl behavior ===
# Per-domain minimum delay in seconds (with jitter added on top).
# Deliberately slow to mimic human browsing and respect rate limits.
DOMAIN_DELAYS = {
    "arxiv.org": (120, 300),
    "api.semanticscholar.org": (60, 180),
    "huggingface.co": (90, 240),
    "github.com": (90, 240),
    "paperswithcode.com": (120, 300),
    "default": (60, 180),
}

MAX_CONCURRENT_FETCHERS = 4
MAX_PAGES_PER_SOURCE_PER_RUN = _env_positive_int("MAX_PAGES_PER_SOURCE_PER_RUN", 20)

# === Content filtering ===
# LLM scores each piece 1–10 on these axes; must meet all thresholds to keep.
TECHNICAL_DEPTH_THRESHOLD = 6
NOVELTY_THRESHOLD = 5
RELEVANCE_THRESHOLD = 5

# === Digest settings ===
TARGET_AUDIO_HOURS = _env_positive_float("TARGET_AUDIO_HOURS", 3.0)
WORDS_PER_MINUTE = 150
TARGET_WORD_COUNT = int(TARGET_AUDIO_HOURS * 60 * WORDS_PER_MINUTE)
MAX_NARRATION_WORDS_PER_ITEM = _env_positive_int("MAX_NARRATION_WORDS_PER_ITEM", 500)

# === Research focus areas (guides the LLM's editorial taste) ===
FOCUS_AREAS = [
    "novel neural architectures and training methods",
    "scaling laws and emergent capabilities",
    "reasoning, planning, and agentic systems",
    "multimodal models (vision-language, video, audio)",
    "efficiency: quantization, distillation, pruning, sparse methods",
    "alignment, RLHF, constitutional AI, safety research",
    "open-source model releases and benchmark results",
    "inference optimization: speculative decoding, KV cache, serving",
    "code generation and software engineering agents",
    "scientific AI: protein folding, drug discovery, materials, climate",
    "robotics and embodied AI",
    "new datasets and evaluation methodologies",
]

# === Sources ===
SOURCES = {
    "arxiv": {
        "enabled": True,
        "categories": ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.MA", "stat.ML"],
        "base_url": "https://export.arxiv.org/api/query",
        "max_results_per_query": 50,
    },
    "huggingface_papers": {
        "enabled": True,
        "url": "https://huggingface.co/papers",
    },
    "github_trending": {
        "enabled": True,
        "url": "https://github.com/trending",
        "languages": ["python", "rust", "c++"],
        "since": "daily",
    },
    "papers_with_code": {
        "enabled": True,
        "url": "https://paperswithcode.com/latest",
    },
    "semantic_scholar": {
        "enabled": True,
        "base_url": "https://api.semanticscholar.org/graph/v1",
        "fields": "title,abstract,url,year,citationCount,openAccessPdf,authors",
    },
    "lab_blogs": {
        "enabled": True,
        "feeds": [
            "https://blog.research.google/feeds/posts/default?alt=rss",
            "https://engineering.fb.com/feed/",
            "https://openai.com/blog/rss.xml",
            "https://www.anthropic.com/research/rss.xml",
            "https://deepmind.google/blog/rss.xml",
        ],
    },
}
