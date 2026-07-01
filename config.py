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

# === LLM inference endpoint (any OpenAI-compatible API) ===
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "default")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "none")
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "4096"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.3"))

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
BASE_DIR = Path(os.environ.get("AGENT_DATA_DIR", Path(__file__).parent))
OUTPUT_DIR = BASE_DIR / "output"
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
MAX_PAGES_PER_SOURCE_PER_RUN = 50
MAX_QUEUE_SIZE = 200

# === Content filtering ===
# LLM scores each piece 1–10 on these axes; must meet all thresholds to keep.
TECHNICAL_DEPTH_THRESHOLD = 6
NOVELTY_THRESHOLD = 5
RELEVANCE_THRESHOLD = 5

# === Digest settings ===
TARGET_AUDIO_HOURS = float(os.environ.get("TARGET_AUDIO_HOURS", "3.0"))
WORDS_PER_MINUTE = 150
TARGET_WORD_COUNT = int(TARGET_AUDIO_HOURS * 60 * WORDS_PER_MINUTE)

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
        "fields": "title,abstract,url,year,citationCount,openAccessPdf",
    },
    "lab_blogs": {
        "enabled": True,
        "feeds": [
            "https://blog.research.google/feeds/posts/default?alt=rss",
            "https://ai.meta.com/blog/rss/",
            "https://openai.com/blog/rss.xml",
            "https://www.anthropic.com/research/rss.xml",
            "https://deepmind.google/blog/rss.xml",
        ],
    },
}
