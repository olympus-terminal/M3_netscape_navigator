"""Configuration for the AI research agent."""

from pathlib import Path

# === M3 inference endpoint ===
M3_BASE_URL = "http://192.168.0.127:8080/v1"
M3_MODEL = "m3"  # llama-server ignores this but the API requires it
M3_MAX_TOKENS = 4096
M3_TEMPERATURE = 0.3

# === TTS ===
TTS_CONDA_PYTHON = "/home/drn2/miniconda3/envs/tts-app/bin/python"
TTS_SCRIPT = "/media/drn2/External/working-tts-gpu/gpu-tts-toolkit/media_to_tts.py"
TTS_SCRIPT_DIR = str(Path(TTS_SCRIPT).parent)
TTS_VOICE = "p246"
TTS_FORMAT = "mp3"

# === yt-dlp ===
YTDLP_BIN = "/home/drn2/miniconda3/bin/yt-dlp"

# === Output ===
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
DIGEST_DIR = OUTPUT_DIR / "digests"
AUDIO_DIR = OUTPUT_DIR / "audio"
SCRIPT_DIR_OUT = OUTPUT_DIR / "scripts"
RAW_DIR = OUTPUT_DIR / "raw"
STATE_FILE = BASE_DIR / "state.json"
LOG_FILE = BASE_DIR / "agent.log"

# === Crawl settings ===
# Per-domain minimum delay in seconds (with jitter added on top)
DOMAIN_DELAYS = {
    "arxiv.org": (120, 300),       # 2-5 min between arxiv fetches
    "api.semanticscholar.org": (60, 180),
    "huggingface.co": (90, 240),
    "github.com": (90, 240),
    "paperswithcode.com": (120, 300),
    "default": (60, 180),          # any unlisted domain
}

MAX_CONCURRENT_FETCHERS = 4  # parallel crawler threads (one per domain group)
MAX_PAGES_PER_SOURCE_PER_RUN = 50  # safety cap per source per daily cycle
MAX_QUEUE_SIZE = 200  # max items waiting for M3

# === Content filtering ===
# M3 scores each piece 1-10 on these axes; must meet thresholds to keep
TECHNICAL_DEPTH_THRESHOLD = 6  # skip pop-sci / rehashes
NOVELTY_THRESHOLD = 5          # skip incremental / obvious extensions
RELEVANCE_THRESHOLD = 5        # broad AI/ML scope

# === Digest settings ===
TARGET_AUDIO_HOURS = 3.0
WORDS_PER_MINUTE = 150  # TTS speaking rate
TARGET_WORD_COUNT = int(TARGET_AUDIO_HOURS * 60 * WORDS_PER_MINUTE)  # ~27,000 words

# === Research focus areas (guides M3's taste) ===
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
