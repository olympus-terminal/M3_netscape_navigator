"""Remote runner for a versioned copy of the canonical screened Qwen3 toolkit.

Deploy beside media_to_tts.py, verify_tts_audio.py, run_offline.py,
voices.json and bundle.json. Voice assets and machine paths stay private.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import fcntl

ROOT = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def catalog():
    return json.loads((ROOT / "voices.json").read_text())


def checked_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise RuntimeError(f"Missing or invalid bundle file: {relative}")
    return path


def prepare_environment():
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    os.environ["HF_HOME"] = str(Path.home() / ".cache/huggingface-tts")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["WANDB_MODE"] = "disabled"


def preflight(voice):
    profiles = catalog()
    if voice not in profiles or profiles[voice].get("status") != "ready":
        raise RuntimeError(f"Voice {voice!r} has no complete reference/transcript profile")
    profile = profiles[voice]
    bundle = json.loads((ROOT / "bundle.json").read_text())
    for name, expected in bundle["sha256"].items():
        if sha256(checked_path(name)) != expected:
            raise RuntimeError(f"Bundle checksum mismatch: {name}")
    for key in ["audio", "transcript"]:
        checked_path(profile[key])
    if not checked_path(profile["transcript"]).read_text().strip():
        raise RuntimeError("Voice reference transcript is empty")
    for name in ["ffmpeg", "ffprobe"]:
        if not shutil.which(name):
            raise RuntimeError(f"Missing runtime program: {name}")
    import torch
    import qwen_tts  # noqa: F401
    import faster_whisper  # noqa: F401
    from huggingface_hub import snapshot_download
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3 narration requires CUDA")
    cache = Path(os.environ["HF_HOME"]) / "hub"
    snapshot = Path(snapshot_download(bundle["model"], revision=bundle["revision"], local_files_only=True))
    for repo in ["models--Systran--faster-whisper-medium", "models--Systran--faster-whisper-large-v3"]:
        if not (cache / repo / "refs/main").is_file():
            raise RuntimeError(f"Missing offline ASR cache: {repo}")
    for path in [snapshot, cache / "models--Systran--faster-whisper-medium", cache / "models--Systran--faster-whisper-large-v3"]:
        if any(p.is_symlink() and not p.exists() for p in path.rglob("*")):
            raise RuntimeError(f"Incomplete model cache: {path.name}")
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(mem["MemAvailable"].split()[0]) * 1024
    if available < 16 * 1024**3:
        raise RuntimeError("Less than 16 GiB available; defer narration until memory is free")
    if shutil.disk_usage(ROOT).free < 500 * 1024**2:
        raise RuntimeError("Less than 500 MiB of disk space available")
    os.environ["QWEN3_TTS_REVISION"] = bundle["revision"]
    return {
        "engine": "qwen3", "model": bundle["model"], "revision": bundle["revision"],
        "voice": voice, "voice_name": profile["name"], "cuda_available": True,
        "device": "cuda:0", "available_memory_bytes": available,
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "qwen-tts", "transformers", "faster-whisper"]},
        "reference_sha256": sha256(checked_path(profile["audio"])),
        "transcript_sha256": sha256(checked_path(profile["transcript"])),
        "pipeline_sha256": sha256(ROOT / "media_to_tts.py"),
        "verifier_sha256": sha256(ROOT / "verify_tts_audio.py"),
    }


def synthesize(voice, input_path, output_dir, audio_format):
    # One narrator per node; lock outside the versioned release directory.
    lock_path = Path.home() / ".cache/ai-assistant-qwen3.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A Qwen3 narration is already running on this node") from exc
        runtime = preflight(voice)
        source = Path(input_path).resolve(strict=True)
        run = Path(output_dir).resolve(strict=True)
        if not source.is_file() or not source.read_text().strip():
            raise RuntimeError("Narration source must be a nonempty text file")
        for name in ["media_to_tts.py", "verify_tts_audio.py", "run_offline.py"]:
            shutil.copy2(ROOT / name, run / name)
        prepared = run / "prepared.txt"
        cmd = [sys.executable, str(run / "media_to_tts.py"), str(source),
               "--no-synth", "--screen", "--output", str(prepared)]
        subprocess.run(cmd, check=True, cwd=run)
        if not prepared.is_file() or not prepared.read_text().strip():
            raise RuntimeError("Text extraction produced no narration")
        profile = catalog()[voice]
        audio_dir = run / "audio"
        audio_dir.mkdir(exist_ok=False)
        started = time.monotonic()
        cmd = [sys.executable, str(run / "run_offline.py"), str(source),
               "--engine", "qwen3", "--model", "base", "--device", "cuda:0",
               "--ref-audio", str(checked_path(profile["audio"])),
               "--ref-text", checked_path(profile["transcript"]).read_text().strip(),
               "--format", audio_format, "--screen", "--output-dir", str(audio_dir),
               "--output", str(prepared)]
        completed = subprocess.run(cmd, cwd=run)
        if completed.returncode not in (0, 2):
            raise RuntimeError(f"Screened synthesis failed with exit {completed.returncode}")
        outputs = list(audio_dir.glob(f"*.{audio_format}"))
        if len(outputs) != 1 or outputs[0].stat().st_size < 1024:
            raise RuntimeError("Pipeline did not produce exactly one nontrivial audio file")
        audio = outputs[0]
        report_path = audio.with_name(audio.stem + "_hallucination_report.json")
        screen_path = audio.with_name(audio.stem + "_text_screen.json")
        report = json.loads(report_path.read_text())
        screen = json.loads(screen_path.read_text())
        status = report["summary"]["status"]
        if status not in ("CLEAN", "REVIEW_NEEDED"):
            raise RuntimeError("Missing or invalid ASR screening status")
        if not report.get("text_words") or not report.get("asr_words"):
            raise RuntimeError("ASR verification did not compare nonempty transcripts")
        guard_path = audio.with_name(audio.stem + "_chunk_guard.json")
        guard = json.loads(guard_path.read_text()) if guard_path.exists() else {}
        if completed.returncode == 2 or screen["text_triggers"] or guard.get("unresolved", 0):
            status = "REVIEW_NEEDED"
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                "format=duration:stream=sample_rate", "-of", "json", str(audio)],
                               check=True, capture_output=True, text=True)
        probe = json.loads(probe.stdout)
        duration = float(probe["format"]["duration"])
        if duration <= 1:
            raise RuntimeError("Audio duration is invalid")
        result = {
            **runtime, "status": status, "duration_seconds": duration,
            "sample_rate": int(probe["streams"][0]["sample_rate"]),
            "size_bytes": audio.stat().st_size, "audio_sha256": sha256(audio),
            "input_sha256": sha256(source), "elapsed_seconds": time.monotonic() - started,
            "audio": str(audio.relative_to(run)),
            "screening_report": str(report_path.relative_to(run)),
            "text_screen": str(screen_path.relative_to(run)),
            "qc_scope": ["pre_synthesis_text_hazards", "qwen3_chunk_duration_guard", "ASR_content_alignment"],
            "asr_summary": report["summary"], "text_triggers": screen["text_triggers"],
        }
        (run / "tts_result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list-voices", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--voice", default="zv")
    ap.add_argument("--input")
    ap.add_argument("--output-dir")
    ap.add_argument("--format", choices=["mp3", "wav"], default="mp3")
    args = ap.parse_args()
    prepare_environment()
    if args.list_voices:
        result = catalog()
    elif args.preflight:
        result = preflight(args.voice)
    else:
        if not args.input or not args.output_dir:
            ap.error("--input and --output-dir are required")
        result = synthesize(args.voice, args.input, args.output_dir, args.format)
    print("TTS_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
