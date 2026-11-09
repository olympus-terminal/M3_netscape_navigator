"""Opt-in two-node narration, with ordered joining and retained per-part QC."""
from __future__ import annotations

from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid

from config import SPARK_TTS_ROOT, SPARK_TTS_PYTHON, SPARK_TTS_VOICE
from spark_tts import _ssh, _worker, inspect, narrate

log = logging.getLogger(__name__)
IDENTITY_KEYS = (
    "engine", "model", "revision", "voice", "reference_sha256",
    "transcript_sha256", "pipeline_sha256", "verifier_sha256",
)


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _save(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def split_narration(text):
    """Split prepared prose near its word midpoint without losing characters.

    Prefer paragraph boundaries in the middle 40% of the text, otherwise use
    sentence endings. Extraction happens first, so Markdown syntax is not split.
    """
    words = [m.end() for m in re.finditer(r"\S+", text)]
    if len(words) < 2:
        raise ValueError("Narration is too short to split across two nodes")
    midpoint = len(words) / 2

    def usable(matches):
        return [m.end() for m in matches if text[:m.end()].strip() and text[m.end():].strip()]

    paragraphs = usable(re.finditer(r"\n[ \t]*\n+", text))
    balanced = [p for p in paragraphs if 0.3 * len(words) <= bisect_right(words, p) <= 0.7 * len(words)]
    # The whitespace requirement avoids splitting decimal numbers and URLs.
    sentences = usable(re.finditer(r'''[.!?]["'”’\)\]]*\s+(?=[A-Z0-9"“‘'])''', text))
    candidates = balanced or sentences or paragraphs
    if not candidates:
        raise ValueError("No paragraph or sentence boundary; use single-node narration")
    cut = min(candidates, key=lambda p: abs(bisect_right(words, p) - midpoint))
    return [text[:cut], text[cut:]]


def prepare_text(source, run, *, host, root, python):
    """Run the canonical cleaner once before partitioning its spoken text."""
    remote = Path(root) / "runs" / ("prepare_" + uuid.uuid4().hex)
    prep = run / "preparation"
    prep.mkdir()
    shutil.copyfile(source, prep / "input.md")
    _ssh(host, ["mkdir", "-p", str(remote)], timeout=30)
    subprocess.run(["rsync", "-rt", "--protect-args", "--", str(prep / "input.md"),
                    f"{host}:{remote}/input.md"], check=True, timeout=120)
    _ssh(host, ["cp", str(Path(root) / "media_to_tts.py"), str(remote / "media_to_tts.py")], timeout=30)
    with (prep / "prepare.log").open("w") as output:
        _ssh(host, [python, str(remote / "media_to_tts.py"), str(remote / "input.md"),
                    "--no-synth", "--screen", "--output", str(remote / "prepared.txt")],
             stdout=output, stderr=subprocess.STDOUT, timeout=120)
    subprocess.run(["rsync", "-rt", "--protect-args", "--", f"{host}:{remote}/prepared.txt",
                    str(prep / "prepared.txt")], check=True, timeout=120)
    text = (prep / "prepared.txt").read_text()
    if not text.strip():
        raise RuntimeError("Canonical extraction produced empty narration")
    return text


def _probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_name,sample_rate,channels",
         "-of", "json", str(path)], check=True, capture_output=True, text=True, timeout=30,
    )
    data = json.loads(result.stdout)
    if len(data["streams"]) != 1:
        raise RuntimeError("Expected one audio stream in each narration part")
    return data["streams"][0], float(data["format"]["duration"])


def verify_spoken_text(source, audio):
    """Ensure the second cleaning pass did not change a prepared part."""
    spoken = audio.with_name(audio.stem + "_spoken.txt")
    if not spoken.is_file() or " ".join(spoken.read_text().split()) != " ".join(source.read_text().split()):
        raise RuntimeError("Spoken text differs from its planned narration part")
    return spoken


def join_audio(parts, run, status):
    """Join verified parts in source order, without re-encoding the MP3 audio."""
    stage = run / "join"
    stage.mkdir()
    streams, durations, entries = [], [], []
    for i, (_, paths) in enumerate(parts, 1):
        stream, duration = _probe(paths["audio"])
        streams.append(stream)
        durations.append(duration)
        # Generated relative filenames avoid ffconcat escaping of user paths.
        target = stage / f"part_{i:02d}.mp3"
        try:
            os.link(paths["audio"], target)
        except OSError:
            shutil.copyfile(paths["audio"], target)
        entries.append(f"file '{target.name}'\n")
    if any(s != streams[0] for s in streams[1:]) or streams[0]["codec_name"] != "mp3":
        raise RuntimeError("Narration parts have incompatible audio formats")
    playlist = stage / "parts.ffconcat"
    playlist.write_text("ffconcat version 1.0\n" + "".join(entries))
    temporary = stage / "joined.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "concat", "-safe", "1",
                    "-i", playlist.name, "-map", "0:a:0", "-c:a", "copy", temporary.name],
                   cwd=stage, check=True, capture_output=True, text=True, timeout=600)
    _, duration = _probe(temporary)
    if duration <= 1 or abs(duration - sum(durations)) > 0.5:
        raise RuntimeError("Joined duration does not match the sum of its parts")
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", str(temporary),
                    "-f", "null", "-"], check=True, capture_output=True, text=True, timeout=600)
    suffix = "_REVIEW_NEEDED" if status == "REVIEW_NEEDED" else ""
    return temporary, run / f"narration{suffix}.mp3", duration, durations


def narrate_parallel(input_path, output_dir, *, hosts, root=SPARK_TTS_ROOT,
                     python=SPARK_TTS_PYTHON, voice=SPARK_TTS_VOICE):
    hosts = list(hosts)
    if len(hosts) != 2 or len(set(hosts)) != 2:
        raise ValueError("Choose two different SSH host aliases")
    source = Path(input_path).resolve(strict=True)
    if not source.is_file() or source.suffix.lower() not in {".txt", ".md", ".markdown"} or not source.read_text().strip():
        raise ValueError("Narration input must be a nonempty Markdown or text file")
    _worker(root, python)
    for binary in ["ffmpeg", "ffprobe", "rsync", "ssh"]:
        if not shutil.which(binary):
            raise RuntimeError(f"Missing required program: {binary}")
    run = Path(output_dir).resolve() / ("parallel_" + uuid.uuid4().hex)
    run.mkdir(parents=True, exist_ok=False)
    manifest_path = run / "parallel_manifest.json"
    manifest = {"status": "RUNNING", "source": str(source), "source_sha256": _sha256(source),
                "voice": voice, "hosts": hosts, "parts": [], "mode": "two_node_narration"}
    _save(manifest_path, manifest)
    final = None
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            checks = [pool.submit(inspect, host=h, root=root, python=python, voice=voice) for h in hosts]
            profiles = [f.result() for f in checks]
        identity = {k: profiles[0][k] for k in IDENTITY_KEYS}
        if any({k: p[k] for k in IDENTITY_KEYS} != identity for p in profiles):
            raise RuntimeError("Nodes do not have matching models, voices, and screening pipelines")
        manifest["identity"] = identity
        text = prepare_text(source, run, host=hosts[0], root=root, python=python)
        split = split_narration(text)
        if "".join(split) != text:
            raise RuntimeError("Split did not preserve the complete narration")
        manifest["prepared_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        cursor = 0
        sources = []
        for i, (host, part) in enumerate(zip(hosts, split), 1):
            part_dir = run / f"part_{i:02d}"
            part_dir.mkdir()
            path = part_dir / "narration.md"
            path.write_text(part)
            sources.append(path)
            manifest["parts"].append({"index": i, "host": host, "status": "RUNNING",
                                      "source": str(path), "sha256": _sha256(path),
                                      "start_character": cursor, "end_character": cursor + len(part),
                                      "words": len(part.split())})
            cursor += len(part)
        _save(manifest_path, manifest)
        results = [None, None]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {}
            for i, (host, path) in enumerate(zip(hosts, sources)):
                log.info("Narrating part %s on %s (%s words)", i + 1, host, manifest["parts"][i]["words"])
                futures[pool.submit(narrate, path, path.parent, host=host, root=root, python=python,
                                    voice=voice, audio_format="mp3")] = i
            for future in as_completed(futures):
                i = futures[future]
                try:
                    record, paths = future.result()
                    if {k: record[k] for k in IDENTITY_KEYS} != identity:
                        raise RuntimeError("Part identity changed after preflight")
                    spoken = verify_spoken_text(sources[i], paths["audio"])
                    paths["spoken_text"] = spoken
                    results[i] = (record, paths)
                    manifest["parts"][i].update({"status": record["status"], "result": record,
                                                 "paths": {k: str(v) for k, v in paths.items()}})
                    log.info("Part %s finished: %s", i + 1, record["status"])
                except Exception as exc:
                    manifest["parts"][i].update({"status": "FAILED", "error": str(exc)})
                _save(manifest_path, manifest)
        if any(r is None for r in results):
            raise RuntimeError("A narration part failed; no combined audio was published")
        status = "REVIEW_NEEDED" if any(r[0]["status"] == "REVIEW_NEEDED" for r in results) else "CLEAN"
        temporary, final, duration, durations = join_audio(results, run, status)
        offset = 0.0
        for part, seconds in zip(manifest["parts"], durations):
            part.update({"start_seconds": offset, "duration_seconds": seconds})
            offset += seconds
        manifest.update({"status": status, "audio_path": str(final), "duration_seconds": duration,
                         "audio_sha256": _sha256(temporary), "size_bytes": temporary.stat().st_size,
                         "qc_scope": ["per_part_text_and_ASR_screening", "joined_duration_and_full_decode"],
                         "manifest_path": str(manifest_path)})
        temporary.replace(final)
        try:
            _save(manifest_path, manifest)
        except Exception:
            final.replace(temporary)
            raise
        return manifest
    except Exception as exc:
        manifest.update({"status": "FAILED", "error": str(exc)})
        _save(manifest_path, manifest)
        raise RuntimeError(f"Parallel narration failed; diagnostics: {manifest_path}") from exc
