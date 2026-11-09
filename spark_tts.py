"""Select a cloned voice and narrate a text file on a configured Spark node."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import uuid

from config import SPARK_TTS_HOST, SPARK_TTS_ROOT, SPARK_TTS_PYTHON, SPARK_TTS_VOICE


def _ssh(host, argv, **kwargs):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", host):
        raise ValueError("Configure an SSH host alias for the narration node")
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, shlex.join(argv)],
        check=True, **kwargs,
    )


def _worker(root, python):
    if not Path(root).is_absolute() or not Path(python).is_absolute():
        raise ValueError("SPARK_TTS_ROOT and SPARK_TTS_PYTHON must be absolute remote paths")
    return [python, str(Path(root) / "spark_tts_worker.py")]


def _result(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith("TTS_RESULT "):
            return json.loads(line.removeprefix("TTS_RESULT "))
    raise RuntimeError("Remote narrator returned no result record")


def inspect(*, host=SPARK_TTS_HOST, root=SPARK_TTS_ROOT, python=SPARK_TTS_PYTHON,
            voice=SPARK_TTS_VOICE, list_voices=False):
    cmd = _worker(root, python) + (["--list-voices"] if list_voices else ["--preflight", "--voice", voice])
    result = _ssh(host, cmd, capture_output=True, text=True, timeout=120)
    return _result(result.stdout)


def verify_download(run_dir):
    """Validate downloaded audio and QC evidence before any publication."""
    run_dir = Path(run_dir).resolve()
    record = json.loads((run_dir / "tts_result.json").read_text())
    paths = {}
    for key in ["audio", "screening_report", "text_screen"]:
        path = (run_dir / record[key]).resolve()
        if not path.is_relative_to(run_dir) or not path.is_file():
            raise RuntimeError(f"Missing or invalid downloaded {key}")
        paths[key] = path
    audio = paths["audio"]
    if audio.stat().st_size != record["size_bytes"] or audio.stat().st_size < 1024:
        raise RuntimeError("Downloaded audio size mismatch")
    h = hashlib.sha256()
    with audio.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != record["audio_sha256"]:
        raise RuntimeError("Downloaded audio checksum mismatch")
    report = json.loads(paths["screening_report"].read_text())
    screen = json.loads(paths["text_screen"].read_text())
    if record["status"] not in {"CLEAN", "REVIEW_NEEDED"} or report["summary"]["status"] not in {"CLEAN", "REVIEW_NEEDED"}:
        raise RuntimeError("Invalid audio screening status")
    if record["status"] == "CLEAN" and (report["summary"]["status"] != "CLEAN" or screen["text_triggers"]):
        raise RuntimeError("Clean result contradicts its screening evidence")
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=noprint_wrappers=1:nokey=1", str(audio)],
                           check=True, capture_output=True, text=True, timeout=30)
    duration = float(probe.stdout.strip())
    if duration <= 1 or abs(duration - record["duration_seconds"]) > 0.5:
        raise RuntimeError("Downloaded audio duration mismatch")
    return record, paths


def narrate(input_path, output_dir, *, host=SPARK_TTS_HOST, root=SPARK_TTS_ROOT,
            python=SPARK_TTS_PYTHON, voice=SPARK_TTS_VOICE, audio_format="mp3"):
    source = Path(input_path).resolve(strict=True)
    if not source.is_file() or source.suffix.lower() not in {".txt", ".md", ".markdown"}:
        raise ValueError("Narration input must be a Markdown or plain-text file")
    if not source.read_text(encoding="utf-8").strip():
        raise ValueError("Narration input is empty")
    if audio_format not in {"mp3", "wav"}:
        raise ValueError("Audio format must be mp3 or wav")
    cmd = _worker(root, python)
    run_id = "narration_" + uuid.uuid4().hex
    remote = Path(root) / "runs" / run_id
    local = Path(output_dir).resolve() / run_id
    local.mkdir(parents=True, exist_ok=False)
    # Markdown input avoids the canonical plain-text cleaner's citation-number
    # stripping. The source content itself is preserved byte for byte.
    shutil.copyfile(source, local / "input.md")
    _ssh(host, ["mkdir", "-p", str(remote)], timeout=30)
    subprocess.run(["rsync", "-rt", "--protect-args", "--", str(local / "input.md"),
                    f"{host}:{remote}/input.md"], check=True, timeout=120)
    cmd += ["--voice", voice, "--input", str(remote / "input.md"),
            "--output-dir", str(remote), "--format", audio_format]
    error = None
    try:
        with (local / "remote.log").open("w") as log:
            _ssh(host, cmd, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=43200)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        error = exc
    # Retrieve the reports even on failure; an exit code alone never approves audio.
    subprocess.run(["rsync", "-rt", "--protect-args", "--", f"{host}:{remote}/", str(local) + "/"],
                   check=True, timeout=1800)
    if error:
        raise RuntimeError(f"Remote narration failed; inspect {local / 'remote.log'}") from error
    record, paths = verify_download(local)
    record.update({"host": host, "remote_run": str(remote), "local_run": str(local),
                   "source_path": str(source), "voice": voice})
    (local / "tts_result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record, paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", nargs="?")
    nodes = ap.add_mutually_exclusive_group()
    nodes.add_argument("--host", default=SPARK_TTS_HOST)
    nodes.add_argument("--hosts", nargs=2, metavar=("FIRST", "SECOND"),
                       help="Narrate two ordered parts concurrently, then join their MP3s")
    ap.add_argument("--voice", default=SPARK_TTS_VOICE)
    ap.add_argument("--list-voices", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--output-dir", default="output/spark_narrations")
    args = ap.parse_args()
    if args.hosts and (args.list_voices or args.preflight):
        ap.error("Use --host for voice listing or preflight; --hosts is for narration")
    if args.list_voices or args.preflight:
        result = inspect(host=args.host, voice=args.voice, list_voices=args.list_voices)
    else:
        if not args.input:
            ap.error("input is required unless listing voices or checking readiness")
        if args.hosts:
            from parallel_tts import narrate_parallel
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            result = narrate_parallel(args.input, args.output_dir, hosts=args.hosts, voice=args.voice)
        else:
            result, paths = narrate(args.input, args.output_dir, host=args.host, voice=args.voice)
            result["audio_path"] = str(paths["audio"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
