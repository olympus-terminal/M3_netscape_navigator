import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import spark_tts


class DownloadVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = self.root / "sample.mp3"
        self.audio.write_bytes(b"fixture" * 300)
        self.report = {"summary": {"status": "CLEAN"}}
        self.screen = {"text_triggers": 0}
        self.result = {
            "audio": "sample.mp3", "screening_report": "report.json", "text_screen": "screen.json",
            "size_bytes": self.audio.stat().st_size,
            "audio_sha256": hashlib.sha256(self.audio.read_bytes()).hexdigest(),
            "status": "CLEAN", "duration_seconds": 12.5,
        }
        self.probe = patch.object(spark_tts.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "12.5\n"))
        self.probe.start()
        self.addCleanup(self.probe.stop)

    def verify(self):
        (self.root / "tts_result.json").write_text(json.dumps(self.result))
        (self.root / "report.json").write_text(json.dumps(self.report))
        (self.root / "screen.json").write_text(json.dumps(self.screen))
        return spark_tts.verify_download(self.root)

    def test_verified_and_review_outputs_keep_their_status(self):
        record, _ = self.verify()
        self.assertEqual(record["status"], "CLEAN")
        self.result["status"] = self.report["summary"]["status"] = "REVIEW_NEEDED"
        record, _ = self.verify()
        self.assertEqual(record["status"], "REVIEW_NEEDED")

    def test_damaged_transfer_is_rejected(self):
        self.audio.write_bytes(b"damaged" * 300)
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            self.verify()

    def test_report_cannot_approve_flagged_audio_or_text(self):
        self.report["summary"]["status"] = "REVIEW_NEEDED"
        with self.assertRaisesRegex(RuntimeError, "contradicts"):
            self.verify()
        self.report["summary"]["status"] = "CLEAN"
        self.screen["text_triggers"] = 1
        with self.assertRaisesRegex(RuntimeError, "contradicts"):
            self.verify()

    def test_path_outside_download_is_rejected(self):
        self.result["audio"] = "../outside.mp3"
        with self.assertRaisesRegex(RuntimeError, "invalid downloaded"):
            self.verify()

    def test_invalid_duration_is_rejected(self):
        with patch.object(spark_tts.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "1\n")):
            with self.assertRaisesRegex(RuntimeError, "duration"):
                self.verify()

    def test_host_cannot_inject_ssh_options(self):
        with self.assertRaises(ValueError):
            spark_tts._ssh("-oProxyCommand=bad", ["true"])

    def test_input_is_checked_before_remote_work(self):
        with self.assertRaises(ValueError):
            spark_tts.narrate(self.audio, self.root)
