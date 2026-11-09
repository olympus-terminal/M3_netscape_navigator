import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import parallel_tts as parallel


class SplitTests(unittest.TestCase):
    def test_paragraph_split_preserves_all_text_and_numbers(self):
        first = "First section has 20 measurements and a 3.14 ratio.\n\n"
        second = "Second section keeps its Unicode α, quotes, and source names.\n"
        self.assertEqual(parallel.split_narration(first + second), [first, second])

    def test_single_paragraph_uses_sentence_boundary(self):
        text = "The value was 3.14 in each sample. Another sentence follows with results. Final remarks."
        parts = parallel.split_narration(text)
        self.assertEqual("".join(parts), text)
        self.assertEqual(len(parts), 2)
        self.assertTrue(parts[0].rstrip().endswith("."))
        self.assertIn("3.14", parts[0])

    def test_no_safe_boundary_is_rejected(self):
        for text in ["", "one", "words without a sentence or paragraph break"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parallel.split_narration(text)


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.md"
        self.source.write_text("First section has enough words for one part.\n\nSecond section is here with comparable length.")
        self.profile = {key: f"test-{key}" for key in parallel.IDENTITY_KEYS}
        self.barrier = threading.Barrier(2)
        self.second_finished = threading.Event()
        self.fail_host = None
        self.review_host = None
        self.alter_text_host = None
        self.joined = []

        def narrate(source, output, *, host, **kwargs):
            self.barrier.wait(timeout=3)  # fails if scheduling becomes sequential
            if host == "node-one":
                self.second_finished.wait(timeout=3)
            else:
                self.second_finished.set()
            if host == self.fail_host:
                raise RuntimeError("Deliberate worker failure")
            record = {**self.profile, "host": host,
                      "status": "REVIEW_NEEDED" if host == self.review_host else "CLEAN"}
            (Path(output) / "part_spoken.txt").write_text(
                "Changed narration" if host == self.alter_text_host else Path(source).read_text()
            )
            return record, {"audio": Path(output) / "part.mp3"}

        def join(parts, run, status):
            self.joined = [part[0]["host"] for part in parts]
            temporary = run / "temporary.mp3"
            temporary.write_bytes(b"joined test artifact")
            suffix = "_REVIEW_NEEDED" if status == "REVIEW_NEEDED" else ""
            return temporary, run / f"narration{suffix}.mp3", 20.0, [9.0, 11.0]

        patches = [
            patch.object(parallel, "inspect", return_value=self.profile),
            patch.object(parallel, "prepare_text", return_value=self.source.read_text()),
            patch.object(parallel, "narrate", side_effect=narrate),
            patch.object(parallel, "join_audio", side_effect=join),
            patch.object(parallel.shutil, "which", return_value="/tool"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def run_narration(self, hosts=("node-one", "node-two")):
        return parallel.narrate_parallel(self.source, self.root, hosts=hosts,
                                         root="/bundle", python="/python", voice="test-voice")

    def test_concurrent_completion_preserves_source_order_and_offsets(self):
        result = self.run_narration()
        self.assertEqual(self.joined, ["node-one", "node-two"])
        self.assertEqual([p["start_seconds"] for p in result["parts"]], [0, 9])
        self.assertEqual(result["status"], "CLEAN")
        self.assertTrue(Path(result["audio_path"]).is_file())

    def test_review_flags_reach_combined_status_and_filename(self):
        self.review_host = "node-two"
        result = self.run_narration()
        self.assertEqual(result["status"], "REVIEW_NEEDED")
        self.assertIn("REVIEW_NEEDED", Path(result["audio_path"]).name)

    def test_failed_part_keeps_diagnostics_without_combined_audio(self):
        self.fail_host = "node-one"
        with self.assertRaisesRegex(RuntimeError, "diagnostics"):
            self.run_narration()
        manifest_path = next(self.root.glob("parallel_*/parallel_manifest.json"))
        result = json.loads(manifest_path.read_text())
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual([p["status"] for p in result["parts"]], ["FAILED", "CLEAN"])
        self.assertFalse(self.joined)
        self.assertFalse(list(manifest_path.parent.glob("*.mp3")))

    def test_profile_mismatch_is_rejected_before_synthesis(self):
        with patch.object(parallel, "inspect", side_effect=[self.profile, {**self.profile, "reference_sha256": "other"}]):
            with self.assertRaises(RuntimeError):
                self.run_narration()
        self.assertFalse(self.joined)
        manifest = json.loads(next(self.root.glob("parallel_*/parallel_manifest.json")).read_text())
        self.assertIn("matching", manifest["error"])

    def test_same_node_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "different"):
            self.run_narration(("same-node", "same-node"))

    def test_changed_spoken_text_prevents_combined_publication(self):
        self.alter_text_host = "node-two"
        with self.assertRaisesRegex(RuntimeError, "diagnostics"):
            self.run_narration()
        self.assertFalse(self.joined)
        manifest = json.loads(next(self.root.glob("parallel_*/parallel_manifest.json")).read_text())
        self.assertIn("Spoken text differs", manifest["parts"][1]["error"])
