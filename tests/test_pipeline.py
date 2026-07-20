import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import agent
import digest
import judge
from queue_manager import ContentItem, ContentQueue, ContentType


def item(number: int = 0) -> ContentItem:
    return ContentItem(
        url=f"https://example.test/{number}",
        title=f"Research item {number}",
        source="test",
        content_type=ContentType.PAPER,
        raw_text="A technically detailed abstract.",
        abstract="A technically detailed abstract.",
    )


def result(index: int, verdict: str = "keep") -> dict:
    return {
        "item_index": index,
        "technical_depth": 8,
        "novelty": 7,
        "relevance": 9,
        "verdict": verdict,
        "reason": "Substantive new method.",
        "key_contributions": "A sparse method; a controlled evaluation.",
        "summary": "The method changes the architecture. It reduces memory at matched accuracy.",
    }


class JudgeContractTests(unittest.IsolatedAsyncioTestCase):
    def test_batch_validation_reorders_by_index(self):
        validated = judge._validate_batch_result(
            {"results": [result(1), result(0)]}, expected_count=2
        )
        self.assertEqual([entry["item_index"] for entry in validated], [0, 1])

    def test_batch_validation_rejects_bool_score(self):
        invalid = result(0)
        invalid["technical_depth"] = True
        with self.assertRaisesRegex(ValueError, "technical_depth"):
            judge._validate_batch_result({"results": [invalid]}, expected_count=1)

    async def test_call_parses_structured_codex_output(self):
        mocked = AsyncMock(return_value=json.dumps({"results": [result(0)]}))
        with patch.object(judge, "run_codex", mocked):
            output = await judge._call_codex_judge([item()])
        self.assertEqual(output[0]["verdict"], "keep")
        _, payload = mocked.await_args.args
        self.assertEqual(json.loads(payload)["items"][0]["item_index"], 0)
        self.assertIsNotNone(mocked.await_args.kwargs["output_schema"])

    async def test_join_waits_for_slow_inflight_judge(self):
        queue = ContentQueue()
        stop = asyncio.Event()

        async def slow_call(items):
            await asyncio.sleep(0.05)
            return [result(index) for index, _ in enumerate(items)]

        with (
            patch.object(judge, "_call_codex_judge", side_effect=slow_call),
            patch.object(judge, "CODEX_JUDGE_BATCH_WAIT_SECONDS", 0.01),
        ):
            task = asyncio.create_task(judge.judge_loop(queue, stop))
            await queue.submit(item())
            await queue.join_pending()
            stop.set()
            await asyncio.wait_for(task, timeout=2)

        self.assertEqual(queue.stats["judged"], 1)
        self.assertEqual(queue.stats["kept"], 1)

    async def test_judge_retries_are_bounded(self):
        queue = ContentQueue()
        stop = asyncio.Event()
        with (
            patch.object(judge, "_call_codex_judge", AsyncMock(return_value=None)),
            patch.object(judge, "CODEX_MAX_JUDGE_ATTEMPTS", 2),
            patch.object(judge, "CODEX_RETRY_DELAY_SECONDS", 0),
            patch.object(judge, "CODEX_JUDGE_BATCH_WAIT_SECONDS", 0.01),
        ):
            task = asyncio.create_task(judge.judge_loop(queue, stop))
            await queue.submit(item())
            await queue.join_pending()
            stop.set()
            await asyncio.wait_for(task, timeout=2)

        self.assertEqual(queue.stats["retried"], 1)
        self.assertEqual(queue.stats["failed"], 1)
        self.assertEqual((await queue.drain_failed())[0].judge_attempts, 2)

    async def test_drain_surfaces_worker_crash_instead_of_hanging(self):
        queue = ContentQueue()
        stop = asyncio.Event()
        await queue.submit(item())

        async def crash():
            await asyncio.sleep(0)
            raise ValueError("worker bug")

        async def idle():
            await asyncio.sleep(30)

        tasks = [asyncio.create_task(crash()), asyncio.create_task(idle())]
        try:
            with self.assertRaisesRegex(RuntimeError, "worker crashed"):
                await asyncio.wait_for(
                    agent._drain_judges(queue, tasks, stop),
                    timeout=1,
                )
        finally:
            await agent._cancel_judges(tasks, stop)


class WriterTests(unittest.IsolatedAsyncioTestCase):
    async def test_writer_preserves_multiline_unicode(self):
        prose = "First paragraph.\n\nSecond paragraph about α and β."
        response = json.dumps({"covered_item_indices": [1], "segment": prose})
        mocked = AsyncMock(return_value=response)
        with patch.object(digest, "run_codex", mocked):
            output = await digest._call_codex_scriptwriter(
                "source material",
                expected_indices={1},
            )
        self.assertEqual(output, prose)
        self.assertEqual(mocked.await_args.kwargs["output_schema"], digest.WRITER_SCHEMA)

    async def test_writer_rejects_incomplete_coverage(self):
        response = json.dumps(
            {"covered_item_indices": [1], "segment": "Enough technical words for a segment."}
        )
        with (
            patch.object(digest, "run_codex", AsyncMock(return_value=response)),
            patch.object(digest, "CODEX_MAX_WRITER_ATTEMPTS", 1),
        ):
            output = await digest._call_codex_scriptwriter(
                "source material",
                expected_indices={1, 2},
            )
        self.assertIsNone(output)


class TTSTests(unittest.TestCase):
    def _write_fake_tts(
        self,
        root: Path,
        *,
        create_audio: bool,
        complete_qc: bool = True,
        qc_status: str = "CLEAN",
    ) -> Path:
        script = root / "fake_tts.py"
        body = [
            "import json",
            "from pathlib import Path",
            "out = Path('fake_voice_output')",
            "out.mkdir(exist_ok=True)",
        ]
        if create_audio:
            body.extend(
                [
                    "(out / 'digest_complete.mp3').write_bytes(b'x' * 2048)",
                    "report = {'metadata': {'chunks': 1}, 'summary': {",
                    "    'screened_text_chunks': 1," if complete_qc else "    'screened_text_chunks': 0,",
                    f"    'screened_audio_chunks': 1, 'status': {qc_status!r}}}}}",
                    "(out / 'hallucination_report.json').write_text(json.dumps(report))",
                ]
            )
        script.write_text("\n".join(body), encoding="utf-8")
        return script

    def test_tts_accepts_only_new_fully_screened_audio(self):
        import sys

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            input_path = root / "digest.txt"
            input_path.write_text("A short test narration.")
            script = self._write_fake_tts(root, create_audio=True)
            with (
                patch.object(digest, "AUDIO_DIR", audio_dir),
                patch.object(digest, "TTS_CONDA_PYTHON", sys.executable),
                patch.object(digest, "TTS_SCRIPT", str(script)),
                patch.object(digest, "TTS_SCRIPT_DIR", str(root)),
                patch.object(digest, "TTS_VOICE", "p246"),
                patch.object(digest, "TTS_FORMAT", "mp3"),
                patch.object(digest, "TTS_ENABLED", True),
                patch.object(digest, "_probe_audio_duration", return_value=12.3),
            ):
                output = digest.synthesize_audio(input_path)
            self.assertIsNotNone(output)
            self.assertTrue(output.approved)
            self.assertEqual(output.path.name, "digest_complete.mp3")
            manifest = json.loads(output.manifest_path.read_text())
            self.assertEqual(manifest["voice_resolved"], None)
            self.assertEqual(manifest["screened_audio_chunks"], 1)

    def test_tts_never_returns_stale_audio(self):
        import sys

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            (audio_dir / "old.mp3").write_bytes(b'x' * 2048)
            input_path = root / "digest.txt"
            input_path.write_text("A short test narration.")
            script = self._write_fake_tts(root, create_audio=False)
            with (
                patch.object(digest, "AUDIO_DIR", audio_dir),
                patch.object(digest, "TTS_CONDA_PYTHON", sys.executable),
                patch.object(digest, "TTS_SCRIPT", str(script)),
                patch.object(digest, "TTS_SCRIPT_DIR", str(root)),
                patch.object(digest, "TTS_VOICE", "p246"),
                patch.object(digest, "TTS_FORMAT", "mp3"),
                patch.object(digest, "TTS_ENABLED", True),
            ):
                self.assertIsNone(digest.synthesize_audio(input_path))

    def test_tts_rejects_incomplete_chunk_screening(self):
        import sys

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            input_path = root / "digest.txt"
            input_path.write_text("A short test narration.")
            script = self._write_fake_tts(root, create_audio=True, complete_qc=False)
            with (
                patch.object(digest, "AUDIO_DIR", audio_dir),
                patch.object(digest, "TTS_CONDA_PYTHON", sys.executable),
                patch.object(digest, "TTS_SCRIPT", str(script)),
                patch.object(digest, "TTS_SCRIPT_DIR", str(root)),
                patch.object(digest, "TTS_VOICE", "p246"),
                patch.object(digest, "TTS_FORMAT", "mp3"),
                patch.object(digest, "TTS_ENABLED", True),
                patch.object(digest, "_probe_audio_duration", return_value=12.3),
            ):
                self.assertIsNone(digest.synthesize_audio(input_path))

    def test_tts_retains_review_artifact_with_status(self):
        import sys

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            input_path = root / "digest.txt"
            input_path.write_text("A short test narration.")
            script = self._write_fake_tts(
                root,
                create_audio=True,
                qc_status="REVIEW_NEEDED",
            )
            with (
                patch.object(digest, "AUDIO_DIR", audio_dir),
                patch.object(digest, "TTS_CONDA_PYTHON", sys.executable),
                patch.object(digest, "TTS_SCRIPT", str(script)),
                patch.object(digest, "TTS_SCRIPT_DIR", str(root)),
                patch.object(digest, "TTS_VOICE", "p246"),
                patch.object(digest, "TTS_FORMAT", "mp3"),
                patch.object(digest, "TTS_ENABLED", True),
                patch.object(digest, "_probe_audio_duration", return_value=12.3),
            ):
                output = digest.synthesize_audio(input_path)
            self.assertIsNotNone(output)
            self.assertFalse(output.approved)
            self.assertEqual(output.qc_status, "review_needed")
            self.assertTrue(output.manifest_path.is_file())


if __name__ == "__main__":
    unittest.main()
