import asyncio
import errno
import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import agent
import crawlers
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
    def test_writer_requires_source_backed_research_attribution(self):
        instruction = digest.SCRIPT_WRITER_INSTRUCTION
        self.assertIn("first substantive mention", instruction)
        self.assertIn("and colleagues", instruction)
        self.assertIn("source-provided affiliation", instruction)
        self.assertIn("Never infer", instruction)

    def test_writer_payload_includes_source_backed_author_affiliations(self):
        research = item()
        research.authors = "Alex Rivera, Sam Lee"
        research.metadata["author_affiliations"] = [
            {
                "author": "Alex Rivera",
                "affiliation": "Example Research Institute",
            }
        ]

        payload = json.loads(digest._prepare_batch_payload([research]))

        self.assertEqual(payload["items"][0]["authors"], research.authors)
        self.assertEqual(
            payload["items"][0]["author_affiliations"],
            research.metadata["author_affiliations"],
        )

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


class AttributionMetadataTests(unittest.TestCase):
    def test_arxiv_author_metadata_preserves_source_affiliation_pairing(self):
        entry = ET.fromstring(
            """
            <entry xmlns="http://www.w3.org/2005/Atom"
                   xmlns:arxiv="http://arxiv.org/schemas/atom">
              <author>
                <name>Alex Rivera</name>
                <arxiv:affiliation>Example Research Institute</arxiv:affiliation>
              </author>
              <author><name>Sam Lee</name></author>
            </entry>
            """
        )

        authors, author_affiliations = crawlers._arxiv_author_metadata(entry)

        self.assertEqual(authors, "Alex Rivera, Sam Lee")
        self.assertEqual(
            author_affiliations,
            [
                {
                    "author": "Alex Rivera",
                    "affiliation": "Example Research Institute",
                }
            ],
        )

    def test_semantic_scholar_author_names_ignore_malformed_entries(self):
        authors = crawlers._semantic_scholar_author_names(
            [
                {"authorId": "1", "name": "Alex Rivera"},
                None,
                {"authorId": "2"},
                {"authorId": "3", "name": "  Sam Lee  "},
            ]
        )

        self.assertEqual(authors, "Alex Rivera, Sam Lee")


class SourceNotesTests(unittest.TestCase):
    def test_source_notes_render_source_backed_items_safely_in_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            notes_path = root / "digest.md"

            first = item(1)
            first.title = "Method [alpha]\n# not a heading"
            first.url = "https://example.test/paper?q=source notes"
            first.source = "arxiv"
            first.authors = "Alex_Rivera, Sam Lee"
            first.published = "2026-07-27"
            first.technical_depth = 8
            first.novelty = 7
            first.relevance = 9
            first.summary = "Uses <source evidence> while preserving α and β."
            first.metadata["author_affiliations"] = [
                {
                    "author": "Alex_Rivera",
                    "affiliation": "Example [Research] Institute",
                },
                {"author": "Incomplete entry"},
                "not a mapping",
            ]

            second = item(2)
            second.title = "Second study"
            second.url = "javascript:alert('not a source')"
            second.authors = ""
            second.technical_depth = 7
            second.novelty = 6
            second.relevance = 8
            second.summary = "A second assistant-generated summary."

            self.assertEqual(
                digest.write_digest_source_notes([first, second], notes_path),
                notes_path,
            )
            notes = notes_path.read_text(encoding="utf-8")
            self.assertLess(notes.index("Method"), notes.index("Second study"))
            self.assertNotIn("\n# not a heading", notes)
            self.assertIn("Alex\\_Rivera", notes)
            self.assertIn("Example \\[Research\\] Institute", notes)
            self.assertNotIn("Incomplete entry", notes)
            self.assertIn("https://example.test/paper?q=source%20notes", notes)
            self.assertNotIn("javascript:", notes)
            self.assertEqual(notes.count("Original source:"), 1)
            self.assertIn("Assistant summary:", notes)
            self.assertIn("α and β", notes)
            self.assertNotIn("Unknown", notes)
            self.assertNotIn("N/A", notes)
            self.assertIsNone(digest._safe_http_url("http://[broken"))

    def test_digest_metadata_records_published_source_notes_and_authors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            digest_dir = root / "digests"
            script_path = root / "digest.txt"
            script_path.write_text("Narration.", encoding="utf-8")
            internal_notes_path = root / "digest.md"
            internal_notes_path.write_text("# Sources\n", encoding="utf-8")
            public_notes_path = (
                root
                / "ai_research_digest_2026-07-27_14-23-56+0400_02_REVIEW_NEEDED.md"
            )
            public_notes_path.write_text("# Sources\n", encoding="utf-8")
            audio_result = SimpleNamespace(
                path=public_notes_path.with_suffix(".mp3"),
                notes_path=public_notes_path,
                manifest_path=root / "tts_manifest.json",
                screening_report_path=root / "hallucination_report.json",
                qc_status="review_needed",
                approved=False,
            )

            research = item(1)
            research.authors = "Alex Rivera, Sam Lee"
            research.published = "2026-07-27"
            research.metadata["author_affiliations"] = [
                {
                    "author": "Alex Rivera",
                    "affiliation": "Example Research Institute",
                },
                {"author": "Incomplete entry"},
            ]

            with patch.object(digest, "DIGEST_DIR", digest_dir):
                digest.save_digest_metadata(
                    [research],
                    script_path,
                    internal_notes_path,
                    audio_result,
                )

            [meta_path] = list(digest_dir.glob("digest_*.json"))
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["source_notes_path"],
                str(public_notes_path.resolve()),
            )
            self.assertEqual(
                metadata["source_notes_sha256"],
                digest._sha256_file(public_notes_path),
            )
            self.assertEqual(metadata["items"][0]["authors"], research.authors)
            self.assertEqual(metadata["items"][0]["published"], research.published)
            self.assertEqual(
                metadata["items"][0]["author_affiliations"],
                [
                    {
                        "author": "Alex Rivera",
                        "affiliation": "Example Research Institute",
                    }
                ],
            )

    def test_atomic_text_failure_leaves_no_partial_notes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            destination = root / "sources.md"

            with patch.object(
                digest.os,
                "replace",
                side_effect=OSError(errno.ENOSPC, "disk full"),
            ):
                with self.assertRaises(OSError):
                    digest._write_text_atomic(destination, "complete notes")

            self.assertFalse(destination.exists())
            self.assertEqual(list(root.iterdir()), [])


class TTSTests(unittest.TestCase):
    def _write_notes(self, input_path: Path) -> Path:
        notes_path = input_path.with_suffix(".md")
        notes_path.write_text("# AI Research Digest Sources\n", encoding="utf-8")
        return notes_path

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
            notes_path = self._write_notes(input_path)
            script = self._write_fake_tts(root, create_audio=True)
            with (
                patch.object(digest, "AUDIO_DIR", audio_dir),
                patch.object(digest, "DELIVERY_DIR", root),
                patch.object(digest, "TTS_CONDA_PYTHON", sys.executable),
                patch.object(digest, "TTS_SCRIPT", str(script)),
                patch.object(digest, "TTS_SCRIPT_DIR", str(root)),
                patch.object(digest, "TTS_VOICE", "p246"),
                patch.object(digest, "TTS_FORMAT", "mp3"),
                patch.object(digest, "TTS_ENABLED", True),
                patch.object(digest, "_probe_audio_duration", return_value=12.3),
            ):
                output = digest.synthesize_audio(input_path, notes_path)
            self.assertIsNotNone(output)
            self.assertTrue(output.approved)
            self.assertEqual(output.path.parent, root)
            self.assertRegex(
                output.path.name,
                r"^ai_research_digest_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}[+-]\d{4}\.mp3$",
            )
            self.assertEqual(output.path.stat().st_size, 2048)
            manifest = json.loads(output.manifest_path.read_text())
            self.assertEqual(manifest["voice_resolved"], None)
            self.assertEqual(manifest["screened_audio_chunks"], 1)
            self.assertEqual(manifest["audio_path"], str(output.path.resolve()))
            self.assertEqual(manifest["source_notes_path"], str(output.notes_path.resolve()))
            self.assertEqual(
                manifest["source_notes_sha256"],
                digest._sha256_file(output.notes_path),
            )
            self.assertTrue(Path(manifest["tts_output_dir"]).is_dir())
            self.assertFalse(
                (Path(manifest["tts_output_dir"]) / manifest["tts_output_filename"]).exists()
            )
            self.assertTrue(output.manifest_path.is_file())
            self.assertTrue(output.screening_report_path.is_file())
            self.assertTrue(output.path.is_file())
            self.assertTrue(output.notes_path.is_file())
            self.assertEqual(output.path.stem, output.notes_path.stem)
            self.assertEqual(
                output.notes_path.read_text(encoding="utf-8"),
                notes_path.read_text(encoding="utf-8"),
            )
            self.assertFalse(output.path.is_symlink())

    def test_tts_never_returns_stale_audio(self):
        import sys

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            (audio_dir / "old.mp3").write_bytes(b'x' * 2048)
            input_path = root / "digest.txt"
            input_path.write_text("A short test narration.")
            notes_path = self._write_notes(input_path)
            script = self._write_fake_tts(root, create_audio=False)
            with (
                patch.object(digest, "AUDIO_DIR", audio_dir),
                patch.object(digest, "DELIVERY_DIR", root),
                patch.object(digest, "TTS_CONDA_PYTHON", sys.executable),
                patch.object(digest, "TTS_SCRIPT", str(script)),
                patch.object(digest, "TTS_SCRIPT_DIR", str(root)),
                patch.object(digest, "TTS_VOICE", "p246"),
                patch.object(digest, "TTS_FORMAT", "mp3"),
                patch.object(digest, "TTS_ENABLED", True),
            ):
                self.assertIsNone(digest.synthesize_audio(input_path, notes_path))
            self.assertEqual(list(root.glob("ai_research_digest_*.*")), [])

    def test_tts_rejects_incomplete_chunk_screening(self):
        import sys

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            input_path = root / "digest.txt"
            input_path.write_text("A short test narration.")
            notes_path = self._write_notes(input_path)
            script = self._write_fake_tts(root, create_audio=True, complete_qc=False)
            with (
                patch.object(digest, "AUDIO_DIR", audio_dir),
                patch.object(digest, "DELIVERY_DIR", root),
                patch.object(digest, "TTS_CONDA_PYTHON", sys.executable),
                patch.object(digest, "TTS_SCRIPT", str(script)),
                patch.object(digest, "TTS_SCRIPT_DIR", str(root)),
                patch.object(digest, "TTS_VOICE", "p246"),
                patch.object(digest, "TTS_FORMAT", "mp3"),
                patch.object(digest, "TTS_ENABLED", True),
                patch.object(digest, "_probe_audio_duration", return_value=12.3),
            ):
                self.assertIsNone(digest.synthesize_audio(input_path, notes_path))
            self.assertEqual(list(root.glob("ai_research_digest_*.*")), [])

    def test_tts_retains_review_artifact_with_status(self):
        import sys

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            input_path = root / "digest.txt"
            input_path.write_text("A short test narration.")
            notes_path = self._write_notes(input_path)
            script = self._write_fake_tts(
                root,
                create_audio=True,
                qc_status="REVIEW_NEEDED",
            )
            with (
                patch.object(digest, "AUDIO_DIR", audio_dir),
                patch.object(digest, "DELIVERY_DIR", root),
                patch.object(digest, "TTS_CONDA_PYTHON", sys.executable),
                patch.object(digest, "TTS_SCRIPT", str(script)),
                patch.object(digest, "TTS_SCRIPT_DIR", str(root)),
                patch.object(digest, "TTS_VOICE", "p246"),
                patch.object(digest, "TTS_FORMAT", "mp3"),
                patch.object(digest, "TTS_ENABLED", True),
                patch.object(digest, "_probe_audio_duration", return_value=12.3),
            ):
                output = digest.synthesize_audio(input_path, notes_path)
            self.assertIsNotNone(output)
            self.assertFalse(output.approved)
            self.assertEqual(output.qc_status, "review_needed")
            self.assertTrue(output.path.name.endswith("_REVIEW_NEEDED.mp3"))
            self.assertTrue(output.notes_path.name.endswith("_REVIEW_NEEDED.md"))
            self.assertEqual(output.path.stem, output.notes_path.stem)
            self.assertTrue(output.manifest_path.is_file())

    def test_manifest_failure_rolls_back_delivery_and_retains_staging_audio(self):
        import sys

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            input_path = root / "digest.txt"
            input_path.write_text("A short test narration.")
            notes_path = self._write_notes(input_path)
            script = self._write_fake_tts(root, create_audio=True)
            with (
                patch.object(digest, "AUDIO_DIR", audio_dir),
                patch.object(digest, "DELIVERY_DIR", root),
                patch.object(digest, "TTS_CONDA_PYTHON", sys.executable),
                patch.object(digest, "TTS_SCRIPT", str(script)),
                patch.object(digest, "TTS_SCRIPT_DIR", str(root)),
                patch.object(digest, "TTS_VOICE", "p246"),
                patch.object(digest, "TTS_FORMAT", "mp3"),
                patch.object(digest, "TTS_ENABLED", True),
                patch.object(digest, "_probe_audio_duration", return_value=12.3),
                patch.object(
                    digest,
                    "_write_json_atomic",
                    side_effect=OSError(errno.ENOSPC, "disk full"),
                ),
            ):
                output = digest.synthesize_audio(input_path, notes_path)

            self.assertIsNone(output)
            self.assertEqual(list(root.glob("ai_research_digest_*.mp3")), [])
            self.assertEqual(list(root.glob("ai_research_digest_*.md")), [])
            self.assertEqual(len(list(audio_dir.rglob("digest_complete.mp3"))), 1)

    def test_delivery_collision_gets_a_numbered_name_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first_source = root / "first.mp3"
            second_source = root / "second.mp3"
            first_notes = root / "first.md"
            second_notes = root / "second.md"
            first_source.write_bytes(b"first")
            second_source.write_bytes(b"second")
            first_notes.write_text("first sources", encoding="utf-8")
            second_notes.write_text("second sources", encoding="utf-8")
            fixed_time = datetime(
                2026,
                7,
                23,
                11,
                30,
                tzinfo=timezone(timedelta(hours=4)),
            )
            with patch.object(digest, "DELIVERY_DIR", root / "delivery"):
                first, first_public_notes, _, _ = digest._publish_delivery_bundle(
                    first_source,
                    first_notes,
                    "structural_clean",
                    fixed_time,
                )
                second, second_public_notes, _, _ = digest._publish_delivery_bundle(
                    second_source,
                    second_notes,
                    "structural_clean",
                    fixed_time,
                )

            self.assertEqual(
                first.name,
                "ai_research_digest_2026-07-23_11-30-00+0400.mp3",
            )
            self.assertEqual(
                second.name,
                "ai_research_digest_2026-07-23_11-30-00+0400_02.mp3",
            )
            self.assertEqual(first.stem, first_public_notes.stem)
            self.assertEqual(second.stem, second_public_notes.stem)
            self.assertEqual(first.read_bytes(), b"first")
            self.assertEqual(second.read_bytes(), b"second")
            self.assertEqual(first_public_notes.read_text(), "first sources")
            self.assertEqual(second_public_notes.read_text(), "second sources")
            self.assertFalse(first.is_symlink())

    def test_orphan_notes_force_the_bundle_to_use_the_next_collision_stem(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp3"
            notes = root / "source.md"
            source.write_bytes(b"audio")
            notes.write_text("sources", encoding="utf-8")
            delivery_dir = root / "delivery"
            delivery_dir.mkdir()
            fixed_time = datetime(
                2026,
                7,
                23,
                11,
                30,
                tzinfo=timezone(timedelta(hours=4)),
            )
            orphan = delivery_dir / "ai_research_digest_2026-07-23_11-30-00+0400.md"
            orphan.write_text("older notes", encoding="utf-8")

            with patch.object(digest, "DELIVERY_DIR", delivery_dir):
                audio_path, notes_path, _, _ = digest._publish_delivery_bundle(
                    source,
                    notes,
                    "structural_clean",
                    fixed_time,
                )

            self.assertTrue(audio_path.name.endswith("_02.mp3"))
            self.assertTrue(notes_path.name.endswith("_02.md"))
            self.assertEqual(orphan.read_text(), "older notes")

    def test_delivery_copy_fallback_is_verified_and_atomic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp3"
            notes = root / "source.md"
            source.write_bytes(b"verified audio")
            notes.write_text("verified sources", encoding="utf-8")
            real_link = os.link
            first_link = True

            def cross_device_once(source_path, destination_path):
                nonlocal first_link
                if first_link:
                    first_link = False
                    raise OSError(errno.EXDEV, "cross-device link")
                return real_link(source_path, destination_path)

            fixed_time = datetime(
                2026,
                7,
                23,
                11,
                30,
                tzinfo=timezone(timedelta(hours=4)),
            )
            delivery_dir = root / "delivery"
            with (
                patch.object(digest, "DELIVERY_DIR", delivery_dir),
                patch.object(digest.os, "link", side_effect=cross_device_once),
            ):
                destination, public_notes, method, _ = digest._publish_delivery_bundle(
                    source,
                    notes,
                    "structural_clean",
                    fixed_time,
                )

            self.assertEqual(method, "verified_copy")
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertEqual(public_notes.read_text(), notes.read_text())
            self.assertEqual(destination.stem, public_notes.stem)
            self.assertEqual(
                list(delivery_dir.glob(".ai_research_digest_publish_*.tmp")),
                [],
            )
            self.assertEqual(
                list(delivery_dir.glob(".ai_research_digest_sources_*.tmp")),
                [],
            )

    def test_delivery_copy_failure_leaves_no_partial_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp3"
            notes = root / "source.md"
            source.write_bytes(b"complete source")
            notes.write_text("complete sources", encoding="utf-8")
            delivery_dir = root / "delivery"

            def interrupted_copy(_source, temporary):
                Path(temporary).write_bytes(b"partial")
                raise OSError(errno.ENOSPC, "disk full")

            with (
                patch.object(digest, "DELIVERY_DIR", delivery_dir),
                patch.object(
                    digest.os,
                    "link",
                    side_effect=OSError(errno.EXDEV, "cross-device link"),
                ),
                patch.object(digest.shutil, "copy2", side_effect=interrupted_copy),
            ):
                with self.assertRaises(OSError):
                    digest._publish_delivery_bundle(
                        source,
                        notes,
                        "structural_clean",
                    )

            self.assertEqual(list(delivery_dir.iterdir()), [])

    def test_audio_commit_failure_rolls_back_public_notes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp3"
            notes = root / "source.md"
            source.write_bytes(b"complete source")
            notes.write_text("complete sources", encoding="utf-8")
            delivery_dir = root / "delivery"
            real_link = os.link
            link_calls = 0

            def fail_audio_commit(source_path, destination_path):
                nonlocal link_calls
                link_calls += 1
                if link_calls == 3:
                    raise OSError(errno.ENOSPC, "disk full")
                return real_link(source_path, destination_path)

            with (
                patch.object(digest, "DELIVERY_DIR", delivery_dir),
                patch.object(digest.os, "link", side_effect=fail_audio_commit),
            ):
                with self.assertRaises(OSError):
                    digest._publish_delivery_bundle(
                        source,
                        notes,
                        "structural_clean",
                    )

            self.assertEqual(list(delivery_dir.iterdir()), [])
            self.assertEqual(source.read_bytes(), b"complete source")
            self.assertEqual(notes.read_text(), "complete sources")


class TTSOutcomeStateTests(unittest.TestCase):
    def _result(self, root: Path, *, approved: bool):
        return SimpleNamespace(
            approved=approved,
            qc_status="structural_clean" if approved else "review_needed",
            path=root / "digest.mp3",
            notes_path=root / "digest.md",
            manifest_path=root / "tts_manifest.json",
            screening_report_path=root / "hallucination_report.json",
        )

    def test_review_artifact_waits_for_human_instead_of_hourly_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            state = {"pending_digest": True}
            cycle_success = agent._record_tts_outcome(
                state,
                self._result(Path(temp), approved=False),
            )

        self.assertTrue(cycle_success)
        self.assertFalse(state["pending_digest"])
        self.assertEqual(state["pending_review"]["qc_status"], "review_needed")
        self.assertTrue(state["pending_review"]["sources"].endswith("digest.md"))

    def test_missing_audio_is_retried(self):
        state = {"pending_digest": False}
        self.assertFalse(agent._record_tts_outcome(state, None))
        self.assertTrue(state["pending_digest"])

    def test_approved_audio_clears_an_older_review(self):
        with tempfile.TemporaryDirectory() as temp:
            state = {"pending_digest": True, "pending_review": {"old": True}}
            cycle_success = agent._record_tts_outcome(
                state,
                self._result(Path(temp), approved=True),
            )

        self.assertTrue(cycle_success)
        self.assertFalse(state["pending_digest"])
        self.assertNotIn("pending_review", state)


if __name__ == "__main__":
    unittest.main()
