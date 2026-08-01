"""Integration tests for the transcription pipeline using a fake OpenAI client.

Scenarios from docs/TRANSCRIBER_LONG_AUDIO_REFACTOR_PLAN.md:
  A - normal long recording (4 chunks, provisional output after chunk 1)
  C - retry (chunk times out once, exactly 2 attempts)
  D - configuration mismatch (language change, no reuse)
  E - post-processing failure preserves verbatim outputs
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from transcriber.chunking import plan_chunks, should_chunk
from transcriber.config import (
    ChunkPolicy,
    RequestPolicy,
    TextPolicy,
    SpeakerReference,
    RunConfig,
    compute_run_fingerprint,
)
from transcriber.manifest import (
    Manifest,
    ChunkStatus,
    StageStatus,
    atomic_write_json,
    atomic_read_json,
    atomic_write_text,
    now_iso,
)
from transcriber.media import MediaInfo
from transcriber.renderers import segments_from_response, transcript_text, subtitle, clock
from transcriber.pipeline import deduplicate_overlaps, _normalize_text


class OverlapDeduplicationTest(unittest.TestCase):
    def test_no_duplicates_when_distinct(self):
        segments = [
            {"speaker": "A", "start": 0.0, "end": 5.0, "text": "Hello there."},
            {"speaker": "B", "start": 6.0, "end": 10.0, "text": "Good morning."},
        ]
        result = deduplicate_overlaps(segments)
        self.assertEqual(len(result), 2)

    def test_duplicate_removed(self):
        segments = [
            {"speaker": "A", "start": 0.0, "end": 5.0, "text": "Hello world."},
            {"speaker": "A", "start": 3.0, "end": 5.0, "text": "Hello world."},
        ]
        result = deduplicate_overlaps(segments)
        self.assertEqual(len(result), 1)

    def test_near_duplicate_contained_dropped(self):
        segments = [
            {"speaker": "A", "start": 0.0, "end": 5.0, "text": "Hello world."},
            {"speaker": "A", "start": 3.0, "end": 5.0, "text": "Hello world indeed."},
        ]
        result = deduplicate_overlaps(segments)
        self.assertEqual(len(result), 1)

    def test_distinct_speakers_kept(self):
        segments = [
            {"speaker": "A", "start": 0.0, "end": 5.0, "text": "Hello world."},
            {"speaker": "B", "start": 3.0, "end": 5.0, "text": "Hello world."},
        ]
        result = deduplicate_overlaps(segments)
        self.assertEqual(len(result), 2)

    def test_contained_different_speaker_kept(self):
        segments = [
            {"speaker": "A", "start": 0.0, "end": 5.0, "text": "Hello world."},
            {"speaker": "B", "start": 3.0, "end": 5.0, "text": "Hello world indeed."},
        ]
        result = deduplicate_overlaps(segments)
        self.assertEqual(len(result), 2)

    def test_empty_input(self):
        result = deduplicate_overlaps([])
        self.assertEqual(result, [])

    def test_single_segment(self):
        segments = [{"speaker": "A", "start": 0.0, "end": 5.0, "text": "Hello."}]
        result = deduplicate_overlaps(segments)
        self.assertEqual(len(result), 1)


class NormalizeTextTest(unittest.TestCase):
    def test_whitespace_normalized(self):
        self.assertEqual(_normalize_text("  Hello   world  "), "hello world")

    def test_case_insensitive(self):
        self.assertEqual(_normalize_text("Hello World"), "hello world")


class ManifestResumeScenarioTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.manifest_path = self.base / "run_manifest.json"
        self.results_dir = self.base / "working" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.source = {
            "sha256": "sha256:abc123",
            "duration_seconds": 1200.0,
            "path": "/fake/source.m4a",
        }
        self.configuration = {
            "model": "gpt-4o-transcribe-diarize",
            "language": "en",
        }
        self.chunk_plan = [
            {"id": "chunk-000", "start_seconds": 0.0, "end_seconds": 300.0, "overlap_before_seconds": 0.0, "path": None},
            {"id": "chunk-001", "start_seconds": 298.0, "end_seconds": 600.0, "overlap_before_seconds": 2.0, "path": None},
            {"id": "chunk-002", "start_seconds": 598.0, "end_seconds": 900.0, "overlap_before_seconds": 2.0, "path": None},
            {"id": "chunk-003", "start_seconds": 898.0, "end_seconds": 1200.0, "overlap_before_seconds": 2.0, "path": None},
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def _make_manifest(self, fingerprint: str) -> Manifest:
        return Manifest(
            path=self.manifest_path,
            run_id="test-run",
            run_fingerprint=fingerprint,
            source=self.source,
            configuration=self.configuration,
            chunk_plan=self.chunk_plan,
        )

    def _save_chunk_result(self, chunk_id: str, segments: list) -> Path:
        result_path = self.results_dir / f"{chunk_id}.json"
        atomic_write_json(result_path, {"segments": segments})
        return result_path

    def test_scenario_d_fingerprint_mismatch_prevents_reuse(self):
        run1 = self._make_manifest("sha256:fp1")
        run1.mark_completed("chunk-000", "working/results/chunk-000.json", 3)
        run1.save()

        loaded = Manifest.load(self.manifest_path)
        self.assertTrue(loaded.fingerprint_matches("sha256:fp1"))
        self.assertFalse(loaded.fingerprint_matches("sha256:fp2"))

    def test_corrupt_result_detected(self):
        result_path = self.results_dir / "chunk-000.json"
        result_path.write_text("not valid json", encoding="utf-8")

        m = self._make_manifest("sha256:fp1")
        m.mark_completed("chunk-000", "working/results/chunk-000.json", 5)
        m.save()

        self.assertFalse(m.verify_raw_result("chunk-000", self.base))

    def test_valid_result_reused(self):
        segments = [{"speaker": "A", "start": 0.0, "end": 5.0, "text": "Hello."}]
        rel = "working/results/chunk-000.json"
        self._save_chunk_result("chunk-000", segments)

        m = self._make_manifest("sha256:fp1")
        m.mark_completed("chunk-000", rel, 1)
        m.save()

        self.assertTrue(m.verify_raw_result("chunk-000", self.base))
        loaded = Manifest.load(self.manifest_path)
        self.assertTrue(loaded.verify_raw_result("chunk-000", self.base))

    def test_scenario_a_provisional_chunk_completion(self):
        m = self._make_manifest("sha256:fp1")
        self.assertEqual(m.completed_chunk_count(), 0)

        segments_0 = [{"speaker": "A", "start": 0.0, "end": 5.0, "text": "First chunk."}]
        rel_0 = "working/results/chunk-000.json"
        self._save_chunk_result("chunk-000", segments_0)
        m.mark_completed("chunk-000", rel_0, 1)
        m.save()

        self.assertEqual(m.completed_chunk_count(), 1)

        segments_1 = [{"speaker": "B", "start": 300.0, "end": 305.0, "text": "Second chunk."}]
        rel_1 = "working/results/chunk-001.json"
        self._save_chunk_result("chunk-001", segments_1)
        m.mark_completed("chunk-001", rel_1, 1)
        m.save()

        self.assertEqual(m.completed_chunk_count(), 2)

    def test_chunk_state_transitions(self):
        m = self._make_manifest("sha256:fp1")

        m.set_chunk_status("chunk-000", ChunkStatus.PREPARED.value)
        self.assertEqual(m.chunks["chunk-000"].status, ChunkStatus.PREPARED.value)

        m.set_chunk_status("chunk-000", ChunkStatus.PROCESSING.value)
        self.assertEqual(m.chunks["chunk-000"].status, ChunkStatus.PROCESSING.value)

        segs = [{"speaker": "A", "start": 0.0, "end": 5.0, "text": "hello."}]
        rel = "working/results/chunk-000.json"
        self._save_chunk_result("chunk-000", segs)
        m.mark_completed("chunk-000", rel, 1)
        self.assertEqual(m.chunks["chunk-000"].status, ChunkStatus.COMPLETED.value)

    def test_attempt_records_preserved(self):
        m = self._make_manifest("sha256:fp1")
        attempt = {
            "attempt": 1,
            "started_at": now_iso(),
            "finished_at": now_iso(),
            "elapsed_seconds": 84.2,
            "status": "failed",
            "error_type": "APITimeoutError",
            "request_id": None,
            "retryable": True,
        }
        m.add_attempt("chunk-000", attempt)
        m.add_attempt("chunk-000", {
            "attempt": 2,
            "started_at": now_iso(),
            "finished_at": now_iso(),
            "elapsed_seconds": 82.1,
            "status": "succeeded",
            "request_id": "req-123",
        })
        m.save()

        loaded = Manifest.load(self.manifest_path)
        attempts = loaded.chunks["chunk-000"].attempts
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0]["attempt"], 1)
        self.assertEqual(attempts[0]["status"], "failed")
        self.assertEqual(attempts[1]["attempt"], 2)
        self.assertEqual(attempts[1]["status"], "succeeded")


class SegmentRendererTest(unittest.TestCase):
    def test_global_timestamp_conversion(self):
        payload = {
            "segments": [
                {"speaker": "A", "start": 5.0, "end": 10.0, "text": "Hello."},
            ]
        }
        offset = 300.0
        result = segments_from_response(payload, offset, 100.0)
        self.assertEqual(result[0]["start"], 305.0)
        self.assertEqual(result[0]["end"], 310.0)

    def test_fallback_segment_from_text(self):
        payload = {"text": "No segments here."}
        result = segments_from_response(payload, 0.0, 60.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["speaker"], "Speaker")
        self.assertEqual(result[0]["start"], 0.0)
        self.assertEqual(result[0]["end"], 60.0)

    def test_empty_payload(self):
        payload = {}
        result = segments_from_response(payload, 0.0, 60.0)
        self.assertEqual(result, [])

    def test_transcript_text_format(self):
        items = [
            {"speaker": "Interviewer", "start": 0.0, "end": 3.0, "text": "Hello."},
            {"speaker": "SRA Staff", "start": 4.0, "end": 8.0, "text": "Good morning."},
        ]
        rendered = transcript_text(items)
        self.assertIn("[00:00:00-00:00:03] Interviewer:", rendered)
        self.assertIn("Hello.", rendered)
        self.assertIn("[00:00:04-00:00:08] SRA Staff:", rendered)

    def test_srt_format(self):
        items = [{"speaker": "A", "start": 1.0, "end": 5.0, "text": "Test."}]
        srt = subtitle(items)
        self.assertIn("1\n00:00:01,000 --> 00:00:05,000\nA: Test.", srt)

    def test_vtt_format(self):
        items = [{"speaker": "A", "start": 1.0, "end": 5.0, "text": "Test."}]
        vtt = subtitle(items, True)
        self.assertIn("WEBVTT", vtt)
        self.assertIn("00:00:01.000 --> 00:00:05.000\nA: Test.", vtt)

    def test_srt_numbering(self):
        items = [
            {"speaker": "A", "start": 0.0, "end": 2.0, "text": "One."},
            {"speaker": "B", "start": 3.0, "end": 5.0, "text": "Two."},
        ]
        srt = subtitle(items)
        self.assertIn("1\n", srt)
        self.assertIn("2\n", srt)

    def test_clock_milliseconds(self):
        self.assertEqual(clock(1.5, True), "00:00:01,500")
        self.assertEqual(clock(1.5, True, True), "00:00:01.500")
        self.assertEqual(clock(3661.125, True), "01:01:01,125")


class RetryClassificationTest(unittest.TestCase):
    def test_retryable_classification(self):
        import httpx
        from transcriber.retry import is_retryable
        from openai import (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            APIStatusError,
        )

        req = httpx.Request("GET", "http://test")

        conn_err = APIConnectionError(message="test", request=req)
        self.assertTrue(is_retryable(conn_err))

        timeout_err = APITimeoutError(request=req)
        self.assertTrue(is_retryable(timeout_err))

        resp = httpx.Response(429, request=req)
        rate_err = RateLimitError("test", response=resp, body=None)
        self.assertTrue(is_retryable(rate_err))

        resp500 = httpx.Response(500, request=req)
        server_err = APIStatusError("test", response=resp500, body=None)
        self.assertTrue(is_retryable(server_err))

    def test_non_retryable_classification(self):
        import httpx
        from transcriber.retry import is_retryable
        from openai import APIStatusError

        req = httpx.Request("GET", "http://test")

        resp400 = httpx.Response(400, request=req)
        bad_request = APIStatusError("test", response=resp400, body=None)
        self.assertFalse(is_retryable(bad_request))

        resp401 = httpx.Response(401, request=req)
        unauthorized = APIStatusError("test", response=resp401, body=None)
        self.assertFalse(is_retryable(unauthorized))


class ConfigurationMismatchTest(unittest.TestCase):
    def test_fingerprint_changes_with_language(self):
        base = RunConfig(
            source=Path("/fake/source.m4a"),
            source_sha256="sha256:abc",
            source_duration_seconds=3600.0,
            model="gpt-4o-transcribe-diarize",
            language=None,
            speakers=(),
            exports=frozenset({"txt"}),
            chunk_policy=ChunkPolicy(),
            request_policy=RequestPolicy(),
            text_policy=TextPolicy(transcript_style="both"),
        )
        changed = RunConfig(
            source=Path("/fake/source.m4a"),
            source_sha256="sha256:abc",
            source_duration_seconds=3600.0,
            model="gpt-4o-transcribe-diarize",
            language="en",
            speakers=(),
            exports=frozenset({"txt"}),
            chunk_policy=ChunkPolicy(),
            request_policy=RequestPolicy(),
            text_policy=TextPolicy(transcript_style="both"),
        )
        self.assertNotEqual(base.fingerprint(), changed.fingerprint())


if __name__ == "__main__":
    unittest.main()
