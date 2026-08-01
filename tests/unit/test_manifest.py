import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from transcriber.config import (
    ChunkPolicy,
    RequestPolicy,
    TextPolicy,
    RunConfig,
    compute_run_fingerprint,
    SCHEMA_VERSION,
)
from transcriber.manifest import (
    ChunkStatus,
    StageStatus,
    ChunkRecord,
    StageRecord,
    Manifest,
    atomic_write_json,
    atomic_read_json,
    now_iso,
)


class ChunkStatusTest(unittest.TestCase):
    def test_values(self):
        self.assertEqual(ChunkStatus.PLANNED.value, "planned")
        self.assertEqual(ChunkStatus.COMPLETED.value, "completed")
        self.assertEqual(ChunkStatus.FAILED_RETRYABLE.value, "failed_retryable")
        self.assertEqual(ChunkStatus.FAILED_TERMINAL.value, "failed_terminal")


class StageStatusTest(unittest.TestCase):
    def test_values(self):
        self.assertEqual(StageStatus.PENDING.value, "pending")
        self.assertEqual(StageStatus.COMPLETED.value, "completed")
        self.assertEqual(StageStatus.NOT_REQUESTED.value, "not_requested")


class ChunkRecordTest(unittest.TestCase):
    def test_default(self):
        c = ChunkRecord(id="chunk-000")
        self.assertEqual(c.id, "chunk-000")
        self.assertEqual(c.status, ChunkStatus.PREPARED.value)
        self.assertEqual(c.attempts, [])
        self.assertIsNone(c.raw_result)
        self.assertIsNone(c.segment_count)


class ManifestAtomicIOTest(unittest.TestCase):
    def test_atomic_write_and_read(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "test.json"
            data = {"key": "value", "nested": {"a": 1}}
            atomic_write_json(p, data)
            self.assertTrue(p.exists())
            read = atomic_read_json(p)
            self.assertEqual(read, data)

    def test_atomic_write_no_partial_files(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            data = {"key": "value"}
            atomic_write_json(p, data)
            temps = list(Path(d).glob("*.tmp"))
            self.assertEqual(len(temps), 0)

    def test_atomic_overwrites(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            atomic_write_json(p, {"v": 1})
            self.assertEqual(atomic_read_json(p)["v"], 1)
            atomic_write_json(p, {"v": 2})
            self.assertEqual(atomic_read_json(p)["v"], 2)


class ManifestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "manifest.json"
        self.chunk_plan = [
            {
                "id": "chunk-000",
                "start_seconds": 0.0,
                "end_seconds": 300.0,
                "overlap_before_seconds": 0.0,
                "path": None,
            },
            {
                "id": "chunk-001",
                "start_seconds": 298.0,
                "end_seconds": 602.0,
                "overlap_before_seconds": 2.0,
                "path": None,
            },
        ]
        self.source = {"sha256": "sha256:abc", "duration_seconds": 720.0, "path": "/fake/source.m4a"}
        self.configuration = {"model": "gpt-4o-transcribe-diarize", "language": None}

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_and_save(self):
        m = Manifest(
            path=self.path,
            run_id="test-run-1",
            run_fingerprint="sha256:abc123",
            source=self.source,
            configuration=self.configuration,
            chunk_plan=self.chunk_plan,
        )
        m.save()
        self.assertTrue(self.path.exists())

    def test_roundtrip(self):
        m = Manifest(
            path=self.path,
            run_id="test-run-1",
            run_fingerprint="sha256:abc123",
            source=self.source,
            configuration=self.configuration,
            chunk_plan=self.chunk_plan,
        )
        m.mark_completed("chunk-000", "results/chunk-000.json", 48)
        m.mark_failed("chunk-001", False, {
            "attempt": 1,
            "started_at": now_iso(),
            "finished_at": now_iso(),
            "elapsed_seconds": 30.0,
            "status": "failed",
            "error_type": "APITimeoutError",
            "request_id": None,
            "retryable": True,
        })
        m.set_stage("transcription", StageStatus.RUNNING)
        m.save()

        loaded = Manifest.load(self.path)
        self.assertEqual(loaded.run_id, "test-run-1")
        self.assertEqual(loaded.run_fingerprint, "sha256:abc123")
        self.assertEqual(loaded.chunk_plan, self.chunk_plan)
        self.assertEqual(loaded.chunks["chunk-000"].status, ChunkStatus.COMPLETED.value)
        self.assertEqual(loaded.chunks["chunk-000"].segment_count, 48)
        self.assertEqual(loaded.chunks["chunk-000"].raw_result, "results/chunk-000.json")
        self.assertEqual(loaded.chunks["chunk-001"].status, ChunkStatus.FAILED_RETRYABLE.value)
        self.assertEqual(len(loaded.chunks["chunk-001"].attempts), 1)
        self.assertEqual(loaded.stages["transcription"].status, StageStatus.RUNNING.value)

    def test_fingerprint_matches(self):
        m = Manifest(
            path=self.path,
            run_id="test-run-1",
            run_fingerprint="sha256:abc123",
            source=self.source,
            configuration=self.configuration,
            chunk_plan=self.chunk_plan,
        )
        self.assertTrue(m.fingerprint_matches("sha256:abc123"))
        self.assertFalse(m.fingerprint_matches("sha256:different"))

    def test_fingerprint_verification_raises(self):
        m = Manifest(
            path=self.path,
            run_id="test-run-1",
            run_fingerprint="sha256:abc123",
            source=self.source,
            configuration=self.configuration,
            chunk_plan=self.chunk_plan,
        )
        with self.assertRaises(ValueError):
            m.verify_fingerprint("sha256:different")

    def test_verify_raw_result_missing_file(self):
        m = Manifest(
            path=self.path,
            run_id="test-run-1",
            run_fingerprint="sha256:abc123",
            source=self.source,
            configuration=self.configuration,
            chunk_plan=self.chunk_plan,
        )
        m.mark_completed("chunk-000", "results/chunk-000.json", 48)
        self.assertFalse(m.verify_raw_result("chunk-000", Path(self.tmp.name)))

    def test_verify_raw_result_valid(self):
        results_dir = Path(self.tmp.name) / "results"
        results_dir.mkdir(parents=True)
        result_path = results_dir / "chunk-000.json"
        atomic_write_json(result_path, {"segments": [{"text": "hello"}]})

        m = Manifest(
            path=self.path,
            run_id="test-run-1",
            run_fingerprint="sha256:abc123",
            source=self.source,
            configuration=self.configuration,
            chunk_plan=self.chunk_plan,
        )
        m.mark_completed("chunk-000", "results/chunk-000.json", 1)
        self.assertTrue(m.verify_raw_result("chunk-000", Path(self.tmp.name)))

    def test_verify_raw_result_corrupt(self):
        results_dir = Path(self.tmp.name) / "results"
        results_dir.mkdir(parents=True)
        result_path = results_dir / "chunk-000.json"
        result_path.write_text("not valid json", encoding="utf-8")

        m = Manifest(
            path=self.path,
            run_id="test-run-1",
            run_fingerprint="sha256:abc123",
            source=self.source,
            configuration=self.configuration,
            chunk_plan=self.chunk_plan,
        )
        m.mark_completed("chunk-000", "results/chunk-000.json", 1)
        self.assertFalse(m.verify_raw_result("chunk-000", Path(self.tmp.name)))

    def test_completed_chunk_count(self):
        m = Manifest(
            path=self.path,
            run_id="test-run-1",
            run_fingerprint="sha256:abc123",
            source=self.source,
            configuration=self.configuration,
            chunk_plan=self.chunk_plan,
        )
        m.mark_completed("chunk-000", "results/chunk-000.json", 10)
        m.mark_completed("chunk-001", "results/chunk-001.json", 12)
        self.assertEqual(m.completed_chunk_count(), 2)

    def test_set_chunk_status_unknown_chunk(self):
        m = Manifest(
            path=self.path,
            run_id="test-run-1",
            run_fingerprint="sha256:abc123",
            source=self.source,
            configuration=self.configuration,
            chunk_plan=[],
        )
        m.set_chunk_status("chunk-new", ChunkStatus.COMPLETED.value, raw_result="r.json", segment_count=5)
        self.assertIn("chunk-new", m.chunks)
        self.assertEqual(m.chunks["chunk-new"].status, ChunkStatus.COMPLETED.value)
        self.assertEqual(m.chunks["chunk-new"].segment_count, 5)

    def test_add_attempt_to_existing_chunk(self):
        m = Manifest(
            path=self.path,
            run_id="test-run-1",
            run_fingerprint="sha256:abc123",
            source=self.source,
            configuration=self.configuration,
            chunk_plan=self.chunk_plan,
        )
        attempt = {
            "attempt": 1,
            "started_at": now_iso(),
            "finished_at": now_iso(),
            "elapsed_seconds": 60.0,
            "status": "succeeded",
            "request_id": "req-123",
        }
        m.add_attempt("chunk-000", attempt)
        self.assertEqual(len(m.chunks["chunk-000"].attempts), 1)
        self.assertEqual(m.chunks["chunk-000"].attempts[0]["request_id"], "req-123")

    def test_schema_version_in_save(self):
        m = Manifest(
            path=self.path,
            run_id="test-run-1",
            run_fingerprint="sha256:abc123",
            source=self.source,
            configuration=self.configuration,
            chunk_plan=self.chunk_plan,
        )
        m.save()
        from json import load
        with self.path.open() as f:
            data = load(f)
        self.assertEqual(data["schema_version"], SCHEMA_VERSION)

    def test_load_wrong_schema_raises(self):
        wrong_data = {
            "schema_version": 1,
            "run_id": "old-run",
            "run_fingerprint": "sha256:old",
            "source": {"sha256": "sha256:old", "duration_seconds": 0},
            "configuration": {},
            "chunk_plan": [],
            "chunks": {},
            "stages": {},
            "status": "running",
            "current_stage": "preflight",
        }
        atomic_write_json(self.path, wrong_data)
        with self.assertRaises(ValueError):
            Manifest.load(self.path)

    def test_stage_set_running_records_started_at(self):
        m = Manifest(
            path=self.path,
            run_id="test-run-1",
            run_fingerprint="sha256:abc123",
            source=self.source,
            configuration=self.configuration,
            chunk_plan=self.chunk_plan,
        )
        m.set_stage("transcription", StageStatus.RUNNING)
        self.assertIsNotNone(m.stages["transcription"].started_at)

    def test_stage_set_completed_records_finished_at(self):
        m = Manifest(
            path=self.path,
            run_id="test-run-1",
            run_fingerprint="sha256:abc123",
            source=self.source,
            configuration=self.configuration,
            chunk_plan=self.chunk_plan,
        )
        m.set_stage("rendering", StageStatus.RUNNING)
        m.set_stage("rendering", StageStatus.COMPLETED)
        self.assertIsNotNone(m.stages["rendering"].started_at)
        self.assertIsNotNone(m.stages["rendering"].finished_at)

    def test_current_stage(self):
        m = Manifest(
            path=self.path,
            run_id="test-run-1",
            run_fingerprint="sha256:abc123",
            source=self.source,
            configuration=self.configuration,
            chunk_plan=self.chunk_plan,
        )
        m.set_current_stage("transcribe")
        self.assertEqual(m.current_stage, "transcribe")

    def test_to_dict_includes_updated_at(self):
        m = Manifest(
            path=self.path,
            run_id="test-run-1",
            run_fingerprint="sha256:abc123",
            source=self.source,
            configuration=self.configuration,
            chunk_plan=self.chunk_plan,
        )
        d = m.to_dict()
        self.assertIn("updated_at", d)


if __name__ == "__main__":
    unittest.main()
