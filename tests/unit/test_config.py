import unittest
from pathlib import Path

from transcriber.config import (
    ChunkPolicy,
    RequestPolicy,
    TextPolicy,
    SpeakerReference,
    RunConfig,
    compute_run_fingerprint,
    make_run_id,
    SCHEMA_VERSION,
)


class ChunkPolicyTest(unittest.TestCase):
    def test_defaults(self):
        cp = ChunkPolicy()
        self.assertEqual(cp.target_seconds, 300)
        self.assertEqual(cp.overlap_seconds, 2)
        self.assertEqual(cp.max_upload_mb, 20.0)
        self.assertEqual(cp.sample_rate, 16000)
        self.assertEqual(cp.channels, 1)
        self.assertEqual(cp.audio_bitrate, "64k")

    def test_immutable(self):
        cp = ChunkPolicy()
        with self.assertRaises(Exception):
            cp.target_seconds = 600  # type: ignore


class RequestPolicyTest(unittest.TestCase):
    def test_defaults(self):
        rp = RequestPolicy()
        self.assertEqual(rp.timeout_seconds, 600)
        self.assertEqual(rp.max_attempts, 2)
        self.assertEqual(rp.workers, 1)


class TextPolicyTest(unittest.TestCase):
    def test_defaults(self):
        tp = TextPolicy()
        self.assertEqual(tp.transcript_style, "verbatim")
        self.assertEqual(tp.analysis_model, "gpt-4.1-mini")
        self.assertFalse(tp.analyze)


class RunConfigFingerprintTest(unittest.TestCase):
    def _base_config(self) -> RunConfig:
        return RunConfig(
            source=Path("/fake/source.m4a"),
            source_sha256="sha256:abc123",
            source_duration_seconds=3600.0,
            model="gpt-4o-transcribe-diarize",
            language=None,
            speakers=(),
            exports=frozenset({"txt", "json"}),
            chunk_policy=ChunkPolicy(),
            request_policy=RequestPolicy(),
            text_policy=TextPolicy(transcript_style="both"),
        )

    def test_fingerprint_stable(self):
        a = self._base_config()
        b = self._base_config()
        self.assertEqual(a.fingerprint(), b.fingerprint())
        self.assertEqual(compute_run_fingerprint(a), compute_run_fingerprint(b))

    def test_model_change_changes_fingerprint(self):
        a = self._base_config()
        b = RunConfig(
            source=Path("/fake/source.m4a"),
            source_sha256="sha256:abc123",
            source_duration_seconds=3600.0,
            model="gpt-4o-transcribe",
            language=None,
            speakers=(),
            exports=frozenset({"txt", "json"}),
            chunk_policy=ChunkPolicy(),
            request_policy=RequestPolicy(),
            text_policy=TextPolicy(transcript_style="both"),
        )
        self.assertNotEqual(a.fingerprint(), b.fingerprint())

    def test_language_change_changes_fingerprint(self):
        a = self._base_config()
        b = RunConfig(
            source=Path("/fake/source.m4a"),
            source_sha256="sha256:abc123",
            source_duration_seconds=3600.0,
            model="gpt-4o-transcribe-diarize",
            language="en",
            speakers=(),
            exports=frozenset({"txt", "json"}),
            chunk_policy=ChunkPolicy(),
            request_policy=RequestPolicy(),
            text_policy=TextPolicy(transcript_style="both"),
        )
        self.assertNotEqual(a.fingerprint(), b.fingerprint())

    def test_source_change_changes_fingerprint(self):
        a = self._base_config()
        b = RunConfig(
            source=Path("/fake/source.m4a"),
            source_sha256="sha256:def456",
            source_duration_seconds=3600.0,
            model="gpt-4o-transcribe-diarize",
            language=None,
            speakers=(),
            exports=frozenset({"txt", "json"}),
            chunk_policy=ChunkPolicy(),
            request_policy=RequestPolicy(),
            text_policy=TextPolicy(transcript_style="both"),
        )
        self.assertNotEqual(a.fingerprint(), b.fingerprint())

    def test_speaker_change_changes_fingerprint(self):
        a = self._base_config()
        b = RunConfig(
            source=Path("/fake/source.m4a"),
            source_sha256="sha256:abc123",
            source_duration_seconds=3600.0,
            model="gpt-4o-transcribe-diarize",
            language=None,
            speakers=(SpeakerReference(
                name="Alice",
                path=Path("/fake/alice.m4a"),
                sha256="sha256:spk1",
                duration_seconds=5.0,
            ),),
            exports=frozenset({"txt", "json"}),
            chunk_policy=ChunkPolicy(),
            request_policy=RequestPolicy(),
            text_policy=TextPolicy(transcript_style="both"),
        )
        self.assertNotEqual(a.fingerprint(), b.fingerprint())

    def test_chunk_policy_change_changes_fingerprint(self):
        a = self._base_config()
        b = RunConfig(
            source=Path("/fake/source.m4a"),
            source_sha256="sha256:abc123",
            source_duration_seconds=3600.0,
            model="gpt-4o-transcribe-diarize",
            language=None,
            speakers=(),
            exports=frozenset({"txt", "json"}),
            chunk_policy=ChunkPolicy(target_seconds=600),
            request_policy=RequestPolicy(),
            text_policy=TextPolicy(transcript_style="both"),
        )
        self.assertNotEqual(a.fingerprint(), b.fingerprint())

    def test_request_policy_change_changes_fingerprint(self):
        a = self._base_config()
        b = RunConfig(
            source=Path("/fake/source.m4a"),
            source_sha256="sha256:abc123",
            source_duration_seconds=3600.0,
            model="gpt-4o-transcribe-diarize",
            language=None,
            speakers=(),
            exports=frozenset({"txt", "json"}),
            chunk_policy=ChunkPolicy(),
            request_policy=RequestPolicy(max_attempts=5),
            text_policy=TextPolicy(transcript_style="both"),
        )
        self.assertNotEqual(a.fingerprint(), b.fingerprint())

    def test_make_run_id_format(self):
        rid = make_run_id()
        parts = rid.split("-")
        self.assertEqual(len(parts), 2)
        self.assertRegex(parts[0], r"^\d{8}T\d{6}Z$")
        self.assertEqual(len(parts[1]), 8)
        self.assertTrue(parts[1].isalnum() or all(c in "0123456789abcdef" for c in parts[1]))

    def test_fingerprint_hash_format(self):
        config = self._base_config()
        fp = config.fingerprint()
        self.assertTrue(fp.startswith("sha256:"))
        self.assertEqual(len(fp), len("sha256:") + 64)

    def test_to_fingerprint_dict_includes_schema_version(self):
        config = self._base_config()
        d = config.to_fingerprint_dict()
        self.assertEqual(d["schema_version"], SCHEMA_VERSION)

    def test_identical_configs_have_identical_fingerprint(self):
        a = self._base_config()
        for _ in range(5):
            b = self._base_config()
            self.assertEqual(a.fingerprint(), b.fingerprint())

    def test_text_policy_style_changes_fingerprint(self):
        a = self._base_config()
        b = RunConfig(
            source=Path("/fake/source.m4a"),
            source_sha256="sha256:abc123",
            source_duration_seconds=3600.0,
            model="gpt-4o-transcribe-diarize",
            language=None,
            speakers=(),
            exports=frozenset({"txt", "json"}),
            chunk_policy=ChunkPolicy(),
            request_policy=RequestPolicy(),
            text_policy=TextPolicy(transcript_style="verbatim"),
        )
        self.assertNotEqual(a.fingerprint(), b.fingerprint())


if __name__ == "__main__":
    unittest.main()
