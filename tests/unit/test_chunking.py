import unittest
from pathlib import Path

from transcriber.chunking import plan_chunks, should_chunk, _ceil_div
from transcriber.config import ChunkPolicy
from transcriber.media import MediaInfo


class ShouldChunkTest(unittest.TestCase):
    def test_short_small_no_chunk(self):
        info = MediaInfo(
            duration_seconds=120,
            size_bytes=5 * 1024 * 1024,
            format_name="mp4",
            audio_codec="aac",
            sample_rate=44100,
            channels=2,
        )
        policy = ChunkPolicy(target_seconds=300, max_upload_mb=20.0)
        self.assertFalse(should_chunk(info, policy))

    def test_long_duration_triggers_chunk(self):
        info = MediaInfo(
            duration_seconds=400,
            size_bytes=5 * 1024 * 1024,
            format_name="mp4",
            audio_codec="aac",
            sample_rate=44100,
            channels=2,
        )
        policy = ChunkPolicy(target_seconds=300, max_upload_mb=20.0)
        self.assertTrue(should_chunk(info, policy))

    def test_large_size_triggers_chunk(self):
        info = MediaInfo(
            duration_seconds=60,
            size_bytes=25 * 1024 * 1024,
            format_name="wav",
            audio_codec="pcm_s16le",
            sample_rate=44100,
            channels=2,
        )
        policy = ChunkPolicy(target_seconds=300, max_upload_mb=20.0)
        self.assertTrue(should_chunk(info, policy))

    def test_at_boundary_no_chunk(self):
        info = MediaInfo(
            duration_seconds=300,
            size_bytes=20 * 1024 * 1024,
            format_name="mp4",
            audio_codec="aac",
            sample_rate=44100,
            channels=2,
        )
        policy = ChunkPolicy(target_seconds=300, max_upload_mb=20.0)
        self.assertFalse(should_chunk(info, policy))

    def test_at_boundary_just_over_duration(self):
        info = MediaInfo(
            duration_seconds=301,
            size_bytes=20 * 1024 * 1024,
            format_name="mp4",
            audio_codec="aac",
            sample_rate=44100,
            channels=2,
        )
        policy = ChunkPolicy(target_seconds=300, max_upload_mb=20.0)
        self.assertTrue(should_chunk(info, policy))


class PlanChunksTest(unittest.TestCase):
    def test_no_chunk_returns_single(self):
        info = MediaInfo(
            duration_seconds=120,
            size_bytes=5_000_000,
            format_name="mp4",
            audio_codec="aac",
            sample_rate=44100,
            channels=2,
        )
        policy = ChunkPolicy(target_seconds=300, overlap_seconds=2, max_upload_mb=20.0)
        plan = plan_chunks(info, Path("/fake/source.m4a"), policy)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["id"], "chunk-000")
        self.assertEqual(plan[0]["start_seconds"], 0.0)
        self.assertEqual(plan[0]["end_seconds"], 120.0)
        self.assertEqual(plan[0]["overlap_before_seconds"], 0.0)

    def test_two_hour_recording_chunk_count(self):
        info = MediaInfo(
            duration_seconds=7200,
            size_bytes=18_000_000,
            format_name="mp4",
            audio_codec="aac",
            sample_rate=44100,
            channels=2,
        )
        policy = ChunkPolicy(target_seconds=300, overlap_seconds=2, max_upload_mb=20.0)
        plan = plan_chunks(info, Path("/fake/source.m4a"), policy)
        self.assertEqual(len(plan), 24)

    def test_chunk_ids_are_padded(self):
        info = MediaInfo(
            duration_seconds=1500,
            size_bytes=10_000_000,
            format_name="mp4",
            audio_codec="aac",
            sample_rate=44100,
            channels=2,
        )
        policy = ChunkPolicy(target_seconds=300, overlap_seconds=2, max_upload_mb=20.0)
        plan = plan_chunks(info, Path("/fake/source.m4a"), policy)
        self.assertEqual(plan[0]["id"], "chunk-000")
        self.assertEqual(plan[1]["id"], "chunk-001")
        self.assertEqual(plan[2]["id"], "chunk-002")

    def test_first_chunk_no_overlap(self):
        info = MediaInfo(
            duration_seconds=1500,
            size_bytes=10_000_000,
            format_name="mp4",
            audio_codec="aac",
            sample_rate=44100,
            channels=2,
        )
        policy = ChunkPolicy(target_seconds=300, overlap_seconds=2, max_upload_mb=20.0)
        plan = plan_chunks(info, Path("/fake/source.m4a"), policy)
        self.assertEqual(plan[0]["start_seconds"], 0.0)
        self.assertEqual(plan[0]["overlap_before_seconds"], 0.0)

    def test_non_first_chunks_have_overlap(self):
        info = MediaInfo(
            duration_seconds=1500,
            size_bytes=10_000_000,
            format_name="mp4",
            audio_codec="aac",
            sample_rate=44100,
            channels=2,
        )
        policy = ChunkPolicy(target_seconds=300, overlap_seconds=2, max_upload_mb=20.0)
        plan = plan_chunks(info, Path("/fake/source.m4a"), policy)
        for i in range(1, len(plan)):
            self.assertEqual(plan[i]["overlap_before_seconds"], 2.0)

    def test_boundary_timestamps(self):
        info = MediaInfo(
            duration_seconds=3600,
            size_bytes=15_000_000,
            format_name="mp4",
            audio_codec="aac",
            sample_rate=44100,
            channels=2,
        )
        policy = ChunkPolicy(target_seconds=300, overlap_seconds=2, max_upload_mb=20.0)
        plan = plan_chunks(info, Path("/fake/source.m4a"), policy)
        self.assertEqual(plan[0]["start_seconds"], 0.0)
        self.assertEqual(plan[0]["end_seconds"], 302.0)
        self.assertEqual(plan[1]["start_seconds"], 298.0)
        self.assertEqual(plan[1]["end_seconds"], 602.0)
        self.assertEqual(plan[2]["start_seconds"], 598.0)
        self.assertEqual(plan[2]["end_seconds"], 902.0)

    def test_chunk_7_boundary(self):
        info = MediaInfo(
            duration_seconds=7200,
            size_bytes=18_000_000,
            format_name="mp4",
            audio_codec="aac",
            sample_rate=44100,
            channels=2,
        )
        policy = ChunkPolicy(target_seconds=300, overlap_seconds=2, max_upload_mb=20.0)
        plan = plan_chunks(info, Path("/fake/source.m4a"), policy)
        self.assertEqual(plan[7]["start_seconds"], 2098.0)
        self.assertEqual(plan[7]["end_seconds"], 2402.0)

    def test_last_chunk_does_not_exceed_total(self):
        info = MediaInfo(
            duration_seconds=400,
            size_bytes=10_000_000,
            format_name="mp4",
            audio_codec="aac",
            sample_rate=44100,
            channels=2,
        )
        policy = ChunkPolicy(target_seconds=300, overlap_seconds=2, max_upload_mb=20.0)
        plan = plan_chunks(info, Path("/fake/source.m4a"), policy)
        self.assertEqual(plan[-1]["end_seconds"], 400.0)
        last_duration = plan[-1]["end_seconds"] - plan[-1]["start_seconds"]
        self.assertLess(last_duration, policy.target_seconds + 2 * policy.overlap_seconds)

    def test_all_chunks_cover_full_range(self):
        info = MediaInfo(
            duration_seconds=3600,
            size_bytes=15_000_000,
            format_name="mp4",
            audio_codec="aac",
            sample_rate=44100,
            channels=2,
        )
        policy = ChunkPolicy(target_seconds=300, overlap_seconds=2, max_upload_mb=20.0)
        plan = plan_chunks(info, Path("/fake/source.m4a"), policy)
        self.assertEqual(plan[0]["start_seconds"], 0.0)
        self.assertEqual(plan[-1]["end_seconds"], 3600.0)

    def test_short_file_exactly_target_no_chunking(self):
        info = MediaInfo(
            duration_seconds=300,
            size_bytes=15_000_000,
            format_name="mp4",
            audio_codec="aac",
            sample_rate=44100,
            channels=2,
        )
        policy = ChunkPolicy(target_seconds=300, overlap_seconds=2, max_upload_mb=20.0)
        plan = plan_chunks(info, Path("/fake/source.m4a"), policy)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["id"], "chunk-000")

    def test_very_short_file(self):
        info = MediaInfo(
            duration_seconds=5.0,
            size_bytes=100_000,
            format_name="wav",
            audio_codec="pcm_s16le",
            sample_rate=44100,
            channels=1,
        )
        policy = ChunkPolicy(target_seconds=300, overlap_seconds=2, max_upload_mb=20.0)
        plan = plan_chunks(info, Path("/fake/source.wav"), policy)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["end_seconds"], 5.0)

    def test_chunk_intervals_are_monotonic(self):
        info = MediaInfo(
            duration_seconds=7200,
            size_bytes=18_000_000,
            format_name="mp4",
            audio_codec="aac",
            sample_rate=44100,
            channels=2,
        )
        policy = ChunkPolicy(target_seconds=300, overlap_seconds=2, max_upload_mb=20.0)
        plan = plan_chunks(info, Path("/fake/source.m4a"), policy)
        prev_end = plan[0]["end_seconds"]
        for chunk in plan[1:]:
            self.assertLess(chunk["start_seconds"], prev_end,
                            f"{chunk['id']} starts at {chunk['start_seconds']} "
                            f"but previous ended at {prev_end}")
            prev_end = chunk["end_seconds"]


class CeilDivTest(unittest.TestCase):
    def test_exact_division(self):
        self.assertEqual(_ceil_div(600, 300), 2)

    def test_remainder(self):
        self.assertEqual(_ceil_div(400, 300), 2)

    def test_small_number(self):
        self.assertEqual(_ceil_div(1, 300), 1)

    def test_zero(self):
        self.assertEqual(_ceil_div(0, 300), 0)


if __name__ == "__main__":
    unittest.main()
