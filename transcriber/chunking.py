from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from transcriber.config import ChunkPolicy
from transcriber.media import MediaInfo, file_hash, run_command


def should_chunk(info: MediaInfo, policy: ChunkPolicy) -> bool:
    size_mb = info.size_bytes / 1024 / 1024
    if info.duration_seconds > policy.target_seconds:
        return True
    if size_mb > policy.max_upload_mb:
        return True
    return False


def plan_chunks(info: MediaInfo, source_path: Path, policy: ChunkPolicy) -> list[dict[str, Any]]:
    if not should_chunk(info, policy):
        return [
            {
                "id": "chunk-000",
                "start_seconds": 0.0,
                "end_seconds": info.duration_seconds,
                "overlap_before_seconds": 0.0,
                "path": None,
                "sha256": None,
            }
        ]

    total = info.duration_seconds
    target = float(policy.target_seconds)
    overlap = float(policy.overlap_seconds)

    chunk_count = max(1, _ceil_div(total, target))
    plan: list[dict[str, Any]] = []

    for i in range(int(chunk_count)):
        chunk_id = f"chunk-{i:03d}"
        if i == 0:
            start = 0.0
        else:
            start = i * target - overlap

        segment_end = min((i + 1) * target + overlap, total)
        end = max(start + 0.1, segment_end)

        overlap_before = 0.0 if i == 0 else overlap

        plan.append({
            "id": chunk_id,
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "overlap_before_seconds": overlap_before,
            "path": None,
            "sha256": None,
        })

    return plan


def extract_chunk(
    plan_item: dict[str, Any],
    source: Path,
    work_dir: Path,
    policy: ChunkPolicy,
) -> None:
    require_binary = shutil.which
    if require_binary("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found in PATH.")

    work_dir.mkdir(parents=True, exist_ok=True)
    chunk_id = plan_item["id"]
    output_path = work_dir / f"{chunk_id}.wav"

    start = plan_item["start_seconds"]
    duration = plan_item["end_seconds"] - plan_item["start_seconds"]

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(source),
        "-t", str(duration),
        "-vn",
        "-ac", "1",
        "-ar", str(policy.sample_rate),
        "-c:a", "pcm_s16le",
        "-map_metadata", "-1",
        str(output_path),
    ]
    run_command(cmd)

    plan_item["path"] = str(output_path)
    plan_item["sha256"] = file_hash(output_path)


def extract_all_chunks(
    plan: list[dict[str, Any]],
    source: Path,
    work_dir: Path,
    policy: ChunkPolicy,
) -> list[dict[str, Any]]:
    for item in plan:
        if not item.get("path") or not Path(item["path"]).exists():
            extract_chunk(item, source, work_dir, policy)
    return plan


def _ceil_div(a: float, b: float) -> int:
    q = a // b
    if a % b == 0:
        return int(q)
    return int(q) + 1
