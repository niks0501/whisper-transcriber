from __future__ import annotations

from pathlib import Path
from typing import Any

from transcriber.config import SpeakerReference
from transcriber.media import file_hash, probe


def validate_speaker_reference(name: str, path: Path) -> SpeakerReference:
    if not path.is_file():
        raise ValueError(f"Speaker sample not found: {path}")
    info = probe(path)
    duration = info.duration_seconds
    if not 2 <= duration <= 10:
        raise ValueError(
            f"Speaker sample {path.name} must be 2-10 seconds; found {duration:.1f}s"
        )
    return SpeakerReference(
        name=name,
        path=path,
        sha256=file_hash(path),
        duration_seconds=duration,
    )


def rename_speakers(
    items: list[dict[str, Any]],
    labels: dict[str, str],
    interactive: bool,
) -> dict[str, str]:
    detected = sorted({str(item["speaker"]) for item in items})
    if interactive and detected:
        print("\nDetected speakers:")
        for raw in detected:
            current = labels.get(raw, raw)
            entered = input(f"Display name for {raw} [{current}]: ").strip()
            labels[raw] = entered or current
    for item in items:
        item["speaker"] = labels.get(str(item["speaker"]), str(item["speaker"]))
    return labels
