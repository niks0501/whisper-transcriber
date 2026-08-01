from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SpeakerReference:
    name: str
    path: Path
    sha256: str
    duration_seconds: float


@dataclass(frozen=True)
class ChunkPolicy:
    target_seconds: int = 300
    overlap_seconds: int = 2
    max_upload_mb: float = 20.0
    sample_rate: int = 16000
    channels: int = 1
    audio_bitrate: str = "64k"


@dataclass(frozen=True)
class RequestPolicy:
    timeout_seconds: int = 600
    max_attempts: int = 2
    workers: int = 1


@dataclass(frozen=True)
class TextPolicy:
    transcript_style: str = "verbatim"
    analysis_model: str = "gpt-4.1-mini"
    analyze: bool = False
    glossary_path: Path | None = None
    research_context_path: Path | None = None


@dataclass(frozen=True)
class RunConfig:
    source: Path
    source_sha256: str
    source_duration_seconds: float
    model: str
    language: str | None
    speakers: tuple[SpeakerReference, ...]
    exports: frozenset[str]
    chunk_policy: ChunkPolicy = field(default_factory=ChunkPolicy)
    request_policy: RequestPolicy = field(default_factory=RequestPolicy)
    text_policy: TextPolicy = field(default_factory=TextPolicy)

    def speaker_names(self) -> list[str]:
        return [s.name for s in self.speakers]

    def speaker_fingerprints(self) -> list[tuple[str, str]]:
        return [(s.name, s.sha256) for s in self.speakers]

    def to_fingerprint_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_sha256": self.source_sha256,
            "model": self.model,
            "language": self.language,
            "speakers": self.speaker_fingerprints(),
            "chunk_policy": asdict(self.chunk_policy),
            "request_policy": asdict(self.request_policy),
            "text_policy": {
                "transcript_style": self.text_policy.transcript_style,
                "analyze": self.text_policy.analyze,
            },
        }

    def fingerprint(self) -> str:
        return compute_run_fingerprint(self)


def compute_run_fingerprint(config: RunConfig) -> str:
    payload = config.to_fingerprint_dict()
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def make_run_id() -> str:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    import os

    suffix = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
    return f"{stamp}-{suffix}"
