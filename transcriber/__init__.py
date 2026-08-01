"""Long-audio transcription pipeline for interview recordings.

This package refactors the single-file transcriber into testable modules.
See docs/TRANSCRIBER_LONG_AUDIO_REFACTOR_PLAN.md for the full design.
"""

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
from transcriber.manifest import (
    ChunkStatus,
    StageStatus,
    Manifest,
)

__all__ = [
    "ChunkPolicy",
    "RequestPolicy",
    "TextPolicy",
    "SpeakerReference",
    "RunConfig",
    "compute_run_fingerprint",
    "make_run_id",
    "SCHEMA_VERSION",
    "ChunkStatus",
    "StageStatus",
    "Manifest",
]
