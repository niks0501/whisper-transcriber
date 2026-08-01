from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from transcriber.config import SCHEMA_VERSION


class ChunkStatus(str, Enum):
    PLANNED = "planned"
    PREPARED = "prepared"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_TERMINAL = "failed_terminal"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    NOT_REQUESTED = "not_requested"


@dataclass
class AttemptRecord:
    attempt: int
    started_at: str
    finished_at: str
    elapsed_seconds: float
    status: str
    error_type: str | None = None
    request_id: str | None = None
    retryable: bool = False


@dataclass
class ChunkRecord:
    id: str
    status: str = ChunkStatus.PREPARED.value
    attempts: list[dict[str, Any]] = field(default_factory=list)
    raw_result: str | None = None
    segment_count: int | None = None


@dataclass
class StageRecord:
    status: str = StageStatus.PENDING.value
    started_at: str | None = None
    finished_at: str | None = None


@dataclass
class RunManifest:
    schema_version: int
    run_id: str
    run_fingerprint: str
    source: dict[str, Any]
    configuration: dict[str, Any]
    chunk_plan: list[dict[str, Any]]
    chunks: dict[str, dict[str, Any]] = field(default_factory=dict)
    stages: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "transcription": {"status": StageStatus.PENDING.value},
        "rendering": {"status": StageStatus.PENDING.value},
        "readable": {"status": StageStatus.PENDING.value},
        "glossary": {"status": StageStatus.PENDING.value},
        "analysis": {"status": StageStatus.NOT_REQUESTED.value},
    })
    status: str = "running"
    current_stage: str = "preflight"
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def atomic_read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class Manifest:
    def __init__(self, path: Path, run_id: str, run_fingerprint: str,
                 source: dict[str, Any], configuration: dict[str, Any],
                 chunk_plan: list[dict[str, Any]]):
        self.path = path
        self.run_id = run_id
        self.run_fingerprint = run_fingerprint
        self.source = dict(source)
        self.configuration = dict(configuration)
        self.chunk_plan = list(chunk_plan)
        self.chunks: dict[str, ChunkRecord] = {}
        for item in chunk_plan:
            cid = item["id"]
            record = ChunkRecord(id=cid)
            record.status = ChunkStatus.PREPARED.value if item.get("path") else ChunkStatus.PREPARED.value
            self.chunks[cid] = record
        self.stages: dict[str, StageRecord] = {
            "transcription": StageRecord(),
            "rendering": StageRecord(),
            "readable": StageRecord(),
            "glossary": StageRecord(),
            "analysis": StageRecord(status=StageStatus.NOT_REQUESTED),
        }
        self.status = "running"
        self.current_stage = "preflight"
        self.schema_version = SCHEMA_VERSION

    @classmethod
    def load(cls, path: Path) -> Manifest:
        data = atomic_read_json(path)
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Manifest schema {data.get('schema_version')} does not match "
                f"expected {SCHEMA_VERSION}. Use --overwrite to start fresh."
            )
        manifest = cls(
            path=path,
            run_id=data["run_id"],
            run_fingerprint=data["run_fingerprint"],
            source=data["source"],
            configuration=data["configuration"],
            chunk_plan=data.get("chunk_plan", []),
        )
        for cid, rec in data.get("chunks", {}).items():
            record = ChunkRecord(
                id=cid,
                status=rec.get("status", ChunkStatus.PREPARED.value),
                attempts=rec.get("attempts", []),
                raw_result=rec.get("raw_result"),
                segment_count=rec.get("segment_count"),
            )
            manifest.chunks[cid] = record
        for stage_name, stage_data in data.get("stages", {}).items():
            manifest.stages[stage_name] = StageRecord(
                status=stage_data.get("status", StageStatus.PENDING.value),
                started_at=stage_data.get("started_at"),
                finished_at=stage_data.get("finished_at"),
            )
        manifest.status = data.get("status", "running")
        manifest.current_stage = data.get("current_stage", "preflight")
        return manifest

    def fingerprint_matches(self, expected: str) -> bool:
        return self.run_fingerprint == expected

    def verify_fingerprint(self, expected: str) -> None:
        if not self.fingerprint_matches(expected):
            raise ValueError(
                f"Run fingerprint mismatch. Configuration or source changed "
                f"since this manifest was created. Use --overwrite to start fresh."
            )

    def verify_raw_result(self, chunk_id: str, base_dir: Path) -> bool:
        record = self.chunks.get(chunk_id)
        if record is None or record.status != ChunkStatus.COMPLETED.value:
            return False
        if record.raw_result is None:
            return False
        result_path = base_dir / record.raw_result
        if not result_path.exists():
            return False
        try:
            data = atomic_read_json(result_path)
            if not isinstance(data, dict):
                return False
        except (json.JSONDecodeError, OSError):
            return False
        return True

    def set_chunk_status(self, chunk_id: str, status: str, raw_result: str | None = None,
                         segment_count: int | None = None, attempt: dict[str, Any] | None = None) -> None:
        record = self.chunks.get(chunk_id)
        if record is None:
            record = ChunkRecord(id=chunk_id)
            self.chunks[chunk_id] = record
        record.status = status
        if raw_result is not None:
            record.raw_result = raw_result
        if segment_count is not None:
            record.segment_count = segment_count
        if attempt is not None:
            record.attempts.append(attempt)

    def mark_completed(self, chunk_id: str, raw_result: str, segment_count: int) -> None:
        self.set_chunk_status(chunk_id, ChunkStatus.COMPLETED.value, raw_result=raw_result, segment_count=segment_count)

    def mark_failed(self, chunk_id: str, terminal: bool, attempt: dict[str, Any]) -> None:
        status = ChunkStatus.FAILED_TERMINAL.value if terminal else ChunkStatus.FAILED_RETRYABLE.value
        self.set_chunk_status(chunk_id, status, attempt=attempt)

    def add_attempt(self, chunk_id: str, attempt: dict[str, Any]) -> None:
        record = self.chunks.get(chunk_id)
        if record is None:
            record = ChunkRecord(id=chunk_id)
            self.chunks[chunk_id] = record
        record.attempts.append(attempt)

    def stage(self, name: str) -> StageRecord:
        return self.stages.setdefault(name, StageRecord())

    def set_stage(self, name: str, status: str) -> None:
        rec = self.stages.get(name)
        if rec is None:
            rec = StageRecord()
            self.stages[name] = rec
        rec.status = status.value if isinstance(status, StageStatus) else status
        if status == StageStatus.RUNNING.value and rec.started_at is None:
            rec.started_at = now_iso()
        elif status in (StageStatus.COMPLETED.value, "failed", StageStatus.FAILED.value):
            rec.finished_at = now_iso()

    def set_current_stage(self, name: str) -> None:
        self.current_stage = name

    def completed_chunk_count(self) -> int:
        return sum(1 for c in self.chunks.values() if c.status == ChunkStatus.COMPLETED.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_fingerprint": self.run_fingerprint,
            "status": self.status,
            "current_stage": self.current_stage,
            "source": self.source,
            "configuration": self.configuration,
            "chunk_plan": self.chunk_plan,
            "chunks": {
                cid: {
                    "status": rec.status,
                    "attempts": rec.attempts,
                    "raw_result": rec.raw_result,
                    "segment_count": rec.segment_count,
                }
                for cid, rec in self.chunks.items()
            },
            "stages": {
                name: {
                    "status": rec.status,
                    "started_at": rec.started_at,
                    "finished_at": rec.finished_at,
                }
                for name, rec in self.stages.items()
            },
            "updated_at": now_iso(),
        }

    def save(self) -> None:
        atomic_write_json(self.path, self.to_dict())

    def reload_if_changed(self) -> bool:
        try:
            data = atomic_read_json(self.path)
        except (OSError, json.JSONDecodeError):
            return False
        return data.get("run_fingerprint") != self.run_fingerprint
