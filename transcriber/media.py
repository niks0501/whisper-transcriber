from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MediaInfo:
    duration_seconds: float
    size_bytes: int
    format_name: str
    audio_codec: str
    sample_rate: int
    channels: int


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_name(path: Path) -> str:
    clean = "".join(c if c.isalnum() or c in "-_" else "-" for c in path.stem).strip("-")
    return clean or "transcript"


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} was not found in PATH. Install FFmpeg and reopen the terminal.")


def run_command(command: list[str], timeout: float = 600.0) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timed out after {timeout:.0f}s: {' '.join(command)}")
    if result.returncode:
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{result.stderr.strip()}")
    return result.stdout.strip()


def probe(path: Path) -> MediaInfo:
    output = run_command([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,size,format_name",
        "-show_entries", "stream=codec_name,sample_rate,channels",
        "-of", "json", str(path),
    ])
    data = json.loads(output)
    fmt = data.get("format", {})
    streams = data.get("stream", [])
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"] if streams else data.get("streams", [])
    audio_streams = [s for s in audio_streams if s.get("codec_type") == "audio"]
    audio = audio_streams[0] if audio_streams else {}
    duration = float(fmt.get("duration", 0))
    size = int(fmt.get("size", 0))
    format_name = fmt.get("format_name", "unknown")
    return MediaInfo(
        duration_seconds=duration,
        size_bytes=size,
        format_name=format_name,
        audio_codec=audio.get("codec_name", "unknown"),
        sample_rate=int(audio.get("sample_rate", 0)),
        channels=int(audio.get("channels", 0)),
    )


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def format_duration(seconds: float) -> str:
    from datetime import timedelta

    td = timedelta(seconds=int(seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def guess_mime(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0]
    return mime or "audio/wav"
