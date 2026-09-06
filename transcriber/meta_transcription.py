from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

META_BASE_URL = "https://api.meta.ai/v1"
MUSE_TRANSCRIBE_MODEL = "muse-voice-transcribe-1.0"

_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "bn": "Bengali",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "mr": "Marathi",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "zh": "Mandarin Chinese",
}


def resolve_meta_api_key() -> str:
    key = os.getenv("MODEL_API_KEY") or os.getenv("META_API_KEY")
    if not key:
        raise RuntimeError(
            "Meta API key is missing. Set MODEL_API_KEY or META_API_KEY in .env."
        )
    return key


def _language_bias(language: str | None) -> list[str]:
    if not language:
        return []
    normalized = language.strip().lower()
    if normalized in {"auto", "mixed"}:
        return []
    if normalized in {"taglish", "en+tl", "tl+en"}:
        return ["English", "Tagalog"]
    base = normalized.replace("_", "-").split("-", 1)[0]
    return [_LANGUAGE_NAMES.get(base, language)]


def transcribe_muse_file(
    audio_path: Path,
    *,
    language: str | None = None,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    request_payload: dict[str, Any] = {
        "model": MUSE_TRANSCRIBE_MODEL,
        "audioEncoding": "WAV",
        "mode": "DIARIZATION",
    }
    language_bias = _language_bias(language)
    if language_bias:
        request_payload["languageBias"] = language_bias

    headers = {"Authorization": f"Bearer {resolve_meta_api_key()}"}
    with audio_path.open("rb") as audio:
        files = {
            "request": (None, json.dumps(request_payload), "application/json"),
            "audio": (audio_path.name, audio, "audio/wav"),
        }
        response = httpx.post(
            f"{META_BASE_URL}/asr/transcribe",
            headers=headers,
            files=files,
            timeout=float(timeout_seconds),
        )
    response.raise_for_status()
    raw = response.json()

    segments: list[dict[str, Any]] = []
    for turn in raw.get("turns") or []:
        text = str(turn.get("transcript") or "").strip()
        if not text:
            continue
        segments.append(
            {
                "speaker": str(turn.get("speaker") or "Speaker"),
                "start": float(turn.get("startMs", 0)) / 1000.0,
                "end": float(turn.get("endMs", turn.get("startMs", 0))) / 1000.0,
                "text": text,
            }
        )

    return {
        "provider": "meta",
        "model": MUSE_TRANSCRIBE_MODEL,
        "session_id": raw.get("sessionId"),
        "duration_seconds": float(raw.get("audioDurationMs", 0)) / 1000.0,
        "text": str(raw.get("transcript") or "").strip(),
        "segments": segments,
        "raw_response": raw,
    }
