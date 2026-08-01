from __future__ import annotations

from typing import Any


def clock(seconds: float, millis: bool = False, vtt: bool = False) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if millis:
        separator = "." if vtt else ","
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d}{separator}{int((secs % 1) * 1000):03d}"
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d}"


def segments_from_response(payload: dict[str, Any], offset: float, fallback_duration: float) -> list[dict[str, Any]]:
    raw = payload.get("segments") or []
    if raw:
        result = []
        for item in raw:
            result.append({
                "speaker": str(item.get("speaker") or "Speaker"),
                "start": float(item.get("start", 0)) + offset,
                "end": float(item.get("end", item.get("start", 0))) + offset,
                "text": str(item.get("text", "")).strip(),
            })
        return [item for item in result if item["text"]]
    text = str(payload.get("text", "")).strip()
    return [] if not text else [
        {"speaker": "Speaker", "start": offset, "end": offset + fallback_duration, "text": text}
    ]


def transcript_text(items: list[dict[str, Any]]) -> str:
    return (
        "\n\n".join(
            f"[{clock(item['start'])}-{clock(item['end'])}] {item['speaker']}:\n{item['text']}"
            for item in items
        )
        + "\n"
    )


def subtitle(items: list[dict[str, Any]], vtt: bool = False) -> str:
    blocks = ["WEBVTT"] if vtt else []
    for i, item in enumerate(items, 1):
        start = clock(item["start"], True, vtt)
        end = clock(max(item["end"], item["start"] + 0.1), True, vtt)
        body = f"{item['speaker']}: {item['text']}"
        if vtt:
            blocks.append(f"{start} --> {end}\n{body}")
        else:
            blocks.append(f"{i}\n{start} --> {end}\n{body}")
    return "\n\n".join(blocks) + "\n"
