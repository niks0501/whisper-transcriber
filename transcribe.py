from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI

PROFILES = {
    "interview": "gpt-4o-transcribe-diarize",
    "accurate": "gpt-4o-transcribe",
    "budget": "gpt-4o-mini-transcribe",
    "legacy": "whisper-1",
}
SUPPORTED = {".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"}
EXPORTS = {"txt", "json", "srt", "vtt"}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interview transcription with diarization, resume support, subtitles, and thesis analysis.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("audio", nargs="?", help="Raw interview audio/video path")
    p.add_argument("--wizard", action="store_true", help="Open the guided setup wizard")
    p.add_argument("--profile", choices=PROFILES, default=None)
    p.add_argument("--model", help="Override the selected transcription model")
    p.add_argument("--language", help="ISO language code, such as en or tl; use auto for detection")
    p.add_argument("--speaker", action="append", default=[], metavar="NAME=FILE", help="Known speaker 2-10 second reference; repeatable")
    p.add_argument("--speaker-label", action="append", default=[], metavar="RAW=DISPLAY", help="Rename detected labels, e.g. A=Interviewer")
    p.add_argument("--map-speakers", action="store_true", help="Rename detected speakers interactively after transcription")
    p.add_argument("--glossary", help="UTF-8 glossary, one approved term per line")
    p.add_argument("--transcript-style", choices=["verbatim", "readable", "both"], default=None)
    p.add_argument("--export", default="txt,json,srt,vtt", help="Comma-separated: txt,json,srt,vtt")
    p.add_argument("--analyze", action="store_true", help="Create thesis-oriented research notes")
    p.add_argument("--research-context", help="YAML file with project and research questions")
    p.add_argument("--analysis-model", default=os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini"))
    p.add_argument("--output", default="transcription_output")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--retries", type=int, default=3)
    return p


def choose(title: str, items: list[str], default: int = 1) -> int:
    print(f"\n{title}")
    for i, item in enumerate(items, 1):
        suffix = " (default)" if i == default else ""
        print(f"  {i}. {item}{suffix}")
    while True:
        raw = input("> ").strip()
        if not raw:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(items):
            return int(raw)
        print("Please enter one of the listed numbers.")


def yes_no(question: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = input(f"{question} [{hint}] ").strip().lower()
    return default if not raw else raw in {"y", "yes"}


def wizard(args: argparse.Namespace) -> argparse.Namespace:
    print("\nWhisper Transcriber — Interview Wizard")
    if not args.audio:
        args.audio = input("\nRaw audio/video path:\n> ").strip().strip('"')

    profile = choose("Profile:", ["Interview with automatic speaker separation", "Accurate plain transcription", "Lower-cost plain transcription", "Legacy Whisper"])
    args.profile = ["interview", "accurate", "budget", "legacy"][profile - 1]

    language = choose("Language:", ["Automatic or mixed language", "English", "Tagalog", "Another ISO code"])
    args.language = ["auto", "en", "tl", None][language - 1]
    if language == 4:
        args.language = input("Language code:\n> ").strip()

    if args.profile == "interview":
        if yes_no("Do you have clean 2-10 second speaker samples?"):
            while True:
                name = input("Speaker label (for example Interviewer):\n> ").strip()
                path = input("Reference sample path:\n> ").strip().strip('"')
                args.speaker.append(f"{name}={path}")
                if not yes_no("Add another known speaker?"):
                    break
        else:
            args.map_speakers = yes_no("Rename Speaker A/B after transcription?", True)

    style = choose("Transcript copies:", ["Verbatim only", "Readable only (raw JSON remains protected)", "Both verbatim and readable"], 3)
    args.transcript_style = ["verbatim", "readable", "both"][style - 1]

    if yes_no("Use a terminology glossary?"):
        args.glossary = input("Glossary path:\n> ").strip().strip('"')

    args.analyze = yes_no("Generate thesis research notes?")
    if args.analyze and yes_no("Use a research-context YAML file?", True):
        args.research_context = input("Research context path:\n> ").strip().strip('"')

    print("\nThe wizard will export TXT, JSON, SRT, and VTT.")
    return args


def assignment(value: str, option: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"{option} expects NAME=VALUE: {value}")
    left, right = value.split("=", 1)
    if not left.strip() or not right.strip():
        raise ValueError(f"{option} expects non-empty NAME=VALUE")
    return left.strip(), right.strip()


def run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{result.stderr.strip()}")
    return result.stdout.strip()


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} was not found in PATH. Install FFmpeg and reopen the terminal.")


def duration(path: Path) -> float:
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)]))


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_name(path: Path) -> str:
    clean = "".join(c if c.isalnum() or c in "-_" else "-" for c in path.stem).strip("-")
    return clean or "transcript"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def prepare_chunks(source: Path, work: Path, max_mb: float = 24.0) -> list[dict[str, Any]]:
    size_mb = source.stat().st_size / 1024 / 1024
    if size_mb <= max_mb:
        return [{"path": source, "offset": 0.0, "key": "chunk-000"}]

    work.mkdir(parents=True, exist_ok=True)
    pattern = work / "chunk-%03d.mp3"
    run(["ffmpeg", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", "-f", "segment", "-segment_time", "900", "-reset_timestamps", "1", str(pattern)])
    chunks = sorted(work.glob("chunk-*.mp3"))
    if not chunks:
        raise RuntimeError("FFmpeg did not create any chunks.")
    return [{"path": item, "offset": i * 900.0, "key": f"chunk-{i:03d}"} for i, item in enumerate(chunks)]


def as_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if isinstance(response, dict):
        return response
    return json.loads(response.json())


def retry(label: str, attempts: int, operation):
    last = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            last = exc
            if attempt == attempts:
                break
            delay = min(2 ** (attempt - 1), 8)
            print(f"{label} failed ({attempt}/{attempts}); retrying in {delay}s: {exc}")
            time.sleep(delay)
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last}")


def transcribe(client: OpenAI, chunk: Path, model: str, language: str | None, speakers: list[tuple[str, Path]], attempts: int) -> dict[str, Any]:
    def call():
        with chunk.open("rb") as audio:
            kwargs: dict[str, Any] = {"model": model, "file": audio}
            if language:
                kwargs["language"] = language
            if model == "gpt-4o-transcribe-diarize":
                kwargs["response_format"] = "diarized_json"
                kwargs["chunking_strategy"] = "auto"
                if speakers:
                    kwargs["known_speaker_names"] = [name for name, _ in speakers]
                    kwargs["known_speaker_references"] = [as_data_url(path) for _, path in speakers]
            elif model == "whisper-1":
                kwargs["response_format"] = "verbose_json"
                kwargs["timestamp_granularities"] = ["segment"]
            else:
                kwargs["response_format"] = "json"
            return to_dict(client.audio.transcriptions.create(**kwargs))
    return retry(f"Transcription of {chunk.name}", attempts, call)


def segments(payload: dict[str, Any], offset: float, fallback_duration: float) -> list[dict[str, Any]]:
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
    return [] if not text else [{"speaker": "Speaker", "start": offset, "end": offset + fallback_duration, "text": text}]


def clock(seconds: float, millis: bool = False, vtt: bool = False) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if millis:
        separator = "." if vtt else ","
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d}{separator}{int((secs % 1) * 1000):03d}"
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d}"


def rename_speakers(items: list[dict[str, Any]], labels: dict[str, str], interactive: bool) -> dict[str, str]:
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


def transcript_text(items: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"[{clock(item['start'])}-{clock(item['end'])}] {item['speaker']}:\n{item['text']}" for item in items) + "\n"


def subtitle(items: list[dict[str, Any]], vtt: bool = False) -> str:
    blocks = ["WEBVTT\n"] if vtt else []
    for i, item in enumerate(items, 1):
        start = clock(item["start"], True, vtt)
        end = clock(max(item["end"], item["start"] + 0.1), True, vtt)
        body = f"{item['speaker']}: {item['text']}"
        blocks.append(f"{start} --> {end}\n{body}\n" if vtt else f"{i}\n{start} --> {end}\n{body}\n")
    return "\n".join(blocks).rstrip() + "\n"


def text_model(client: OpenAI, model: str, prompt: str, attempts: int) -> str:
    response = retry("Text generation", attempts, lambda: client.responses.create(model=model, input=prompt))
    output = getattr(response, "output_text", None)
    if output:
        return output.strip()
    return str(to_dict(response))


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parser().parse_args(argv)
    try:
        if args.wizard:
            args = wizard(args)
        if not args.audio:
            raise ValueError("Provide an audio path or run with --wizard.")
        source = Path(args.audio).expanduser().resolve()
        if not source.is_file() or source.suffix.lower() not in SUPPORTED:
            raise ValueError(f"Unsupported or missing audio file: {source}")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing. Copy .env.example to .env and add the key.")
        require_binary("ffmpeg")
        require_binary("ffprobe")

        profile = args.profile or "interview"
        model = args.model or PROFILES[profile]
        language = None if not args.language or args.language.lower() == "auto" else args.language
        style = args.transcript_style or "both"
        exports = {x.strip().lower() for x in args.export.split(",") if x.strip()}
        if exports - EXPORTS:
            raise ValueError(f"Unsupported exports: {', '.join(sorted(exports - EXPORTS))}")

        refs: list[tuple[str, Path]] = []
        for value in args.speaker:
            name, raw_path = assignment(value, "--speaker")
            ref = Path(raw_path).expanduser().resolve()
            if not ref.is_file():
                raise ValueError(f"Speaker sample not found: {ref}")
            ref_duration = duration(ref)
            if not 2 <= ref_duration <= 10:
                raise ValueError(f"Speaker sample {ref.name} must be 2-10 seconds; found {ref_duration:.1f}s")
            refs.append((name, ref))
        if len(refs) > 4:
            raise ValueError("At most four known-speaker references are supported.")

        labels = dict(assignment(value, "--speaker-label") for value in args.speaker_label)
        output = Path(args.output) / safe_name(source)
        if args.overwrite and output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / "run_manifest.json"
        source_sha = file_hash(source)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"chunks": {}}
        if manifest.get("source_sha256") not in {None, source_sha}:
            raise RuntimeError("Output folder belongs to a different source. Use --overwrite or another --output.")
        manifest.update({"source": str(source), "source_sha256": source_sha, "model": model, "status": "running", "updated_at": datetime.now(timezone.utc).isoformat()})
        atomic_json(manifest_path, manifest)

        chunks = prepare_chunks(source, output / "working" / "audio")
        client = OpenAI()
        all_segments: list[dict[str, Any]] = []
        raw_requests: list[dict[str, Any]] = []
        results_dir = output / "working" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        if len(chunks) > 1 and model == PROFILES["interview"] and not refs:
            print("Warning: speaker letters may change across chunks. Known-speaker samples improve consistency.")

        for index, chunk in enumerate(chunks, 1):
            result_path = results_dir / f"{chunk['key']}.json"
            reusable = not args.no_resume and manifest["chunks"].get(chunk["key"], {}).get("status") == "completed" and result_path.exists()
            if reusable:
                print(f"[{index}/{len(chunks)}] Reusing {chunk['key']}")
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                print(f"[{index}/{len(chunks)}] Transcribing {chunk['path'].name}")
                payload = transcribe(client, chunk["path"], model, language, refs, args.retries)
                atomic_json(result_path, payload)
                manifest["chunks"][chunk["key"]] = {"status": "completed", "result": str(result_path)}
                atomic_json(manifest_path, manifest)
            raw_requests.append({"chunk": chunk["key"], "offset": chunk["offset"], "response": payload})
            all_segments.extend(segments(payload, chunk["offset"], duration(chunk["path"])))

        all_segments.sort(key=lambda item: (item["start"], item["end"]))
        for name, _ in refs:
            labels.setdefault(name, name)
        labels = rename_speakers(all_segments, labels, args.map_speakers)
        raw = {"source": str(source), "source_sha256": source_sha, "model": model, "speaker_label_map": labels, "segments": all_segments, "requests": raw_requests}
        atomic_json(output / "raw_transcript.json", raw)
        if "json" in exports:
            atomic_json(output / "transcript.json", {"segments": all_segments})
        verbatim = transcript_text(all_segments)
        if "txt" in exports and style in {"verbatim", "both"}:
            atomic_text(output / "verbatim_transcript.txt", verbatim)
        if "srt" in exports:
            atomic_text(output / "transcript.srt", subtitle(all_segments))
        if "vtt" in exports:
            atomic_text(output / "transcript.vtt", subtitle(all_segments, True))

        glossary = Path(args.glossary).read_text(encoding="utf-8") if args.glossary else ""
        if style in {"readable", "both"}:
            prompt = f"""Create a readable copy of this interview transcript. Preserve every timestamp, speaker label, claim, number, uncertainty, and language choice. Fix punctuation and obvious formatting only. Do not summarize or invent. Use [VERIFY] when a glossary correction is uncertain.\n\nApproved glossary:\n{glossary or '(none)'}\n\nTranscript:\n{verbatim}"""
            atomic_text(output / "readable_transcript.txt", text_model(client, args.analysis_model, prompt, args.retries) + "\n")

        if glossary:
            review_prompt = f"""Review the transcript against the glossary. Return valid JSON only with an items array. Each item must have timestamp, speaker, excerpt, and reason. Flag suspicious names or technical terms; do not invent acoustic confidence.\n\nGlossary:\n{glossary}\n\nTranscript:\n{verbatim}"""
            review_raw = text_model(client, args.analysis_model, review_prompt, args.retries).strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                atomic_json(output / "review_flags.json", json.loads(review_raw))
            except json.JSONDecodeError:
                atomic_json(output / "review_flags.json", {"items": [], "unparsed_model_output": review_raw})

        if args.analyze:
            context = yaml.safe_load(Path(args.research_context).read_text(encoding="utf-8")) if args.research_context else {}
            analysis_prompt = f"""Analyze this research interview using only the transcript. Never invent quotations or findings. Keep speaker names and timestamps with every quotation. Clearly distinguish evidence from interpretation. Return Markdown with exactly these headings: # Interview Summary, # Themes and Subthemes, # Key Quotations, # Question-and-Answer Map, # Follow-up Questions.\n\nResearch context:\n{yaml.safe_dump(context, allow_unicode=True, sort_keys=False)}\n\nTranscript:\n{verbatim}"""
            atomic_text(output / "research" / "research_analysis.md", text_model(client, args.analysis_model, analysis_prompt, args.retries) + "\n")

        manifest.update({"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(), "speaker_label_map": labels, "segment_count": len(all_segments)})
        atomic_json(manifest_path, manifest)
        print(f"\nDone. Outputs: {output}")
        return 0
    except KeyboardInterrupt:
        print("\nCancelled. Completed chunks remain available for resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
