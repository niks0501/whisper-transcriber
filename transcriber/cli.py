from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

PROFILES = {
    "interview": "gpt-4o-transcribe-diarize",
    "accurate": "gpt-4o-transcribe",
    "budget": "gpt-4o-mini-transcribe",
    "legacy": "whisper-1",
}

SUPPORTED = {".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"}
EXPORTS = {"txt", "json", "srt", "vtt"}

SCHEMA_VERSION = 2


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcribe.py",
        description="Interview transcription with diarization, resume support, subtitles, and thesis analysis.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("audio", nargs="?", help="Raw interview audio/video path")
    p.add_argument("--wizard", action="store_true", help="Open the guided setup wizard")

    p.add_argument("--profile", choices=PROFILES, default=None,
                   help="Named model profile: interview, accurate, budget, legacy")
    p.add_argument("--model", help="Override the selected transcription model")
    p.add_argument("--language", help="ISO language code, such as en or tl; use auto for detection")

    p.add_argument("--speaker", action="append", default=[], metavar="NAME=FILE",
                   help="Known speaker 2-10 second reference; repeatable")
    p.add_argument("--speaker-label", action="append", default=[], metavar="RAW=DISPLAY",
                   help="Rename detected labels, e.g. A=Interviewer")
    p.add_argument("--map-speakers", action="store_true",
                   help="Rename detected speakers interactively after transcription")

    p.add_argument("--glossary", help="UTF-8 glossary, one approved term per line")
    p.add_argument("--transcript-style", choices=["verbatim", "readable", "both"], default=None)
    p.add_argument("--export", default="txt,json,srt,vtt",
                   help="Comma-separated: txt,json,srt,vtt")
    p.add_argument("--analyze", action="store_true",
                   help="Create thesis-oriented research notes")
    p.add_argument("--research-context", help="YAML file with project and research questions")
    p.add_argument("--analysis-model", default=None,
                   help="Text model for readable cleanup and analysis")

    p.add_argument("--output", default="transcription_output")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--retries", type=int, default=None,
                   help="[Deprecated] Use --max-attempts (total attempts including first).")

    p.add_argument("--chunk-seconds", type=int, default=None,
                   help="Target chunk duration in seconds (default 300)")
    p.add_argument("--chunk-overlap-seconds", type=int, default=None,
                   help="Overlap between adjacent chunks in seconds (default 2)")
    p.add_argument("--request-timeout", type=int, default=None,
                   help="Per-request timeout in seconds (default 600)")
    p.add_argument("--max-attempts", type=int, default=None,
                   help="Maximum API attempts per chunk, including first (default 2)")
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel transcription workers (default 1)")
    p.add_argument("--transcribe-only", action="store_true",
                   help="Skip readable cleanup, glossary review, and analysis")
    p.add_argument("--postprocess-only", action="store_true",
                   help="Skip transcription; resume only post-processing")
    p.add_argument("--status", action="store_true",
                   help="Show run status and exit")

    p.add_argument("--max-failed-chunks", type=int, default=None,
                   help="Stop after this many terminal chunk failures (default 1)")
    p.add_argument("--max-total-attempts", type=int, default=None,
                   help="Stop after this many total API attempts across all chunks")
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


def assignment(value: str, option: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"{option} expects NAME=VALUE: {value}")
    left, right = value.split("=", 1)
    if not left.strip() or not right.strip():
        raise ValueError(f"{option} expects non-empty NAME=VALUE")
    return left.strip(), right.strip()


def wizard(args: argparse.Namespace) -> argparse.Namespace:
    print("\nWhisper Transcriber — Interview Wizard")
    if not args.audio:
        args.audio = input("\nRaw audio/video path:\n> ").strip().strip('"')

    profile = choose("Profile:", [
        "Interview with automatic speaker separation",
        "Accurate plain transcription",
        "Lower-cost plain transcription",
        "Legacy Whisper",
    ])
    args.profile = ["interview", "accurate", "budget", "legacy"][profile - 1]

    language = choose("Language:", [
        "Automatic or mixed language",
        "English",
        "Tagalog",
        "Another ISO code",
    ])
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

    style = choose("Transcript copies:", [
        "Verbatim only",
        "Readable only (raw JSON remains protected)",
        "Both verbatim and readable",
    ], 3)
    args.transcript_style = ["verbatim", "readable", "both"][style - 1]

    if yes_no("Use a terminology glossary?"):
        args.glossary = input("Glossary path:\n> ").strip().strip('"')

    args.analyze = yes_no("Generate thesis research notes?")
    if args.analyze and yes_no("Use a research-context YAML file?", True):
        args.research_context = input("Research context path:\n> ").strip().strip('"')

    print("\nThe wizard will export TXT, JSON, SRT, and VTT.")
    return args
