"""Thin CLI entrypoint for the Whisper Transcriber.

All transcription logic lives in the transcriber/ package.
See docs/TRANSCRIBER_LONG_AUDIO_REFACTOR_PLAN.md for the design.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from transcriber import RunConfig, ChunkPolicy, RequestPolicy, TextPolicy, SpeakerReference
from transcriber.api_client import create_client
from transcriber.cli import parser as make_parser, wizard, assignment
from transcriber.media import file_hash, probe, safe_name, require_binary
from transcriber.pipeline import run_pipeline


SUPPORTED = {".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"}
EXPORTS = {"txt", "json", "srt", "vtt"}

PROFILES = {
    "interview": "muse-voice-transcribe-1.0",
    "openai-interview": "gpt-4o-transcribe-diarize",
    "accurate": "gpt-4o-transcribe",
    "budget": "gpt-4o-mini-transcribe",
    "legacy": "whisper-1",
}


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = make_parser()
    args = p.parse_args(argv)

    try:
        if args.wizard:
            args = wizard(args)

        if args.status:
            return _show_status(args)

        if not args.audio:
            raise ValueError("Provide an audio path or run with --wizard.")

        source = Path(args.audio).expanduser().resolve()
        if not source.is_file() or source.suffix.lower() not in SUPPORTED:
            raise ValueError(f"Unsupported or missing audio file: {source}")

        require_binary("ffmpeg")
        require_binary("ffprobe")

        profile = args.profile or "interview"
        model = args.model or PROFILES[profile]
        language = None if not args.language or args.language.lower() == "auto" else args.language
        style = args.transcript_style or "both"

        if model == "muse-voice-transcribe-1.0":
            if not (os.getenv("MODEL_API_KEY") or os.getenv("META_API_KEY")):
                raise RuntimeError(
                    "Meta API key is missing. Set MODEL_API_KEY or META_API_KEY in .env."
                )
        elif not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Copy .env.example to .env and add the key."
            )

        exports = {x.strip().lower() for x in args.export.split(",") if x.strip()}
        if exports - EXPORTS:
            raise ValueError(f"Unsupported exports: {', '.join(sorted(exports - EXPORTS))}")

        speakers = _build_speakers(args.speaker)
        source_sha = file_hash(source)
        info = probe(source)
        if info.duration_seconds <= 0:
            raise ValueError(
                f"Could not determine a positive audio duration for {source}. "
                f"Check that the file is valid and complete."
            )

        max_attempts = args.max_attempts
        if max_attempts is None and args.retries is not None:
            max_attempts = args.retries
        if max_attempts is None:
            max_attempts = 2
        if max_attempts < 1:
            raise ValueError("--max-attempts/--retries must be at least 1.")

        chunk_target = args.chunk_seconds or 300
        chunk_overlap = args.chunk_overlap_seconds or 2
        if chunk_target < 1:
            raise ValueError("--chunk-seconds must be at least 1.")
        if chunk_overlap < 0:
            raise ValueError("--chunk-overlap-seconds cannot be negative.")
        if chunk_overlap >= chunk_target:
            raise ValueError(
                f"--chunk-overlap-seconds ({chunk_overlap}) must be smaller than "
                f"--chunk-seconds ({chunk_target})."
            )

        analysis_model = args.analysis_model or os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")

        if speakers and model != "gpt-4o-transcribe-diarize":
            print(
                "Warning: --speaker references are only used by the OpenAI diarize profile "
                f"({PROFILES['openai-interview']}); ignoring them for model {model}.",
                file=sys.stderr,
            )

        if model == "muse-voice-transcribe-1.0" and not os.getenv("OPENAI_API_KEY"):
            if not args.transcribe_only and (style in {"readable", "both"} or args.glossary or args.analyze):
                print(
                    "Warning: Muse transcription will work, but readable cleanup and research "
                    "analysis still use OPENAI_TEXT_MODEL. Without OPENAI_API_KEY those optional "
                    "post-processing stages will be skipped after reporting an error.",
                    file=sys.stderr,
                )

        speaker_labels: dict[str, str] = {}
        for value in args.speaker_label:
            raw, display = assignment(value, "--speaker-label")
            speaker_labels[raw] = display

        config = RunConfig(
            source=source,
            source_sha256=source_sha,
            source_duration_seconds=info.duration_seconds,
            model=model,
            language=language,
            speakers=tuple(speakers),
            exports=frozenset(exports),
            chunk_policy=ChunkPolicy(
                target_seconds=chunk_target,
                overlap_seconds=chunk_overlap,
            ),
            request_policy=RequestPolicy(
                timeout_seconds=args.request_timeout or 600,
                max_attempts=max_attempts,
                workers=args.workers or 1,
            ),
            text_policy=TextPolicy(
                transcript_style=style,
                analysis_model=analysis_model,
                analyze=args.analyze,
                glossary_path=_maybe_path(args.glossary),
                research_context_path=_maybe_path(args.research_context),
            ),
        )

        output = Path(args.output) / safe_name(source)
        if args.overwrite and output.exists():
            import shutil
            count = sum(1 for _ in output.rglob("*")) if output.exists() else 0
            print(f"Overwriting existing output: {output} ({count} files)")
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)

        client = create_client(timeout=config.request_policy.timeout_seconds)

        return run_pipeline(
            config,
            output,
            client,
            no_resume=args.no_resume,
            transcribe_only=args.transcribe_only,
            postprocess_only=args.postprocess_only,
            speaker_labels=speaker_labels,
            map_speakers=args.map_speakers,
            max_failed_chunks=args.max_failed_chunks,
            max_total_attempts=args.max_total_attempts,
        )

    except KeyboardInterrupt:
        print("\nCancelled. Completed chunks remain available for resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


def _build_speakers(speaker_args: list[str]) -> list[SpeakerReference]:
    from transcriber.speakers import validate_speaker_reference

    refs: list[SpeakerReference] = []
    for value in speaker_args:
        name, raw_path = assignment(value, "--speaker")
        ref = validate_speaker_reference(name, Path(raw_path).expanduser().resolve())
        refs.append(ref)
    if len(refs) > 4:
        raise ValueError("At most four known-speaker references are supported.")
    return refs


def _maybe_path(raw: str | None) -> Path | None:
    return Path(raw).expanduser().resolve() if raw else None


def _show_status(args) -> int:
    if not args.audio:
        print("Provide an audio path to check status.", file=sys.stderr)
        return 1

    source = Path(args.audio).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported or missing audio file: {source}")

    output = Path(args.output) / safe_name(source)
    manifest_path = output / "run_manifest.json"

    if not manifest_path.exists():
        print("No run found for this source.")
        return 0

    from transcriber.manifest import Manifest
    from transcriber.progress import status_summary

    try:
        m = Manifest.load(manifest_path)
    except Exception as exc:
        print(f"Could not read manifest: {exc}")
        return 1

    total = len(m.chunk_plan) if m.chunk_plan else len(m.chunks)
    completed = m.completed_chunk_count()
    processed = 0.0
    total_duration = m.source.get("duration_seconds", 0)
    if m.chunks:
        completed_ids = {cid for cid, rec in m.chunks.items() if rec.status == "completed"}
        for item in m.chunk_plan:
            if item["id"] in completed_ids:
                processed = max(processed, item.get("end_seconds", 0))

    core_status = "partial"
    if m.stages.get("transcription") and m.stages["transcription"].status == "completed":
        core_status = "complete"
    elif m.stages.get("rendering") and m.stages["rendering"].status == "completed":
        core_status = "complete"

    readable = "pending"
    if m.stages.get("readable"):
        readable = m.stages["readable"].status

    last_error = "none"
    for rec in m.chunks.values():
        if rec.attempts:
            last = rec.attempts[-1]
            if last.get("status") == "failed":
                last_error = last.get("error_type", "unknown")

    print(f"Run: {m.run_id}")
    print(status_summary(
        completed, max(total, 1), processed, total_duration,
        m.current_stage, core_status, readable, last_error,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())