from __future__ import annotations

import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

from transcriber.chunking import plan_chunks, extract_all_chunks
from transcriber.config import RunConfig
from transcriber.manifest import (
    ChunkStatus,
    StageStatus,
    Manifest,
    atomic_write_json,
    atomic_write_text,
    atomic_read_json,
    now_iso,
)
from transcriber.media import MediaInfo, guess_mime
from transcriber.progress import Progress
from transcriber.renderers import segments_from_response, transcript_text, subtitle
from transcriber.retry import is_retryable, error_name
from transcriber.speakers import rename_speakers


def to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if isinstance(response, dict):
        return response
    return json.loads(response.json())


def as_data_url(path: Path) -> str:
    mime = guess_mime(path)
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def transcribe_chunk_api(
    client: OpenAI,
    chunk_path: Path,
    model: str,
    language: str | None,
    speakers: tuple,
) -> dict[str, Any]:
    with chunk_path.open("rb") as audio:
        kwargs: dict[str, Any] = {"model": model, "file": audio}
        if language:
            kwargs["language"] = language
        if model == "gpt-4o-transcribe-diarize":
            kwargs["response_format"] = "diarized_json"
            kwargs["chunking_strategy"] = "auto"
            if speakers:
                kwargs["known_speaker_names"] = [s.name for s in speakers]
                kwargs["known_speaker_references"] = [as_data_url(s.path) for s in speakers]
        else:
            kwargs["response_format"] = "verbose_json"
            kwargs["timestamp_granularities"] = ["segment"]
        return to_dict(client.audio.transcriptions.create(**kwargs))


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _segments_overlap(a: dict[str, Any], b: dict[str, Any], threshold: float = 0.5) -> bool:
    a_start = float(a["start"])
    a_end = float(a["end"])
    b_start = float(b["start"])
    b_end = float(b["end"])
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    if overlap_end <= overlap_start:
        return False
    overlap_duration = overlap_end - overlap_start
    a_duration = a_end - a_start
    b_duration = b_end - b_start
    min_duration = min(a_duration, b_duration) if min(a_duration, b_duration) > 0 else 0.001
    return overlap_duration / min_duration >= threshold


def _fully_contained(inner: dict[str, Any], outer: dict[str, Any], eps: float = 1e-3) -> bool:
    return inner["start"] >= outer["start"] - eps and inner["end"] <= outer["end"] + eps


def deduplicate_overlaps(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(segments) < 2:
        return segments
    result: list[dict[str, Any]] = []
    for seg in segments:
        is_duplicate = False
        for kept in result:
            if _segments_overlap(seg, kept):
                same_speaker = seg.get("speaker") == kept.get("speaker")
                same_text = _normalize_text(seg["text"]) == _normalize_text(kept["text"])
                if same_speaker and (same_text or _fully_contained(seg, kept)):
                    is_duplicate = True
                    break
        if not is_duplicate:
            result.append(seg)
    return result


class ChunkTerminalFailure(RuntimeError):
    def __init__(self, records: list[dict[str, Any]], message: str):
        super().__init__(message)
        self.records = records


def _transcribe_one_chunk(
    client: OpenAI,
    config: RunConfig,
    chunk_id: str,
    chunk_path: Path,
    max_attempts: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    records: list[dict[str, Any]] = []
    for attempt_num in range(1, max_attempts + 1):
        started = now_iso()
        t0 = time.monotonic()
        try:
            payload = transcribe_chunk_api(
                client, chunk_path, config.model, config.language, config.speakers,
            )
            elapsed = time.monotonic() - t0
            records.append({
                "attempt": attempt_num,
                "started_at": started,
                "finished_at": now_iso(),
                "elapsed_seconds": round(elapsed, 2),
                "status": "succeeded",
                "request_id": None,
            })
            return payload, records, elapsed
        except Exception as exc:
            elapsed = time.monotonic() - t0
            retryable = is_retryable(exc)
            records.append({
                "attempt": attempt_num,
                "started_at": started,
                "finished_at": now_iso(),
                "elapsed_seconds": round(elapsed, 2),
                "status": "failed",
                "error_type": error_name(exc),
                "request_id": None,
                "retryable": retryable,
            })
            if not retryable or attempt_num == max_attempts:
                raise ChunkTerminalFailure(
                    records,
                    f"{chunk_id} failed after {attempt_num} attempt(s): {exc}",
                ) from exc
            delay = min(2 ** (attempt_num - 1), 8)
            print(f"  {chunk_id} failed (attempt {attempt_num}/{max_attempts}); retrying in {delay}s: {exc}")
            time.sleep(delay)
    raise ChunkTerminalFailure(records, f"{chunk_id} failed after {max_attempts} attempts")


def _acquire_output_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:
        return True, None
    lock_path = output_dir / ".run.lock"
    try:
        fd = lock_path.open("w")
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        if fd:
            fd.close()
        return False, None
    return True, fd


def _release_output_lock(lock_fd) -> None:
    if lock_fd is None:
        return
    try:
        import fcntl
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        lock_fd.close()


def run_pipeline(
    config: RunConfig,
    output: Path,
    client: OpenAI,
    no_resume: bool = False,
    transcribe_only: bool = False,
    postprocess_only: bool = False,
    speaker_labels: dict[str, str] | None = None,
    map_speakers: bool = False,
    max_failed_chunks: int | None = None,
    max_total_attempts: int | None = None,
) -> int:
    acquired, lock_fd = _acquire_output_lock(output)
    if not acquired:
        print("Another transcription run is already using this output directory.", file=sys.stderr)
        return 1
    try:
        try:
            return _run(
                config, output, client, no_resume=no_resume,
                transcribe_only=transcribe_only, postprocess_only=postprocess_only,
                speaker_labels=speaker_labels, map_speakers=map_speakers,
                max_failed_chunks=max_failed_chunks, max_total_attempts=max_total_attempts,
            )
        except KeyboardInterrupt:
            print("\nCancelled. Completed chunks remain available for resume.", file=sys.stderr)
            return 130
        except Exception as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            return 1
    finally:
        _release_output_lock(lock_fd)


def _run(
    config: RunConfig,
    output_dir: Path,
    client: OpenAI,
    no_resume: bool = False,
    transcribe_only: bool = False,
    postprocess_only: bool = False,
    speaker_labels: dict[str, str] | None = None,
    map_speakers: bool = False,
    max_failed_chunks: int | None = None,
    max_total_attempts: int | None = None,
) -> int:
    work_dir = output_dir / "working"
    audio_dir = work_dir / "audio"
    results_dir = work_dir / "results"
    manifest_path = output_dir / "run_manifest.json"
    run_fingerprint = config.fingerprint()

    should_make_new = True
    if not os.path.exists(manifest_path) or no_resume:
        should_make_new = True
    else:
        try:
            existing = Manifest.load(manifest_path)
            if existing.run_fingerprint == run_fingerprint:
                should_make_new = False
                print("Resuming previous run...")
            else:
                raise ValueError(
                    "Run fingerprint mismatch. Configuration or source changed. "
                    "Use --overwrite to start fresh."
                )
        except ValueError:
            raise

    if should_make_new:
        from transcriber.config import make_run_id

        run_id = make_run_id()
        source_dict = {
            "path": str(config.source),
            "sha256": config.source_sha256,
            "duration_seconds": config.source_duration_seconds,
        }
        config_dict = {
            "model": config.model,
            "language": config.language,
            "chunk_seconds": config.chunk_policy.target_seconds,
            "chunk_overlap_seconds": config.chunk_policy.overlap_seconds,
        }
        chunk_plan = plan_chunks(
            MediaInfo(
                duration_seconds=config.source_duration_seconds,
                size_bytes=config.source.stat().st_size,
                format_name="",
                audio_codec="",
                sample_rate=0,
                channels=0,
            ),
            config.source,
            config.chunk_policy,
        )
        manifest = Manifest(
            path=manifest_path,
            run_id=run_id,
            run_fingerprint=run_fingerprint,
            source=source_dict,
            configuration=config_dict,
            chunk_plan=chunk_plan,
        )
        manifest.set_current_stage("preflight")
    else:
        manifest = Manifest.load(manifest_path)
        manifest.verify_fingerprint(run_fingerprint)

    manifest.set_stage("transcription", StageStatus.RUNNING)
    manifest.set_current_stage("prepare")
    manifest.save()

    if not postprocess_only:
        manifest.set_current_stage("prepare")
        extract_all_chunks(manifest.chunk_plan, config.source, audio_dir, config.chunk_policy)
        for item in manifest.chunk_plan:
            record = manifest.chunks.get(item["id"])
            if record is None or record.status != ChunkStatus.COMPLETED.value:
                manifest.set_chunk_status(item["id"], ChunkStatus.PREPARED.value, raw_result=None)
        manifest.save()

        manifest.set_current_stage("transcribe")
        total_chunks = len(manifest.chunk_plan)
        progress = Progress(total_chunks, config.source_duration_seconds)
        all_segments: list[dict[str, Any]] = []

        results_dir.mkdir(parents=True, exist_ok=True)

        pending: list[tuple[int, dict[str, Any], Path]] = []
        for index, item in enumerate(manifest.chunk_plan, 1):
            chunk_id = item["id"]
            chunk_record = manifest.chunks.get(chunk_id)
            result_path = results_dir / f"{chunk_id}.json"

            if chunk_record and chunk_record.status == ChunkStatus.COMPLETED.value:
                if manifest.verify_raw_result(chunk_id, output_dir):
                    print(f"[{index}/{total_chunks}] Reusing {chunk_id}")
                    payload = atomic_read_json(result_path)
                    all_segments.extend(
                        segments_from_response(
                            payload,
                            item["start_seconds"],
                            item["end_seconds"] - item["start_seconds"],
                        )
                    )
                    progress.chunk_done(chunk_id, item["end_seconds"], 0.0)
                    continue

            chunk_path = Path(item["path"]) if item.get("path") else None
            if chunk_path is None or not chunk_path.exists():
                raise RuntimeError(
                    f"Chunk audio missing for {chunk_id}: {chunk_path}. "
                    f"Rerun without --postprocess-only to re-extract."
                )

            print(f"[{index}/{total_chunks}] Transcribing {chunk_id}")
            manifest.set_chunk_status(chunk_id, ChunkStatus.PROCESSING.value)
            manifest.save()
            pending.append((index, item, chunk_path))

        max_attempts = config.request_policy.max_attempts
        failed_limit = max_failed_chunks if max_failed_chunks is not None else 1
        attempt_budget = max_total_attempts
        total_attempts_used = 0
        failed_chunks = 0

        if pending:
            workers = max(1, config.request_policy.workers)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_transcribe_one_chunk, client, config, item["id"], chunk_path, max_attempts): (index, item)
                    for index, item, chunk_path in pending
                }
                try:
                    for future in as_completed(futures):
                        index, item = futures[future]
                        chunk_id = item["id"]
                        result_path = results_dir / f"{chunk_id}.json"
                        try:
                            payload, records, elapsed = future.result()
                        except ChunkTerminalFailure as failure:
                            for record in failure.records[:-1]:
                                manifest.add_attempt(chunk_id, record)
                            manifest.mark_failed(chunk_id, True, failure.records[-1])
                            manifest.save()
                            failed_chunks += 1
                            total_attempts_used += len(failure.records)
                            print(f"  {chunk_id} failed terminally: {failure}")
                            _cancel_futures(futures)
                            if attempt_budget is not None and total_attempts_used >= attempt_budget:
                                raise RuntimeError(
                                    f"Stopped: total attempt budget of {attempt_budget} exhausted."
                                )
                            if failed_chunks >= failed_limit:
                                raise RuntimeError(
                                    f"Stopped after {failed_chunks} terminal chunk failure(s): {failure}"
                                )
                            continue

                        for record in records:
                            manifest.add_attempt(chunk_id, record)
                        total_attempts_used += len(records)
                        if attempt_budget is not None and total_attempts_used >= attempt_budget:
                            _cancel_futures(futures)
                            manifest.save()
                            raise RuntimeError(
                                f"Stopped: total attempt budget of {attempt_budget} exhausted."
                            )
                        atomic_write_json(result_path, payload)
                        manifest.mark_completed(
                            chunk_id,
                            str(result_path.relative_to(output_dir)),
                            len(payload.get("segments", [])) if payload else 0,
                        )
                        manifest.save()

                        chunk_segments = segments_from_response(
                            payload,
                            item["start_seconds"],
                            item["end_seconds"] - item["start_seconds"],
                        )
                        all_segments.extend(chunk_segments)

                        progress.chunk_done(chunk_id, item["end_seconds"], elapsed)
                        print(progress.format_line(chunk_id))
                        print(f"Saved: provisional transcript available.\n")

                        if should_save_provisional(total_chunks, index):
                            _save_provisional(output_dir, all_segments, config)
                except KeyboardInterrupt:
                    _cancel_futures(futures)
                    raise

        all_segments.sort(key=lambda s: (s["start"], s["end"]))
        all_segments = deduplicate_overlaps(all_segments)

        manifest.set_current_stage("merge")
        manifest.set_stage("transcription", StageStatus.COMPLETED)
        manifest.save()
    else:
        all_segments = _load_existing_segments(output_dir)

    labels: dict[str, str] = {}
    for speaker in config.speakers:
        labels.setdefault(speaker.name, speaker.name)
    if speaker_labels:
        labels.update(speaker_labels)
    if map_speakers or speaker_labels:
        labels = rename_speakers(all_segments, labels, interactive=bool(map_speakers))

    _finalize_transcript_outputs(output_dir, config, all_segments, labels, manifest)

    if transcribe_only:
        print(f"\nDone. Outputs: {output_dir}")
        return 0

    print("\nTranscription completed.\nCore outputs are ready.\n")

    verbatim = transcript_text(all_segments)

    if config.text_policy.transcript_style in {"readable", "both"} and not postprocess_only:
        print("Starting readable transcript stage...")
        manifest.set_stage("readable", StageStatus.RUNNING)
        manifest.set_current_stage("readable")
        manifest.save()
        try:
            readable = _text_model_call(
                client, config.text_policy.analysis_model,
                _readable_prompt(verbatim, config.text_policy.glossary_path),
            )
            atomic_write_text(output_dir / "readable_transcript.txt", readable + "\n")
            manifest.set_stage("readable", StageStatus.COMPLETED)
        except Exception as exc:
            manifest.set_stage("readable", StageStatus.FAILED)
            print(f"Readable transcript failed.\n"
                  f"The verbatim transcript, JSON, SRT, and VTT remain available.\n"
                  f"Rerun with --postprocess-only to continue.\n"
                  f"Error: {exc}")
        manifest.save()

    if config.text_policy.glossary_path and not postprocess_only:
        print("Starting glossary review...")
        manifest.set_stage("glossary", StageStatus.RUNNING)
        manifest.set_current_stage("glossary")
        manifest.save()
        try:
            glossary = config.text_policy.glossary_path.read_text(encoding="utf-8")
            review_prompt = _review_prompt(verbatim, glossary)
            review_raw = _text_model_call(client, config.text_policy.analysis_model, review_prompt)
            review_raw = review_raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                atomic_write_json(output_dir / "review_flags.json", json.loads(review_raw))
            except json.JSONDecodeError:
                atomic_write_json(output_dir / "review_flags.json", {"items": [], "unparsed_model_output": review_raw})
            manifest.set_stage("glossary", StageStatus.COMPLETED)
        except Exception as exc:
            manifest.set_stage("glossary", StageStatus.FAILED)
            print(f"Glossary review failed: {exc}")
        manifest.save()

    if config.text_policy.analyze and not postprocess_only:
        print("Starting research analysis...")
        manifest.set_stage("analysis", StageStatus.RUNNING)
        manifest.set_current_stage("analysis")
        manifest.save()
        try:
            context = {}
            if config.text_policy.research_context_path:
                context = yaml.safe_load(config.text_policy.research_context_path.read_text(encoding="utf-8")) or {}
            analysis_prompt = _analysis_prompt(verbatim, context)
            analysis = _text_model_call(client, config.text_policy.analysis_model, analysis_prompt)
            research_dir = output_dir / "research"
            research_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(research_dir / "research_analysis.md", analysis + "\n")
            manifest.set_stage("analysis", StageStatus.COMPLETED)
        except Exception as exc:
            manifest.set_stage("analysis", StageStatus.FAILED)
            print(f"Research analysis failed: {exc}")
        manifest.save()

    manifest.status = "completed"
    manifest.save()
    print(f"\nDone. Outputs: {output_dir}")
    return 0


def _save_provisional(output_dir: Path, segments: list[dict[str, Any]], config: RunConfig) -> None:
    atomic_write_json(output_dir / "raw_transcript.partial.json", {"segments": segments})
    if "txt" in config.exports and config.text_policy.transcript_style in {"verbatim", "both"}:
        atomic_write_text(output_dir / "verbatim_transcript.partial.txt", transcript_text(segments))
    if "srt" in config.exports:
        atomic_write_text(output_dir / "transcript.partial.srt", subtitle(segments))
    if "vtt" in config.exports:
        atomic_write_text(output_dir / "transcript.partial.vtt", subtitle(segments, True))


def should_save_provisional(total_chunks: int, current_index: int) -> bool:
    if total_chunks == 1:
        return False
    if current_index == total_chunks:
        return False
    return True


def _cancel_futures(futures: dict) -> None:
    for future in futures:
        if not future.done():
            future.cancel()


def _finalize_transcript_outputs(
    output_dir: Path,
    config: RunConfig,
    all_segments: list[dict[str, Any]],
    labels: dict[str, str],
    manifest: Manifest,
) -> None:
    manifest.set_stage("rendering", StageStatus.RUNNING)
    manifest.set_current_stage("rendering")
    manifest.save()

    raw = {
        "source": str(config.source),
        "source_sha256": config.source_sha256,
        "model": config.model,
        "speaker_label_map": labels,
        "segments": all_segments,
    }
    atomic_write_json(output_dir / "raw_transcript.json", raw)

    if "json" in config.exports:
        atomic_write_json(output_dir / "transcript.json", {"segments": all_segments})

    verbatim = transcript_text(all_segments)
    if "txt" in config.exports and config.text_policy.transcript_style in {"verbatim", "both"}:
        atomic_write_text(output_dir / "verbatim_transcript.txt", verbatim)

    if "srt" in config.exports:
        atomic_write_text(output_dir / "transcript.srt", subtitle(all_segments))
    if "vtt" in config.exports:
        atomic_write_text(output_dir / "transcript.vtt", subtitle(all_segments, True))

    for ext in [".partial.json", ".partial.txt", ".partial.srt", ".partial.vtt"]:
        partial = output_dir / f"raw_transcript{ext}" if ".partial.json" in ext else output_dir / f"verbatim_transcript{ext}" if ".partial.txt" in ext else output_dir / f"transcript{ext}"
        if os.path.exists(partial):
            try:
                os.remove(partial)
            except OSError:
                pass

    manifest.set_stage("rendering", StageStatus.COMPLETED)
    manifest.save()


def _load_existing_segments(output_dir: Path) -> list[dict[str, Any]]:
    raw_path = output_dir / "raw_transcript.json"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"No raw_transcript.json found in {output_dir}; nothing to post-process. "
            f"Run the full transcription first."
        )
    data = atomic_read_json(raw_path)
    return data.get("segments", [])


def _text_model_call(client: OpenAI, model: str, prompt: str, max_attempts: int = 2) -> str:
    for attempt_num in range(1, max_attempts + 1):
        try:
            response = client.responses.create(model=model, input=prompt)
            output = getattr(response, "output_text", None)
            if output:
                return output.strip()
            return str(to_dict(response))
        except Exception as exc:
            if not is_retryable(exc) or attempt_num == max_attempts:
                raise
            delay = min(2 ** (attempt_num - 1), 8)
            time.sleep(delay)
    return ""


def _readable_prompt(verbatim: str, glossary_path: Path | None) -> str:
    glossary = glossary_path.read_text(encoding="utf-8") if glossary_path else "(none)"
    return (
        f"Create a readable copy of this interview transcript. Preserve every timestamp, "
        f"speaker label, claim, number, uncertainty, and language choice. Fix punctuation "
        f"and obvious formatting only. Do not summarize or invent. Use [VERIFY] when a "
        f"glossary correction is uncertain.\n\n"
        f"Approved glossary:\n{glossary}\n\n"
        f"Transcript:\n{verbatim}"
    )


def _review_prompt(verbatim: str, glossary: str) -> str:
    return (
        f"Review the transcript against the glossary. Return valid JSON only with an "
        f"items array. Each item must have timestamp, speaker, excerpt, and reason. "
        f"Flag suspicious names or technical terms; do not invent acoustic confidence.\n\n"
        f"Glossary:\n{glossary}\n\n"
        f"Transcript:\n{verbatim}"
    )


def _analysis_prompt(verbatim: str, context: dict) -> str:
    return (
        "Analyze this research interview using only the transcript. Never invent quotations "
        "or findings. Keep speaker names and timestamps with every quotation. Clearly "
        "distinguish evidence from interpretation. Return Markdown with exactly these headings: "
        "# Interview Summary, # Themes and Subthemes, # Key Quotations, # Question-and-Answer Map, "
        f"# Follow-up Questions.\n\nResearch context:\n"
        f"{yaml.safe_dump(context, allow_unicode=True, sort_keys=False)}\n\n"
        f"Transcript:\n{verbatim}"
    )
