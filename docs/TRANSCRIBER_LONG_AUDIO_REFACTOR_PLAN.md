# Long-Audio Transcriber Refactor Plan

**Repository:** `niks0501/whisper-transcriber`  
**Target:** `main` after the interview-transcriber enhancement  
**Primary use case:** Two-hour or longer thesis interviews with speaker diarization  
**Status:** Implementation-ready plan

---

## 1. Objective

Refactor the transcriber so long recordings are:

- split according to duration as well as file size;
- processed in small, recoverable API requests;
- protected against duplicate retries and unnecessary charges;
- written to visible provisional outputs after every completed chunk;
- safely resumable after interruption;
- observable through progress, elapsed time, processed duration, and failures;
- post-processed separately so transcription results are available before cleanup or research analysis finishes.

The existing user-facing workflows must remain valid:

```bash
python3 transcribe.py "recordings/SRA-Interview.m4a"
```

```bash
python3 transcribe.py --wizard
```

```bash
python3 transcribe.py "recordings/SRA-Interview.m4a" \
  --profile interview \
  --language auto \
  --speaker "Interviewer=speaker_references/nikko.m4a" \
  --speaker "SRA Staff=speaker_references/sra-staff.m4a" \
  --glossary "config/sugarcane-glossary.txt" \
  --transcript-style both \
  --export txt,json,srt,vtt
```

---

## 2. Current Design Problems

### 2.1 Size-only chunking

The current `prepare_chunks()` sends the original recording directly when it is below the file-size threshold, regardless of its duration.

A highly compressed two-hour M4A can therefore be treated as one API request.

### 2.2 Oversized 15-minute chunks

When splitting does occur, the current implementation uses 900-second chunks. These requests are slow, difficult to recover, and can produce large transcription responses.

### 2.3 Stacked retry behavior

The application has its own retry loop while the default OpenAI client can also retry requests internally.

This makes the true number of attempts unclear and can cause:

- repeated processing;
- repeated charges;
- long silent waits;
- difficulty identifying which request failed.

### 2.4 User-facing outputs are delayed

A chunk response is saved internally, but the normal transcript, JSON, SRT, and VTT files are created only after all transcription requests finish.

A user can therefore see API usage while the output directory still appears empty or incomplete.

### 2.5 One function controls the whole workflow

The current `main()` handles:

- CLI parsing;
- validation;
- media probing;
- chunk preparation;
- API client creation;
- retries;
- transcription;
- resume;
- segment merging;
- speaker mapping;
- rendering;
- readable cleanup;
- glossary review;
- research analysis.

This makes failures hard to isolate and behavior hard to test.

### 2.6 Resume validation is incomplete

Completed chunks are reused primarily from the source checksum and chunk status.

Changing any of these should invalidate or separate a prior run:

- model;
- language;
- speaker-reference files;
- chunk duration;
- chunk overlap;
- audio encoding policy;
- diarization mode;
- application schema version.

### 2.7 Full-transcript post-processing

Readable cleanup, glossary review, and research analysis operate on the complete merged transcript.

For a multi-hour interview, this can:

- take a long time;
- exceed a model's practical input/output limits;
- fail after transcription has already succeeded;
- make the entire command appear unfinished;
- create extra charges without intermediate outputs.

---

## 3. Target Architecture

Replace the single-file implementation with a thin compatibility entrypoint and a small internal package.

```text
whisper-transcriber/
├── transcribe.py
├── transcriber/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── media.py
│   ├── chunking.py
│   ├── api_client.py
│   ├── retry.py
│   ├── manifest.py
│   ├── pipeline.py
│   ├── progress.py
│   ├── speakers.py
│   ├── renderers.py
│   └── postprocess.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── docs/
    └── long-audio.md
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `transcribe.py` | Backward-compatible launcher only |
| `cli.py` | Arguments, wizard, validation messages |
| `config.py` | Typed runtime configuration and defaults |
| `media.py` | FFprobe metadata and FFmpeg execution |
| `chunking.py` | Chunk plan creation, extraction, overlap |
| `api_client.py` | Explicit OpenAI client configuration |
| `retry.py` | Single retry policy and retry classification |
| `manifest.py` | Durable run state and atomic updates |
| `pipeline.py` | Stage orchestration and resume behavior |
| `progress.py` | Console progress, elapsed time, ETA |
| `speakers.py` | Reference validation and label mapping |
| `renderers.py` | TXT, JSON, SRT, and VTT generation |
| `postprocess.py` | Readable copy, glossary review, analysis |

Keep functions simple and explicit. Avoid framework-style abstractions or dependency injection machinery that would make the project harder for a student developer to understand.

---

## 4. Runtime Pipeline

The new workflow should behave as a durable state machine.

```text
PREFLIGHT
   ↓
PLAN_CHUNKS
   ↓
PREPARE_CHUNKS
   ↓
TRANSCRIBE_CHUNKS
   ↓
MERGE_AND_RENDER
   ↓
OPTIONAL_READABLE_COPY
   ↓
OPTIONAL_GLOSSARY_REVIEW
   ↓
OPTIONAL_RESEARCH_ANALYSIS
   ↓
COMPLETED
```

Each stage writes its state to `run_manifest.json`.

A failure in a later stage must not invalidate a completed transcription.

Example:

```text
Transcription: completed
Readable copy: failed
Glossary review: pending
Research analysis: not requested
```

The user must still receive the verbatim transcript and subtitles.

---

## 5. Phase 1 — Foundation and Safe Configuration

### Goal

Extract the current behavior into testable modules without changing the basic command.

### Tasks

#### 5.1 Add typed configuration

Create dataclasses such as:

```python
@dataclass(frozen=True)
class ChunkPolicy:
    target_seconds: int = 300
    overlap_seconds: int = 2
    max_upload_mb: float = 20.0
    sample_rate: int = 16000
    channels: int = 1
    audio_bitrate: str = "64k"
```

```python
@dataclass(frozen=True)
class RequestPolicy:
    timeout_seconds: int = 600
    max_attempts: int = 2
    workers: int = 1
```

```python
@dataclass(frozen=True)
class RunConfig:
    source: Path
    model: str
    language: str | None
    speakers: tuple[SpeakerReference, ...]
    chunk_policy: ChunkPolicy
    request_policy: RequestPolicy
    exports: frozenset[str]
    transcript_style: str
```

#### 5.2 Preserve CLI compatibility

Keep these flags working:

- positional audio path;
- `--wizard`;
- `--profile`;
- `--model`;
- `--language`;
- `--speaker`;
- `--speaker-label`;
- `--map-speakers`;
- `--glossary`;
- `--transcript-style`;
- `--export`;
- `--analyze`;
- `--research-context`;
- `--output`;
- `--overwrite`;
- `--no-resume`;
- `--retries`.

Add a deprecation path:

```text
--retries N
```

maps to:

```text
--max-attempts N
```

Document clearly whether the number includes the first attempt. Recommended meaning:

```text
--max-attempts 2 = one initial request plus one retry
```

#### 5.3 Add advanced long-audio options

```text
--chunk-seconds 300
--chunk-overlap-seconds 2
--request-timeout 600
--max-attempts 2
--workers 1
--transcribe-only
--postprocess-only
```

These options should not be required for normal use.

#### 5.4 Add run fingerprints

Generate a deterministic hash from:

- source SHA-256;
- model;
- language;
- speaker names and reference-file SHA-256 values;
- chunk policy;
- request-relevant settings;
- manifest schema version.

Example:

```json
{
  "schema_version": 2,
  "run_fingerprint": "sha256:...",
  "source_sha256": "sha256:...",
  "configuration": {
    "model": "gpt-4o-transcribe-diarize",
    "language": null,
    "chunk_seconds": 300,
    "chunk_overlap_seconds": 2
  }
}
```

A mismatched fingerprint must never silently reuse prior chunk responses.

### Phase 1 acceptance criteria

- Existing commands still parse.
- Configuration can be tested without API calls.
- Changing a speaker reference or model changes the run fingerprint.
- `transcribe.py` becomes a thin entrypoint.
- No transcription behavior is lost.

---

## 6. Phase 2 — Long-Audio Chunking and Request Control

### Goal

Guarantee that a two-hour recording is never sent as one giant request and that every API attempt is controlled by one retry layer.

### 6.1 Probe metadata before processing

Use FFprobe once and record:

```json
{
  "duration_seconds": 7564.2,
  "size_bytes": 18392044,
  "format": "mov,mp4,m4a,3gp,3g2,mj2",
  "audio_codec": "aac",
  "sample_rate": 44100,
  "channels": 2
}
```

Print a preflight summary:

```text
Source: recordings/SRA-Interview.m4a
Duration: 02:06:04
Size: 17.54 MB
Mode: long-audio
Chunk target: 5 minutes
Estimated chunks: 26
```

### 6.2 Trigger chunking by duration and size

Recommended policy:

```text
Use original source only when:
- duration <= 300 seconds; and
- size <= 20 MB.

Otherwise:
- create speech-optimized chunks.
```

The decision must use both duration and file size.

### 6.3 Use five-minute chunks

Recommended default:

```text
target_seconds = 300
overlap_seconds = 2
```

A two-hour recording produces approximately 24–26 chunks.

The chunk manifest must store actual start and end times rather than calculating offsets only from the chunk index.

```json
{
  "id": "chunk-0007",
  "start_seconds": 2098.0,
  "end_seconds": 2402.0,
  "overlap_before_seconds": 2.0,
  "path": "working/audio/chunk-0007.mp3",
  "sha256": "..."
}
```

### 6.4 Create chunks with explicit intervals

Prefer one extraction command per planned interval:

```bash
ffmpeg -y \
  -ss START \
  -i SOURCE \
  -t DURATION \
  -vn \
  -ac 1 \
  -ar 16000 \
  -b:a 64k \
  OUTPUT.mp3
```

Reasons:

- offsets are explicit;
- the final chunk duration is accurate;
- chunks can be recreated individually;
- overlap is easy to represent;
- resume does not require rebuilding every chunk.

### 6.5 Handle boundary overlap

Use a small overlap to avoid losing a word cut at the exact boundary.

After transcription:

- convert local chunk timestamps to global timestamps;
- compare overlapping segments;
- remove duplicate segments using time intersection and normalized text similarity;
- never delete both copies when similarity is uncertain;
- record deduplication decisions in debug logs.

Start with a conservative deduplication rule:

```text
Treat two segments as duplicates only when:
- timestamps overlap substantially; and
- normalized text is equal or nearly equal.
```

### 6.6 Configure one retry layer

Create the client explicitly:

```python
client = OpenAI(
    max_retries=0,
    timeout=request_timeout,
)
```

The application owns retries.

Retry only failures considered transient, such as:

- connection errors;
- request timeout;
- HTTP 408;
- HTTP 409;
- HTTP 429;
- HTTP 5xx.

Do not retry:

- invalid speaker references;
- unsupported files;
- authentication failures;
- malformed request parameters;
- other deterministic HTTP 4xx failures.

Record each attempt:

```json
{
  "attempt": 1,
  "started_at": "...",
  "finished_at": "...",
  "elapsed_seconds": 84.2,
  "status": "failed",
  "error_type": "APITimeoutError",
  "request_id": null,
  "retryable": true
}
```

### 6.7 Protect against unbounded spending

Add optional controls:

```text
--max-failed-chunks 1
--max-total-attempts 30
```

Stop the run if the limit is exceeded and preserve completed work.

### Phase 2 acceptance criteria

- Any audio longer than five minutes is locally chunked.
- A two-hour file is not uploaded as one request.
- No chunk exceeds the configured duration or upload limit.
- One chunk can produce no more than `max_attempts` API calls.
- SDK retries are disabled.
- Request IDs and attempt duration are recorded when available.
- Completed chunks are not retranscribed after restart.

---

## 7. Phase 3 — Durable Pipeline and Incremental Results

### Goal

Make useful results visible after the first successful chunk and make interruption safe.

### 7.1 Expand the manifest

Recommended structure:

```json
{
  "schema_version": 2,
  "run_id": "20260801T031500Z-a3d91c2f",
  "run_fingerprint": "...",
  "status": "running",
  "current_stage": "transcribe",
  "source": {},
  "configuration": {},
  "chunk_plan": [],
  "chunks": {
    "chunk-0000": {
      "status": "completed",
      "attempts": [],
      "raw_result": "working/results/chunk-0000.json",
      "segment_count": 48
    }
  },
  "stages": {
    "transcription": {"status": "running"},
    "rendering": {"status": "running"},
    "readable": {"status": "pending"},
    "glossary": {"status": "pending"},
    "analysis": {"status": "not_requested"}
  }
}
```

### 7.2 Use explicit chunk states

```text
planned
prepared
uploading
processing
completed
failed_retryable
failed_terminal
```

Write the manifest atomically before and after every network attempt.

### 7.3 Save provisional outputs after every chunk

After a successful chunk:

1. save raw chunk response;
2. update the manifest;
3. rebuild merged segments from all completed chunks;
4. write provisional outputs atomically.

Expected files while still running:

```text
transcription_output/SRA-Interview/
├── run_manifest.json
├── progress.json
├── raw_transcript.partial.json
├── verbatim_transcript.partial.txt
├── transcript.partial.srt
├── transcript.partial.vtt
└── working/
    ├── audio/
    └── results/
```

When transcription completes:

```text
raw_transcript.json
verbatim_transcript.txt
transcript.srt
transcript.vtt
```

Do not make the user wait for readable cleanup before finalizing the verbatim artifacts.

### 7.4 Display meaningful progress

Example:

```text
[07/26] Completed chunk-0006
Audio processed: 00:35:02 / 02:06:04 (27.8%)
Chunk time: 01:24
Elapsed: 10:51
Estimated transcription remaining: 29:31
Saved provisional transcript: verbatim_transcript.partial.txt
```

ETA should be described as an estimate and calculated from completed chunk averages.

### 7.5 Correct resume behavior

On rerun:

- load the manifest;
- verify fingerprint;
- verify raw result files exist and parse;
- reuse only valid completed chunks;
- restart the first incomplete or corrupt chunk;
- regenerate provisional outputs from completed chunks;
- never require `--resume` for the normal safe behavior.

`--no-resume` should start a new run rather than quietly mixing old and new results.

### 7.6 Separate transcription completion from post-processing

The terminal should clearly announce:

```text
Transcription completed.
Core outputs are ready.

Starting readable transcript stage...
```

If post-processing fails:

```text
Readable transcript failed.
The verbatim transcript, JSON, SRT, and VTT remain available.
Rerun with --postprocess-only to continue.
```

### Phase 3 acceptance criteria

- A partial TXT and JSON file appear after the first completed chunk.
- `Ctrl+C` preserves all completed chunks.
- Restarting reuses completed chunks without API calls.
- A corrupted chunk result is detected and retranscribed.
- Transcription outputs are finalized before optional AI post-processing.
- The manifest correctly reports partial failure states.

---

## 8. Phase 4 — Scalable Post-Processing, UX, and Rollout

### Goal

Prevent readable cleanup, glossary review, and analysis from becoming another long single request.

### 8.1 Batch readable cleanup

Split the verbatim transcript by timestamped segments, not raw character slicing.

Recommended batch target:

```text
10–15 minutes of transcript per batch
```

For each batch:

- preserve speaker labels and timestamps;
- save `working/readable/part-000.json`;
- resume completed parts;
- merge in timestamp order.

### 8.2 Batch glossary review

Review the same transcript batches and merge findings.

Deduplicate flags using:

- timestamp;
- speaker;
- normalized excerpt;
- reason category.

### 8.3 Use map-reduce for research analysis

For long interviews:

1. generate evidence notes per batch;
2. save all batch notes;
3. synthesize final themes from those notes;
4. retain links to original timestamps and speakers;
5. reject quotations not found verbatim in the transcript.

Expected files:

```text
research/
├── parts/
│   ├── part-000-notes.md
│   └── part-001-notes.md
├── research_analysis.md
└── analysis_manifest.json
```

### 8.4 Make expensive stages explicit

Recommended UX decision:

- basic transcription defaults to `verbatim`;
- `--transcript-style both` explicitly enables readable cleanup;
- `--glossary` explicitly enables glossary review;
- `--analyze` explicitly enables research analysis.

This avoids unexpected text-model charges from the simplest command.

If backward compatibility is preferred, retain `both` temporarily but print:

```text
Readable cleanup is an additional API stage.
Use --transcript-style verbatim for transcription-only output.
```

### 8.5 Add a status command

Recommended syntax:

```bash
python3 transcribe.py status "recordings/SRA-Interview.m4a"
```

Example output:

```text
Run: 20260801T031500Z-a3d91c2f
Transcription: 17/26 chunks completed
Processed: 01:25:01 / 02:06:04
Current stage: transcribe
Last error: none
Core transcript: partial
Readable copy: pending
```

For compatibility, status can initially be implemented as:

```bash
python3 transcribe.py "recordings/SRA-Interview.m4a" --status
```

### 8.6 Add bounded concurrency only after correctness

Implement optional concurrency after sequential processing is proven stable.

```text
--workers 1   safe default
--workers 2   faster optional mode
```

Rules:

- cap workers at a small number;
- write each chunk independently;
- serialize manifest updates;
- stop scheduling new chunks after a terminal failure;
- never allow two workers to process the same chunk;
- preserve output ordering by global timestamps.

Do not make concurrency the first optimization. Reliable chunking and incremental persistence are more important.

### 8.7 Update the wizard

For recordings longer than five minutes, display:

```text
Long recording detected: 02:06:04

Recommended settings:
- 5-minute chunks
- 2-second boundary overlap
- 2 maximum attempts per chunk
- 1 worker
- resumable checkpoints
```

Wizard questions:

```text
Use safe long-audio defaults? [Y/n]
Generate a readable copy after transcription? [y/N]
Run glossary review? [y/N]
Run thesis analysis? [y/N]
```

### 8.8 Update README and add long-audio documentation

README changes:

- explain that long recordings are locally split;
- show where provisional files appear;
- explain resume behavior;
- explain why API usage may appear before final completion;
- distinguish transcription costs from cleanup/analysis costs;
- document `--postprocess-only`;
- document `status`;
- document timeout and retry settings;
- show WSL2 examples.

Add:

```text
docs/long-audio.md
```

with troubleshooting commands for:

- viewing the manifest;
- counting completed chunks;
- checking the current process;
- safely cancelling;
- resuming;
- rerunning post-processing only.

### Phase 4 acceptance criteria

- A two-hour transcript is never passed to one cleanup request.
- Failed post-processing can resume without retranscribing audio.
- Basic usage does not unexpectedly run research analysis.
- The wizard explains long-audio behavior before starting.
- Optional two-worker mode produces the same ordered transcript as one-worker mode.
- README contains a complete WSL2 long-audio manual.

---

## 9. Testing Strategy

### 9.1 Unit tests

Add tests for:

- duration-and-size chunk decision;
- chunk interval generation;
- final short chunk;
- overlap boundaries;
- run fingerprint stability;
- run fingerprint change after configuration change;
- retryable versus non-retryable errors;
- maximum attempt enforcement;
- manifest atomic writes;
- corrupt result detection;
- global timestamp conversion;
- overlap deduplication;
- progress percentage and ETA;
- transcript, SRT, and VTT rendering.

### 9.2 API-free integration tests

Use a fake transcription client that can:

- return normal diarized responses;
- fail once then succeed;
- timeout repeatedly;
- return a deterministic 400 error;
- simulate request IDs;
- record how many times each chunk was requested.

Required scenarios:

#### Scenario A — normal long recording

- 20-minute synthetic input;
- four chunks;
- four API calls;
- provisional output after chunk one;
- final output after chunk four.

#### Scenario B — interruption

- stop after chunk two;
- rerun;
- chunks one and two are reused;
- only chunks three and four call the fake API.

#### Scenario C — retry

- chunk two times out once;
- chunk two is attempted exactly twice;
- no other chunk is repeated.

#### Scenario D — configuration mismatch

- first run uses English;
- second run uses automatic language;
- prior results are not silently reused.

#### Scenario E — post-processing failure

- transcription completes;
- readable stage fails;
- final verbatim files still exist;
- `--postprocess-only` resumes cleanup.

### 9.3 FFmpeg integration tests

Generate short synthetic audio in CI using FFmpeg.

Verify:

- correct number of chunks;
- each chunk duration is within tolerance;
- final chunk is not empty;
- chunks use mono 16 kHz audio;
- output paths are deterministic.

### 9.4 Manual validation matrix

| Recording | Purpose |
|---|---|
| 2–3 minutes | No-chunk baseline |
| 12 minutes | Boundary and overlap |
| 35 minutes | Resume and progress |
| 2+ hours | Thesis production gate |
| Noisy interview | Speaker reference validation |
| Taglish interview | Automatic-language path |

Do not use a sensitive thesis recording in public CI.

---

## 10. Performance and Cost Gates

The refactor is ready for production use only when all gates pass.

### Correctness gates

- No missing or duplicated timestamp ranges at chunk boundaries.
- Speaker references are passed consistently to every diarized chunk.
- Completed chunks are never repeated during normal resume.
- Final transcript ordering is deterministic.

### Reliability gates

- A simulated timeout cannot cause more attempts than configured.
- Every completed chunk has a durable raw result before moving forward.
- `Ctrl+C` leaves a valid manifest.
- Failed post-processing does not damage the raw transcript.

### Visibility gates

- First provisional transcript appears after the first chunk.
- Progress reports processed audio duration and elapsed time.
- The current stage is visible in both terminal output and manifest.

### Cost-control gates

- SDK automatic retries are disabled.
- Application attempts are counted explicitly.
- Post-processing stages are separately identified.
- Resume avoids API calls for completed chunks.
- Basic command behavior regarding readable cleanup is documented and intentional.

### Long-audio gate

A real or sanitized two-hour recording must complete with:

- expected chunk count;
- no single two-hour upload;
- no repeated completed chunks;
- useful partial output during processing;
- successful restart after one forced interruption.

---

## 11. Migration and Backward Compatibility

### Existing output

Before changing the manifest schema:

- detect schema version 1;
- preserve old files;
- offer to continue in compatibility mode or create a new run;
- never delete an old run unless `--overwrite` is explicit.

### Existing entrypoint

Keep:

```bash
python3 transcribe_auto_split.py ...
```

as a compatibility wrapper that calls the same new CLI and prints a deprecation notice.

### Existing commands

The current advanced command must continue working unchanged.

New safe defaults should be introduced through configuration, not through mandatory additional arguments.

### Recommended output layout

```text
transcription_output/
└── SRA-Interview/
    ├── latest_run.json
    └── runs/
        └── 20260801T031500Z-a3d91c2f/
            ├── run_manifest.json
            ├── raw_transcript.json
            ├── verbatim_transcript.txt
            ├── readable_transcript.txt
            ├── transcript.srt
            ├── transcript.vtt
            ├── review_flags.json
            ├── research/
            └── working/
```

This prevents incompatible runs from overwriting or reusing each other.

---

## 12. Recommended Pull Request Breakdown

Avoid one massive refactor PR.

### PR 1 — Extract configuration and manifest

- package structure;
- dataclasses;
- run fingerprints;
- manifest schema v2;
- compatibility tests.

### PR 2 — Long-audio chunk engine

- FFprobe metadata;
- duration-aware chunking;
- five-minute chunks;
- overlap;
- chunk-plan tests.

### PR 3 — Request control and incremental outputs

- explicit OpenAI timeout and retry policy;
- request attempt records;
- provisional transcript generation;
- progress and resume tests.

### PR 4 — Batched post-processing and documentation

- readable batches;
- glossary batches;
- map-reduce research analysis;
- status/postprocess commands;
- wizard and README updates;
- two-hour production validation.

Each PR should be independently testable and leave the basic command usable.

---

## 13. User Decisions Required

Fill these before implementation. Recommended defaults are already supplied.

```yaml
long_audio_refactor:
  chunk_seconds: 300
  chunk_overlap_seconds: 2
  max_upload_mb: 20

  request_timeout_seconds: 600
  max_attempts_per_chunk: 2
  default_workers: 1
  allow_optional_workers_2: true

  default_transcript_style:
    selected: verbatim
    alternatives:
      - verbatim
      - both
    note: >
      Verbatim is recommended so the basic command does not automatically
      trigger an additional text-model cleanup stage.

  output_layout:
    selected: per-run-directories
    alternatives:
      - per-run-directories
      - retain-single-output-directory

  keep_legacy_retries_flag: true
  keep_transcribe_auto_split_wrapper: true

  long_audio_validation_source:
    description: "[Provide a sanitized or approved 2+ hour recording]"
    may_be_uploaded_to_openai: "[yes/no]"
    contains_sensitive_participant_data: "[yes/no]"

  acceptable_parallelism:
    selected: 1
    maximum_allowed: 2

  failure_policy:
    stop_after_terminal_chunk_failure: true
    continue_other_chunks_after_failure: false
```

---

## 14. Definition of Done

The refactor is complete when a user can run:

```bash
python3 transcribe.py "recordings/SRA-Interview.m4a" \
  --profile interview \
  --language auto \
  --speaker "Interviewer=speaker_references/nikko.m4a" \
  --speaker "SRA Staff=speaker_references/sra-staff.m4a" \
  --glossary "config/sugarcane-glossary.txt" \
  --transcript-style both \
  --export txt,json,srt,vtt
```

and observe:

1. a preflight summary;
2. duration-based chunk planning;
3. controlled request attempts;
4. visible per-chunk progress;
5. a provisional transcript after the first chunk;
6. safe cancellation with `Ctrl+C`;
7. correct resume without repeating completed chunks;
8. finalized core transcript files before cleanup begins;
9. resumable readable and glossary processing;
10. no hidden SDK retry multiplication;
11. a manifest showing exactly what happened;
12. documentation explaining every output and recovery command.

---

## 15. Recommended First Implementation Step

Start with **PR 1: configuration, manifest schema v2, and run fingerprints**.

Do not begin with concurrency.

The most valuable order is:

```text
Correct chunk planning
→ Controlled requests
→ Durable incremental outputs
→ Reliable resume
→ Batched post-processing
→ Optional concurrency
```

That order fixes the current problem without introducing new race conditions or making the code unnecessarily clever.
