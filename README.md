# Whisper Transcriber

A command-line transcriber for recorded interviews, meetings, and thesis research.

The default workflow uses `gpt-4o-transcribe-diarize` to detect who spoke and when. It preserves the raw API response, creates speaker-labelled transcripts and subtitles, resumes interrupted runs, and can optionally generate a readable copy and thesis research notes.

## Highlights

- One-command basic transcription
- Guided terminal wizard
- Advanced thesis command
- Automatic speaker diarization and timestamps
- Optional known-speaker reference samples
- Interactive or command-based speaker renaming
- Retry and resume support with per-chunk checkpoints
- TXT, JSON, SRT, and VTT exports
- Verbatim and readable transcript copies
- Sugarcane terminology glossary review
- Thesis summary, themes, timestamped quotations, Q&A mapping, and follow-up questions
- Privacy-safe Git defaults for recordings and generated transcripts

## Privacy notice

This tool uploads the selected recording to the OpenAI API for processing. Obtain the required consent before uploading an interview.

Raw recordings, speaker samples, `.env`, private configuration, and generated transcripts are ignored by Git. Do not force-add sensitive research material.

## Requirements

- Python 3.10 or newer
- FFmpeg and FFprobe
- An OpenAI API key

## Installation

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_TEXT_MODEL=gpt-4.1-mini
```

Confirm FFmpeg is available:

```powershell
ffmpeg -version
ffprobe -version
```

### WSL or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then add your API key to `.env`.

## Three ways to run it

### 1. Basic command

```powershell
python transcribe.py "recordings\SRA-interview.m4a"
```

This is the normal command. You only provide the raw recording.

Defaults:

- profile: `interview`
- model: `gpt-4o-transcribe-diarize`
- language: automatic detection
- transcript copies: verbatim and readable
- exports: TXT, JSON, SRT, and VTT
- retries and resume: enabled

Without voice references, detected voices are labelled automatically. The exact raw label may be `A`, `B`, `Speaker A`, or another model-provided label.

### 2. Guided wizard

```powershell
python transcribe.py --wizard
```

The wizard asks for:

- raw audio path;
- transcription profile;
- language;
- optional 2–10 second speaker samples;
- whether detected speakers should be renamed;
- transcript style;
- optional terminology glossary;
- optional thesis analysis and research context.

### 3. Advanced thesis command

```powershell
python transcribe.py "recordings\SRA-interview.m4a" `
  --profile interview `
  --language auto `
  --speaker "Interviewer=speaker_references\nikko.wav" `
  --speaker "SRA Staff=speaker_references\sra-staff.wav" `
  --glossary "config\sugarcane-glossary.txt" `
  --transcript-style both `
  --export txt,json,srt,vtt `
  --analyze `
  --research-context "config\caneguard-thesis.yaml"
```

This transcribes the interview, identifies known voices where possible, produces transcript and subtitle files, checks specialist terminology, and creates thesis-oriented research notes.

## Speaker separation

### Automatic diarization

No `--speaker` option is required:

```powershell
python transcribe.py "recordings\SRA-interview.m4a"
```

The model separates voices and returns timestamped segments. It does not automatically know which person is the interviewer unless you provide references or rename the labels.

### Rename speakers after transcription

```powershell
python transcribe.py "recordings\SRA-interview.m4a" --map-speakers
```

After transcription, the tool asks for display names for each detected label.

You can also map labels directly after checking a previous run:

```powershell
python transcribe.py "recordings\SRA-interview.m4a" `
  --speaker-label "A=Interviewer" `
  --speaker-label "B=SRA Staff"
```

### Known-speaker samples

```powershell
python transcribe.py "recordings\SRA-interview.m4a" `
  --speaker "Interviewer=speaker_references\nikko.wav" `
  --speaker "SRA Staff=speaker_references\sra-staff.wav"
```

Each sample must contain only that person's voice and must be between 2 and 10 seconds. Up to four known speakers can be supplied.

The transcriber validates each reference before uploading it.

## Profiles

| Profile | Model | Best use |
|---|---|---|
| `interview` | `gpt-4o-transcribe-diarize` | Interviews requiring speakers and timestamps |
| `accurate` | `gpt-4o-transcribe` | High-quality plain transcription |
| `budget` | `gpt-4o-mini-transcribe` | Lower-cost plain transcription |
| `legacy` | `whisper-1` | Compatibility with the old workflow |

Examples:

```powershell
python transcribe.py "audio.m4a" --profile accurate
python transcribe.py "audio.m4a" --profile budget
python transcribe.py "audio.m4a" --profile legacy
```

Override the profile model:

```powershell
python transcribe.py "audio.m4a" --model gpt-4o-transcribe-diarize
```

## Language

Automatic or Taglish/mixed-language starting point:

```powershell
python transcribe.py "audio.m4a" --language auto
```

English:

```powershell
python transcribe.py "audio.m4a" --language en
```

Tagalog:

```powershell
python transcribe.py "audio.m4a" --language tl
```

## Transcript styles

Verbatim only:

```powershell
python transcribe.py "audio.m4a" --transcript-style verbatim
```

Readable only, while retaining raw JSON:

```powershell
python transcribe.py "audio.m4a" --transcript-style readable
```

Both, recommended for thesis interviews:

```powershell
python transcribe.py "audio.m4a" --transcript-style both
```

The readable copy is produced by a text model instructed to preserve all speakers, timestamps, claims, numbers, uncertainty, and language choices. Verify it against the raw transcript before quoting or publishing it.

## Glossary

Copy the example:

```powershell
Copy-Item config\sugarcane-glossary.example.txt config\sugarcane-glossary.txt
```

Edit it so there is one approved term per line, then run:

```powershell
python transcribe.py "audio.m4a" `
  --glossary "config\sugarcane-glossary.txt" `
  --transcript-style both
```

The tool creates `review_flags.json` containing names and technical phrases that deserve human verification. These are text-review suggestions, not acoustic confidence scores.

## Thesis research analysis

Copy the example context:

```powershell
Copy-Item config\caneguard-thesis.example.yaml config\caneguard-thesis.yaml
```

Edit the project title, organization, research questions, and privacy settings.

```powershell
python transcribe.py "recordings\SRA-interview.m4a" `
  --analyze `
  --research-context "config\caneguard-thesis.yaml"
```

The research report includes:

- interview summary;
- themes and subthemes;
- key quotations with speaker and timestamp;
- question-and-answer map;
- possible follow-up questions.

The result is an AI-assisted research aid, not final qualitative coding. Verify all quotations and interpretations against the source recording.

Override the text model used for readable cleanup and analysis:

```powershell
python transcribe.py "audio.m4a" --analysis-model gpt-4.1-mini
```

## Resume and overwrite

Every completed audio request is checkpointed immediately.

After an interruption, rerun the same command:

```powershell
python transcribe.py "recordings\SRA-interview.m4a"
```

Completed chunks are reused.

Disable resume:

```powershell
python transcribe.py "recordings\SRA-interview.m4a" --no-resume
```

Start from scratch:

```powershell
python transcribe.py "recordings\SRA-interview.m4a" --overwrite
```

The source file's SHA-256 checksum prevents accidental reuse of an output folder for a different recording.

## Large recordings

Files below the conservative upload threshold are sent as-is. Larger recordings are converted into temporary mono, 16 kHz, 64 kbps MP3 chunks of approximately 15 minutes each. The original file is never modified.

When a large interview is divided into multiple requests, anonymous A/B speaker labels may not remain consistent between chunks. Known-speaker samples improve consistency.

## Output structure

```text
transcription_output/
└── SRA-interview/
    ├── run_manifest.json
    ├── raw_transcript.json
    ├── transcript.json
    ├── verbatim_transcript.txt
    ├── readable_transcript.txt
    ├── transcript.srt
    ├── transcript.vtt
    ├── review_flags.json
    ├── research/
    │   └── research_analysis.md
    └── working/
        ├── audio/
        └── results/
```

Not every optional file is created on every run.

Important files:

- `raw_transcript.json`: model segments and raw API responses;
- `run_manifest.json`: source checksum, progress, model, and completion state;
- `verbatim_transcript.txt`: timestamped speaker-labelled transcript;
- `readable_transcript.txt`: optional cleaned copy;
- `transcript.srt` and `transcript.vtt`: subtitles for review;
- `review_flags.json`: terminology requiring manual checking;
- `research_analysis.md`: optional thesis research notes.

## Command reference

```text
python transcribe.py [AUDIO] [OPTIONS]
```

Common options:

```text
--wizard
--profile interview|accurate|budget|legacy
--model MODEL_ID
--language auto|en|tl|...
--speaker "NAME=REFERENCE_FILE"
--speaker-label "RAW_LABEL=DISPLAY_NAME"
--map-speakers
--glossary FILE
--transcript-style verbatim|readable|both
--export txt,json,srt,vtt
--analyze
--research-context FILE
--analysis-model MODEL_ID
--output DIRECTORY
--retries NUMBER
--no-resume
--overwrite
```

Show built-in help:

```powershell
python transcribe.py --help
```

## Backward compatibility

The old script name remains available:

```powershell
python transcribe_auto_split.py "recordings\SRA-interview.m4a"
```

It forwards all arguments to the new CLI. Editing a hard-coded `INPUT_FILE` variable is no longer necessary.

## Testing

```powershell
python -m compileall -q transcribe.py transcribe_auto_split.py
python -m unittest discover -s tests -v
python transcribe.py --help
```

The unit tests and help check do not call the OpenAI API or upload audio.
