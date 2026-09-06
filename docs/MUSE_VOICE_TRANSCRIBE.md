# Muse Voice Transcribe

The default `interview` profile uses Meta Muse Voice Transcribe with diarization.
The existing OpenAI transcription profiles remain available as fallbacks.

## Credentials

Create `.env` from `.env.example` and set a Meta Model API key:

```env
MODEL_API_KEY=LLM|your_key_here
```

`META_API_KEY` is accepted as an alternative environment variable name.
The standalone transcriber does not reuse Pi's `~/.pi/agent/auth.json` OAuth credential because that credential is managed and refreshed by the Pi extension.

## Basic interview

```bash
python transcribe.py "recordings/interview.m4a" --transcribe-only
```

The input can still be M4A, MP3, MP4, WAV, or another format supported by the CLI.
Before upload, FFmpeg converts each request to mono 16-bit PCM WAV at 16 kHz because that is the format used by the Muse file transcription endpoint.

## Taglish interview

Muse supports multilingual code-switching without forcing one language.
Use `--language auto` for unrestricted automatic recognition, or use `--language taglish` to bias recognition toward English and Tagalog.

```bash
python transcribe.py "recordings/interview.m4a" \
  --profile interview \
  --language taglish \
  --map-speakers \
  --transcribe-only
```

## Profiles

| Profile | Provider/model | Purpose |
| --- | --- | --- |
| `interview` | Meta `muse-voice-transcribe-1.0` | Default interview transcription with diarization |
| `openai-interview` | OpenAI `gpt-4o-transcribe-diarize` | OpenAI diarization and known-speaker references |
| `accurate` | OpenAI `gpt-4o-transcribe` | Plain transcription |
| `budget` | OpenAI `gpt-4o-mini-transcribe` | Lower-cost OpenAI transcription |
| `legacy` | OpenAI `whisper-1` | Legacy compatibility |

## Speaker handling

Muse returns anonymous speaker labels such as `A` and `B` with turn timestamps.
Use `--map-speakers` after transcription to rename those labels to names such as `Interviewer` and `Participant`.
Meta Muse Voice Transcribe does not use the repository's `--speaker NAME=FILE` reference clips, so those remain specific to `openai-interview`.

For long recordings, the repository keeps its chunk checkpoint and resume system.
Speaker labels returned by diarization are scoped to an individual Meta transcription request, so labels can theoretically change identity between chunks.
Review speaker labels when producing research quotations from a long multi-chunk interview.

## Optional text post-processing

Muse is used only for speech-to-text in this change.
Readable cleanup, glossary review, and thesis analysis still use the model configured by `OPENAI_TEXT_MODEL` and therefore require `OPENAI_API_KEY`.
Use `--transcribe-only` if you want the core Muse transcript without OpenAI post-processing.
