import os
import math
import json
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# =========================
# CONFIG
# =========================
INPUT_FILE = "tubo-interview2026.m4a"   # Just change this every time
OUTPUT_ROOT = "transcription_output"    # Main parent folder

MODEL = "whisper-1"
TARGET_CHUNK_MB = 20
LANGUAGE = None

FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"


# =========================
# HELPERS
# =========================
def check_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"'{name}' was not found in PATH.\n"
            f"Install FFmpeg and make sure '{name}' works in your terminal."
        )


def run_cmd(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{' '.join(cmd)}\n\nSTDERR:\n{result.stderr}"
        )
    return result.stdout.strip()


def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def get_duration_seconds(path: Path) -> float:
    output = run_cmd([
        FFPROBE_BIN,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ])
    return float(output)


def estimate_segment_seconds(file_size_mb: float, duration_sec: float, target_chunk_mb: float) -> int:
    """
    Estimate chunk duration so each chunk stays below target_chunk_mb.
    Adds a safety margin.
    """
    if file_size_mb <= target_chunk_mb:
        return math.ceil(duration_sec)

    mb_per_sec = file_size_mb / duration_sec
    safe_target_mb = target_chunk_mb * 0.9
    seg_seconds = int(safe_target_mb / mb_per_sec)

    return max(seg_seconds, 60)


def get_output_folder(input_path: Path, output_root: str) -> Path:
    """
    Create a unique folder for each audio file based on its filename.
    Example: tubo-interview2026.m4a -> transcription_output/tubo-interview2026/
    """
    audio_name = input_path.stem
    return Path(output_root) / audio_name


def split_audio(input_path: Path, chunks_dir: Path, segment_seconds: int) -> list[Path]:
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Preserve original extension for chunk files
    extension = input_path.suffix.lower()
    output_pattern = chunks_dir / f"chunk_%03d{extension}"

    cmd = [
        FFMPEG_BIN,
        "-i", str(input_path),
        "-f", "segment",
        "-segment_time", str(segment_seconds),
        "-c", "copy",
        str(output_pattern)
    ]
    run_cmd(cmd)

    chunk_files = sorted(chunks_dir.glob(f"chunk_*{extension}"))
    if not chunk_files:
        raise RuntimeError("No chunks were created.")
    return chunk_files


def transcribe_chunk(client: OpenAI, chunk_path: Path) -> str:
    with open(chunk_path, "rb") as audio_file:
        kwargs = {
            "model": MODEL,
            "file": audio_file,
        }

        if LANGUAGE:
            kwargs["language"] = LANGUAGE

        transcript = client.audio.transcriptions.create(**kwargs)
        return transcript.text.strip()


def merge_transcripts(transcripts: list[tuple[str, str]], merged_txt_path: Path, merged_json_path: Path) -> None:
    full_text_parts = []
    structured = []

    for filename, text in transcripts:
        full_text_parts.append(f"[{filename}]\n{text}\n")
        structured.append({
            "chunk_file": filename,
            "text": text
        })

    merged_txt_path.write_text("\n".join(full_text_parts), encoding="utf-8")
    merged_json_path.write_text(
        json.dumps(structured, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def main() -> None:
    check_binary(FFMPEG_BIN)
    check_binary(FFPROBE_BIN)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing.\n"
            "Put it in a .env file like:\n"
            "OPENAI_API_KEY=your_api_key_here"
        )

    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Create a dedicated folder for this specific audio file
    output_dir = get_output_folder(input_path, OUTPUT_ROOT)
    chunks_dir = output_dir / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)

    file_size_mb = get_file_size_mb(input_path)
    duration_sec = get_duration_seconds(input_path)

    print(f"Input file: {input_path}")
    print(f"Output folder: {output_dir}")
    print(f"Size: {file_size_mb:.2f} MB")
    print(f"Duration: {duration_sec / 60:.2f} minutes")

    segment_seconds = estimate_segment_seconds(file_size_mb, duration_sec, TARGET_CHUNK_MB)
    print(f"Target chunk size: {TARGET_CHUNK_MB} MB")
    print(f"Estimated segment length: {segment_seconds} seconds")

    chunk_files = split_audio(input_path, chunks_dir, segment_seconds)
    print(f"Created {len(chunk_files)} chunk(s).")

    client = OpenAI()
    transcripts = []

    for idx, chunk in enumerate(chunk_files, start=1):
        chunk_size = get_file_size_mb(chunk)
        print(f"\nTranscribing chunk {idx}/{len(chunk_files)}: {chunk.name} ({chunk_size:.2f} MB)")

        if chunk_size > 25:
            raise RuntimeError(
                f"Chunk {chunk.name} is above 25 MB.\n"
                f"Lower TARGET_CHUNK_MB or re-encode the audio first."
            )

        text = transcribe_chunk(client, chunk)
        transcripts.append((chunk.name, text))

        # Save each chunk transcript inside the same audio-specific folder
        per_chunk_txt = output_dir / f"{chunk.stem}.txt"
        per_chunk_txt.write_text(text, encoding="utf-8")

    merged_txt_path = output_dir / "full_transcript.txt"
    merged_json_path = output_dir / "full_transcript.json"
    merge_transcripts(transcripts, merged_txt_path, merged_json_path)

    print("\nDone.")
    print(f"Final merged TXT: {merged_txt_path}")
    print(f"Structured JSON : {merged_json_path}")


if __name__ == "__main__":
    main()