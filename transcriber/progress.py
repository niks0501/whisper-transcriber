from __future__ import annotations

import time

from transcriber.media import format_duration


class Progress:
    def __init__(self, total_chunks: int, total_duration: float):
        self.total_chunks = total_chunks
        self.total_duration = total_duration
        self.completed = 0
        self.total_processed: float = 0.0
        self.chunk_elapsed: float = 0.0
        self.start_time = time.monotonic()
        self.chunk_times: list[float] = []

    def chunk_done(self, chunk_id: str, processed_seconds: float, chunk_elapsed: float) -> None:
        self.completed += 1
        self.total_processed = processed_seconds
        self.chunk_elapsed = chunk_elapsed
        self.chunk_times.append(chunk_elapsed)

    def percent(self) -> float:
        return (self.total_processed / self.total_duration * 100) if self.total_duration > 0 else 100.0

    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    def eta_seconds(self) -> float:
        if self.completed == 0:
            return 0.0
        avg = sum(self.chunk_times) / len(self.chunk_times)
        remaining = self.total_chunks - self.completed
        return avg * remaining

    def eta(self) -> str:
        return format_duration(self.eta_seconds())

    def format_line(self, chunk_id: str) -> str:
        return (
            f"[{self.completed:02d}/{self.total_chunks:02d}] Completed {chunk_id}\n"
            f"Audio processed: {format_duration(self.total_processed)} / {format_duration(self.total_duration)} "
            f"({self.percent():.1f}%)\n"
            f"Chunk time: {format_duration(self.chunk_elapsed)}\n"
            f"Elapsed: {format_duration(self.elapsed())}\n"
            f"Estimated transcription remaining: {self.eta()}"
        )


def status_summary(
    completed: int,
    total: int,
    processed_duration: float,
    total_duration: float,
    current_stage: str,
    core_status: str,
    readable_status: str,
    last_error: str | None = None,
) -> str:
    return (
        f"Transcription: {completed}/{total} chunks completed\n"
        f"Processed: {format_duration(processed_duration)} / {format_duration(total_duration)}\n"
        f"Current stage: {current_stage}\n"
        f"Last error: {last_error or 'none'}\n"
        f"Core transcript: {core_status}\n"
        f"Readable copy: {readable_status}"
    )
