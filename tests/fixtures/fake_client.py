"""Fake OpenAI client for integration testing without API calls."""

from __future__ import annotations

from typing import Any


class FakeSegment:
    def __init__(self, speaker: str, start: float, end: float, text: str):
        self.speaker = speaker
        self.start = start
        self.end = end
        self.text = text

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class FakeTranscription:
    def __init__(self, segments: list[FakeSegment], text: str = "", request_id: str | None = None):
        self.segments = segments
        self.text = text
        self.id = request_id

    def model_dump(self, mode: str = "json") -> dict[str, Any]:
        return {
            "segments": [
                {"speaker": s.speaker, "start": s.start, "end": s.end, "text": s.text}
                for s in self.segments
            ],
            "text": self.text,
        }


class FakeTextResponse:
    def __init__(self, output_text: str, request_id: str | None = None):
        self.output_text = output_text
        self.id = request_id


class FakeAudioTranscriptions:
    def __init__(self, behavior: FakeClientBehavior):
        self._behavior = behavior

    def create(self, **kwargs) -> FakeTranscription:
        return self._behavior.transcription_call(**kwargs)


class FakeResponses:
    def __init__(self, behavior: FakeClientBehavior):
        self._behavior = behavior

    def create(self, **kwargs) -> FakeTextResponse:
        return self._behavior.text_call(**kwargs)


class FakeClientBehavior:
    def __init__(self):
        self.call_counts: dict[str, int] = {}
        self.fail_counts: dict[str, int] = {}
        self.call_history: list[dict[str, Any]] = []
        self._transcription_results: dict[str, FakeTranscription] = {}
        self._text_result: FakeTextResponse = FakeTextResponse("Clean text")

    def record_call(self, chunk_key: str) -> str:
        count = self.call_counts.get(chunk_key, 0) + 1
        self.call_counts[chunk_key] = count
        call_key = f"{chunk_key}-{count}"
        self.call_history.append({"chunk": chunk_key, "attempt": count})
        return call_key

    def transcription_call(self, **kwargs) -> FakeTranscription:
        file = kwargs.get("file")
        chunk_key = getattr(file, "name", str(file))
        call_key = self.record_call(chunk_key)

        fail_remaining = self.fail_counts.get(chunk_key, 0)
        if fail_remaining > 0:
            self.fail_counts[chunk_key] = fail_remaining - 1
            from openai import APITimeoutError
            raise APITimeoutError("Simulated timeout")

        if chunk_key in self._transcription_results:
            return self._transcription_results[chunk_key]

        return FakeTranscription(
            segments=[
                FakeSegment("A", 0.0, 2.0, "Hello world."),
                FakeSegment("B", 2.5, 5.0, "Good morning."),
            ],
            request_id=call_key,
        )

    def text_call(self, **kwargs) -> FakeTextResponse:
        return self._text_result

    def set_result(self, chunk_key: str, transcription: FakeTranscription) -> None:
        self._transcription_results[chunk_key] = transcription

    def set_fail_count(self, chunk_key: str, count: int) -> None:
        self.fail_counts[chunk_key] = count

    def set_text_result(self, text: str) -> None:
        self._text_result = FakeTextResponse(text)

    def set_text_error(self, exc: Exception) -> None:
        self._text_error = exc

    def calls_for(self, chunk_key: str) -> int:
        return self.call_counts.get(chunk_key, 0)


class FakeOpenAI:
    def __init__(self, behavior: FakeClientBehavior | None = None):
        if behavior is None:
            behavior = FakeClientBehavior()
        self.audio = FakeAudio(behavior)
        self.responses = FakeResponses(behavior)


class FakeAudio:
    def __init__(self, behavior: FakeClientBehavior):
        self.transcriptions = FakeAudioTranscriptions(behavior)


def make_diarized_segments(speakers: list[tuple[str, float, float, str]]) -> list[FakeSegment]:
    return [FakeSegment(name, start, end, text) for name, start, end, text in speakers]


def make_linear_segments(chunk_key: str, total_duration: float, speaker_a: str = "A", speaker_b: str = "B") -> FakeTranscription:
    segs = []
    pos = 0.0
    turn = 0
    segment_dur = 5.0
    while pos < total_duration:
        speaker = speaker_a if turn % 2 == 0 else speaker_b
        end = min(pos + segment_dur, total_duration)
        segs.append(FakeSegment(speaker, pos, end, f"Turn {turn} content."))
        pos = end
        turn += 1
    return FakeTranscription(segments=segs, request_id=f"req-{chunk_key}")
