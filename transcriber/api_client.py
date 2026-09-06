from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from transcriber.meta_transcription import MUSE_TRANSCRIBE_MODEL, transcribe_muse_file


class _RoutedTranscriptions:
    def __init__(self, openai_client: OpenAI, timeout: int) -> None:
        self._openai_client = openai_client
        self._timeout = timeout

    def create(self, **kwargs: Any) -> Any:
        model = str(kwargs.get("model") or "")
        if model == MUSE_TRANSCRIBE_MODEL:
            audio = kwargs.get("file")
            if audio is None or not getattr(audio, "name", None):
                raise ValueError("Muse transcription requires an audio file with a filesystem path.")
            return transcribe_muse_file(
                Path(audio.name),
                language=kwargs.get("language"),
                timeout_seconds=self._timeout,
            )
        return self._openai_client.audio.transcriptions.create(**kwargs)


class _RoutedAudio:
    def __init__(self, openai_client: OpenAI, timeout: int) -> None:
        self.transcriptions = _RoutedTranscriptions(openai_client, timeout)


class RoutedClient:
    def __init__(self, timeout: int = 600) -> None:
        key = os.getenv("OPENAI_API_KEY") or "unused-for-meta-transcription"
        self._openai = OpenAI(
            api_key=key,
            max_retries=0,
            timeout=float(timeout),
        )
        self.audio = _RoutedAudio(self._openai, timeout)
        self.responses = self._openai.responses


def create_client(timeout: int = 600) -> RoutedClient:
    return RoutedClient(timeout=timeout)
