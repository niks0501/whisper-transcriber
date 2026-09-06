import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from transcriber.meta_transcription import _language_bias, transcribe_muse_file


class LanguageBiasTest(unittest.TestCase):
    def test_taglish_biases_english_and_tagalog(self):
        self.assertEqual(_language_bias("taglish"), ["English", "Tagalog"])

    def test_iso_tagalog_maps_to_meta_name(self):
        self.assertEqual(_language_bias("tl"), ["Tagalog"])

    def test_auto_does_not_force_language(self):
        self.assertEqual(_language_bias("auto"), [])


class MuseTranscriptionRequestTest(unittest.TestCase):
    def test_diarized_turns_are_normalized_to_existing_segment_shape(self):
        with TemporaryDirectory() as tmp:
            wav = Path(tmp) / "sample.wav"
            wav.write_bytes(b"RIFF-fake-test-audio")

            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "sessionId": "session-123",
                "audioDurationMs": 4500,
                "transcript": "Hello. Kumusta?",
                "turns": [
                    {
                        "turnId": 1,
                        "startMs": 0,
                        "endMs": 2000,
                        "transcript": "Hello.",
                        "speaker": "A",
                    },
                    {
                        "turnId": 2,
                        "startMs": 2100,
                        "endMs": 4500,
                        "transcript": "Kumusta?",
                        "speaker": "B",
                    },
                ],
            }

            with patch.dict(os.environ, {"MODEL_API_KEY": "LLM|test"}, clear=False):
                with patch("transcriber.meta_transcription.httpx.post", return_value=response) as post:
                    result = transcribe_muse_file(wav, language="taglish", timeout_seconds=30)

            self.assertEqual(result["provider"], "meta")
            self.assertEqual(result["segments"][0]["speaker"], "A")
            self.assertEqual(result["segments"][0]["start"], 0.0)
            self.assertEqual(result["segments"][0]["end"], 2.0)
            self.assertEqual(result["segments"][1]["speaker"], "B")
            self.assertEqual(result["segments"][1]["text"], "Kumusta?")

            _, kwargs = post.call_args
            self.assertEqual(kwargs["headers"]["Authorization"], "Bearer LLM|test")
            request_part = kwargs["files"]["request"]
            request_json = json.loads(request_part[1])
            self.assertEqual(request_json["model"], "muse-voice-transcribe-1.0")
            self.assertEqual(request_json["mode"], "DIARIZATION")
            self.assertEqual(request_json["languageBias"], ["English", "Tagalog"])


if __name__ == "__main__":
    unittest.main()
