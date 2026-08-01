import unittest

from transcriber.cli import assignment
from transcriber.renderers import clock, subtitle


class TranscriberHelpersTest(unittest.TestCase):
    def test_assignment(self):
        self.assertEqual(assignment("A=Interviewer", "--speaker-label"), ("A", "Interviewer"))

    def test_assignment_requires_equals(self):
        with self.assertRaises(ValueError):
            assignment("Interviewer", "--speaker")

    def test_clock_formats(self):
        self.assertEqual(clock(3661.25), "01:01:01")
        self.assertEqual(clock(1.25, True), "00:00:01,250")
        self.assertEqual(clock(1.25, True, True), "00:00:01.250")

    def test_srt_contains_speaker_and_timestamp(self):
        rendered = subtitle([
            {"speaker": "Interviewer", "start": 1.0, "end": 3.5, "text": "Good morning."}
        ])
        self.assertIn("00:00:01,000 --> 00:00:03,500", rendered)
        self.assertIn("Interviewer: Good morning.", rendered)


if __name__ == "__main__":
    unittest.main()
