# whisper-transcriber

Simple, reliable audio transcription using OpenAI Whisper-style processing.

This repository provides a small, focused script to transcribe common audio formats (mp3, mp4, m4a, wav, webm). It splits long audio when needed, runs transcription, and writes human-readable text files to the transcription_output folder.

Features
- Accepts mp3, mp4, m4a, wav, webm
- Auto-splits long audio for more reliable transcription
- Produces plain text output per audio file in transcription_output/

Quickstart
1. Ensure Python 3.8+ is installed.
2. Place your audio file in the repository root (or give a full path).
3. Run the transcriber:

   python transcribe_auto_split.py "path\to\your_audio.m4a"

4. Find transcripts in the transcription_output\ directory.

Notes
- The script is designed to be simple and easy to use. For large files it will split audio to avoid memory or model limits.
- If you need different output formats (SRT, JSON), consider extending the script or post-processing the generated text files.

Files of interest
- transcribe_auto_split.py — main transcription script
- transcription_output\ — folder where transcripts are saved
- tubo-interview2026.m4a — example audio included for testing

License
This project is provided under the repository License (see LICENSE).

If you want more customization (batch processing, different models, or alternate output formats), open an issue or submit a pull request.
