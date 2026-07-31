"""Compatibility entrypoint for the improved transcriber.

New commands should use `python transcribe.py ...`. Existing users may continue
running this filename; all command-line arguments are forwarded unchanged.
"""

from transcribe import main


if __name__ == "__main__":
    raise SystemExit(main())
