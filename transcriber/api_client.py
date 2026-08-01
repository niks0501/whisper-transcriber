from __future__ import annotations

import os

from openai import OpenAI


def create_client(timeout: int = 600) -> OpenAI:
    return OpenAI(max_retries=0, timeout=float(timeout))
