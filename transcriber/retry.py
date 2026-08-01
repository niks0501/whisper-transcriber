from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable

from openai import (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    RateLimitError,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_retryable(exc: Exception) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APITimeoutError):
        return True
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        code = exc.status_code
        if code in (408, 409):
            return True
        if 500 <= code < 600:
            return True
        if 400 <= code < 500:
            return False
    return False


def error_name(exc: Exception) -> str:
    name = type(exc).__name__
    if isinstance(exc, APIStatusError):
        return f"{name}({exc.status_code})"
    return name


def with_retry(
    operation: Callable[[], Any],
    max_attempts: int = 2,
    chunk_id: str = "",
    on_attempt: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    last_exc: Exception | None = None

    for attempt_num in range(1, max_attempts + 1):
        started = now_iso()
        t0 = time.monotonic()
        try:
            result = operation()
            elapsed = time.monotonic() - t0
            record = {
                "attempt": attempt_num,
                "started_at": started,
                "finished_at": now_iso(),
                "elapsed_seconds": round(elapsed, 2),
                "status": "succeeded",
                "request_id": getattr(result, "id", None) if hasattr(result, "id") else None,
            }
            if on_attempt:
                on_attempt(record)
            return {"result": result, "attempts": attempts + [record]}
        except Exception as exc:
            last_exc = exc
            elapsed = time.monotonic() - t0
            retryable = is_retryable(exc)
            record = {
                "attempt": attempt_num,
                "started_at": started,
                "finished_at": now_iso(),
                "elapsed_seconds": round(elapsed, 2),
                "status": "failed",
                "error_type": error_name(exc),
                "request_id": None,
                "retryable": retryable,
            }
            if on_attempt:
                on_attempt(record)
            attempts.append(record)

            if not retryable:
                raise

            if attempt_num == max_attempts:
                break

            delay = min(2 ** (attempt_num - 1), 8)
            time.sleep(delay)

    raise RuntimeError(
        f"{chunk_id} failed after {max_attempts} attempts: {error_name(last_exc) if last_exc else 'unknown'}"
    )
