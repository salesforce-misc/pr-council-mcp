from datetime import UTC, datetime

import anthropic
import httpx2 as httpx
import openai
import pytest

from pr_council.agents.review.common import (
    _MAX_RESET_DELAY_SECONDS,
    _reset_after_seconds,
    call_with_rate_limit_retry,
    is_rate_limit_error,
    is_retryable_error,
    rate_limit_delay,
)


def _litellm_rate_limit(reset_at: str) -> openai.RateLimitError:
    """A 429 shaped like litellm's: reset instant in the message, no Retry-After."""
    message = (
        "Error code: 429 - {'error': {'message': 'litellm.RateLimitError: Rate limit "
        f"exceeded for api_key: abc. Limit type: requests. Current limit: 50, "
        f"Remaining: 0. Limit resets at: {reset_at} UTC', 'type': 'throttling_error'}}}}"
    )
    return openai.RateLimitError(message, response=_response(429), body=None)


def _response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://gateway.example/v1/messages")
    return httpx.Response(status, headers=headers or {}, request=request)


class _StatusError(Exception):
    """Minimal stand-in exposing the ``status_code`` SDK errors carry."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def test_is_rate_limit_error_detects_sdk_and_status():
    assert is_rate_limit_error(openai.RateLimitError("rl", response=_response(429), body=None))
    assert is_rate_limit_error(anthropic.RateLimitError("rl", response=_response(429), body=None))
    assert is_rate_limit_error(_StatusError(429))
    assert not is_rate_limit_error(_StatusError(400))
    assert not is_rate_limit_error(ValueError("nope"))


def test_is_retryable_error_covers_transient_faults():
    request = httpx.Request("POST", "https://gateway.example/v1/messages")
    assert is_retryable_error(_StatusError(429))
    assert is_retryable_error(_StatusError(503))
    assert is_retryable_error(openai.APIConnectionError(message="boom", request=request))
    assert is_retryable_error(anthropic.APITimeoutError(request=request))
    # Deterministic failures must not be retried.
    assert not is_retryable_error(_StatusError(400))
    assert not is_retryable_error(ValueError("validation"))


def test_rate_limit_delay_honors_retry_after_header():
    exc = openai.RateLimitError("rl", response=_response(429, {"retry-after": "7"}), body=None)
    assert rate_limit_delay(exc, 1, base_delay=2.0, max_delay=60.0) == 7.0


def test_rate_limit_delay_caps_retry_after_at_max():
    exc = anthropic.RateLimitError("rl", response=_response(429, {"retry-after": "900"}), body=None)
    assert rate_limit_delay(exc, 1, base_delay=2.0, max_delay=60.0) == 60.0


def test_rate_limit_delay_falls_back_to_exponential_backoff():
    exc = _StatusError(503)
    # No Retry-After: capped exponential backoff base*2**(attempt-1) plus <=25% jitter.
    first = rate_limit_delay(exc, 1, base_delay=2.0, max_delay=60.0)
    second = rate_limit_delay(exc, 2, base_delay=2.0, max_delay=60.0)
    assert 2.0 <= first <= 2.5
    assert 4.0 <= second <= 5.0


def test_reset_after_seconds_parses_litellm_window():
    exc = _litellm_rate_limit("2026-09-03 01:11:46")
    now = datetime(2026, 9, 3, 1, 11, 16, tzinfo=UTC)
    assert _reset_after_seconds(exc, now=now) == 30.0
    # A window already in the past collapses to an immediate retry.
    past = datetime(2026, 9, 3, 1, 12, 0, tzinfo=UTC)
    assert _reset_after_seconds(exc, now=past) == 0.0
    # Errors without the sentinel yield no reset hint.
    assert _reset_after_seconds(_StatusError(429)) is None
    # A bogus far-future instant (the message is untrusted) is clamped to the
    # fixed ceiling rather than trusted as a multi-year wait.
    far_future = _litellm_rate_limit("2099-01-01 00:00:00")
    assert _reset_after_seconds(far_future, now=now) == _MAX_RESET_DELAY_SECONDS


def test_reset_after_seconds_matches_regardless_of_casing():
    # The gateway controls this message's capitalization; parsing must not break
    # if it arrives fully upper-cased.
    exc = _StatusError(429)
    exc.args = ("RATE LIMIT EXCEEDED. LIMIT RESETS AT: 2026-09-03 01:11:46 UTC",)
    now = datetime(2026, 9, 3, 1, 11, 16, tzinfo=UTC)
    assert _reset_after_seconds(exc, now=now) == 30.0


def test_rate_limit_delay_waits_for_reset_window_without_retry_after():
    # Reset already elapsed -> only the jittered buffer (1s + <1s) applies.
    exc = _litellm_rate_limit("2000-01-01 00:00:00")
    delay = rate_limit_delay(exc, 1, base_delay=2.0, max_delay=90.0)
    assert 1.0 <= delay < 2.0


def test_rate_limit_delay_caps_reset_wait_at_max():
    # A reset far in the future is bounded by max_delay, not the raw window.
    exc = _litellm_rate_limit("2099-01-01 00:00:00")
    assert rate_limit_delay(exc, 1, base_delay=2.0, max_delay=90.0) == 90.0


def test_rate_limit_delay_clamps_reset_wait_to_fixed_ceiling():
    # Even with a generous operator max_delay, an untrusted far-future reset
    # instant cannot stall a reviewer past the fixed ceiling (plus jitter).
    exc = _litellm_rate_limit("2099-01-01 00:00:00")
    delay = rate_limit_delay(exc, 1, base_delay=2.0, max_delay=100_000.0)
    assert _MAX_RESET_DELAY_SECONDS < delay <= _MAX_RESET_DELAY_SECONDS + 2.0


def test_rate_limit_delay_ignores_retry_after_for_non_rate_limit_errors():
    # A 5xx that happens to carry Retry-After still uses backoff, not the header.
    exc = _StatusError(503)
    exc.response = _response(503, {"retry-after": "900"})  # type: ignore[attr-defined]
    assert rate_limit_delay(exc, 1, base_delay=2.0, max_delay=60.0) <= 2.5


async def test_call_with_rate_limit_retry_recovers_after_transient_error(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(delay):
        slept.append(delay)

    monkeypatch.setattr("pr_council.agents.review.common.asyncio.sleep", fake_sleep)

    attempts = 0
    retries: list[tuple[int, float]] = []

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise _StatusError(429)
        return "ok"

    result = await call_with_rate_limit_retry(
        operation,
        max_attempts=3,
        base_delay=2.0,
        max_delay=60.0,
        on_retry=lambda attempt, delay, exc: retries.append((attempt, delay)),
    )

    assert result == "ok"
    assert attempts == 3
    assert len(slept) == 2
    assert [attempt for attempt, _ in retries] == [1, 2]


async def test_call_with_rate_limit_retry_reraises_non_retryable_immediately(monkeypatch):
    async def fake_sleep(delay):
        raise AssertionError("must not sleep for a non-retryable error")

    monkeypatch.setattr("pr_council.agents.review.common.asyncio.sleep", fake_sleep)

    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1
        raise ValueError("deterministic")

    with pytest.raises(ValueError, match="deterministic"):
        await call_with_rate_limit_retry(operation, max_attempts=3, base_delay=2.0, max_delay=60.0)
    assert attempts == 1


async def test_call_with_rate_limit_retry_exhausts_attempts_and_raises(monkeypatch):
    async def fake_sleep(delay):
        return None

    monkeypatch.setattr("pr_council.agents.review.common.asyncio.sleep", fake_sleep)

    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1
        raise _StatusError(429)

    with pytest.raises(_StatusError):
        await call_with_rate_limit_retry(operation, max_attempts=3, base_delay=2.0, max_delay=60.0)
    assert attempts == 3
