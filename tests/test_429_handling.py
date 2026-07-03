"""Tests for 429 rate limit detection and retry behavior."""
import asyncio
import time
from unittest.mock import patch, MagicMock

import pytest

from pageindex.utils import (
    _is_rate_limit_error,
    _extract_retry_headers,
    _extract_retry_delay,
    llm_completion,
    llm_acompletion,
)


class MockRateLimitError(Exception):
    def __init__(self, message, headers=None):
        super().__init__(message)
        self._response_headers = headers or {}


class TestRateLimitDetection:

    def test_detects_429_in_message(self):
        err = Exception("HTTP 429 Too Many Requests")
        assert _is_rate_limit_error(err) is True

    def test_detects_rate_limit_text(self):
        err = Exception("Rate limit exceeded for this model")
        assert _is_rate_limit_error(err) is True

    def test_detects_too_many_requests(self):
        err = Exception("Error: too many requests, please slow down")
        assert _is_rate_limit_error(err) is True

    def test_non_rate_limit_error(self):
        err = Exception("Connection refused")
        assert _is_rate_limit_error(err) is False

    def test_detects_litellm_rate_limit_error(self):
        import litellm
        if hasattr(litellm, "RateLimitError"):
            err = litellm.RateLimitError(
                "Rate limited",
                llm_provider="openai",
                model="gpt-4"
            )
            assert _is_rate_limit_error(err) is True


class TestRetryHeaderExtraction:

    def test_extracts_retry_after_from_headers(self):
        err = MockRateLimitError("429", headers={"Retry-After": "30"})
        retry_after, reset = _extract_retry_headers(err)
        assert retry_after == "30"

    def test_extracts_ratelimit_reset_from_headers(self):
        err = MockRateLimitError("429", headers={
            "x-ratelimit-reset-requests": "45",
        })
        retry_after, reset = _extract_retry_headers(err)
        assert reset == "45"

    def test_returns_none_when_no_headers(self):
        err = Exception("429 Too Many Requests")
        retry_after, reset = _extract_retry_headers(err)
        assert retry_after is None
        assert reset is None


class TestRetryDelayCalculation:

    def test_retry_after_header_respected(self):
        err = MockRateLimitError("429", headers={"Retry-After": "10"})
        delay = _extract_retry_delay(err, 0)
        assert delay >= 10

    def test_fallback_exponential_backoff(self):
        err = Exception("429 Too Many Requests")
        delay_0 = _extract_retry_delay(err, 0)
        delay_1 = _extract_retry_delay(err, 1)
        delay_2 = _extract_retry_delay(err, 2)
        assert delay_1 > delay_0
        assert delay_2 > delay_1

    def test_non_429_uses_standard_backoff(self):
        err = Exception("Connection timeout")
        delay = _extract_retry_delay(err, 0)
        assert delay <= 10.5

    def test_no_thundering_herd(self):
        err = Exception("429 Too Many Requests")
        delays = [_extract_retry_delay(err, 0) for _ in range(20)]
        unique_delays = set(round(d, 2) for d in delays)
        assert len(unique_delays) > 1

    def test_retry_after_capped(self):
        err = MockRateLimitError("429", headers={"Retry-After": "3600"})
        delay = _extract_retry_delay(err, 0)
        assert delay <= 121


class TestLLMCompletionRetry:

    def test_llm_completion_returns_empty_on_max_retries(self):
        call_count = 0

        def mock_completion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("429 Too Many Requests")

        with patch("pageindex.utils.litellm.completion", side_effect=mock_completion):
            with patch("pageindex.utils.time.sleep"):
                result = llm_completion("gpt-4", "test")
                assert result == ""
                assert call_count == 10

    def test_llm_completion_succeeds_on_retry(self):
        call_count = 0

        def mock_completion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("429 Too Many Requests")
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = '{"answer": "yes"}'
            response.choices[0].finish_reason = "finished"
            return response

        with patch("pageindex.utils.litellm.completion", side_effect=mock_completion):
            with patch("pageindex.utils.time.sleep"):
                result = llm_completion("gpt-4", "test")
                assert result == '{"answer": "yes"}'

    @pytest.mark.asyncio
    async def test_llm_acompletion_returns_empty_on_max_retries(self):
        call_count = 0

        async def mock_acompletion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("429 Too Many Requests")

        with patch("pageindex.utils.litellm.acompletion", side_effect=mock_acompletion):
            with patch("pageindex.utils.asyncio.sleep"):
                result = await llm_acompletion("gpt-4", "test")
                assert result == ""
                assert call_count == 10
