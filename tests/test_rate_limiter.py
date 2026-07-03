"""Tests for SlidingWindowRateLimiter rate limiting logic."""
import asyncio
import time
from unittest.mock import patch

from pageindex.utils import SlidingWindowRateLimiter


class TestSlidingWindowRateLimiter:

    def test_wait_time_under_limit(self):
        limiter = SlidingWindowRateLimiter(requests_per_minute=60)
        wait = limiter._wait_time()
        assert wait == 0.0

    def test_wait_time_over_limit(self):
        limiter = SlidingWindowRateLimiter(requests_per_minute=1)
        limiter._wait_time()
        wait = limiter._wait_time()
        assert wait > 0
        assert wait <= 60.0

    def test_window_sliding(self):
        limiter = SlidingWindowRateLimiter(requests_per_minute=2)
        limiter._wait_time()
        limiter._wait_time()

        limiter.request_times = [time.time() - 61.0]
        wait = limiter._wait_time()
        assert wait == 0.0

    def test_minimum_spacing(self):
        rpm = 30
        limiter = SlidingWindowRateLimiter(requests_per_minute=rpm)
        expected_spacing = 60.0 / rpm

        limiter._wait_time()
        wait = limiter._wait_time()
        assert wait >= expected_spacing - 0.01
        assert wait <= expected_spacing + 0.01

    def test_sync_wait_completes(self):
        limiter = SlidingWindowRateLimiter(requests_per_minute=60)
        limiter.wait_sync()
        start = time.time()
        limiter.wait_sync()
        elapsed = time.time() - start
        assert elapsed >= 0.9

    def test_async_wait_completes(self):
        limiter = SlidingWindowRateLimiter(requests_per_minute=60)
        limiter.wait_sync()

        async def run_test():
            start = time.time()
            await limiter.wait_async()
            return time.time() - start

        elapsed = asyncio.run(run_test())
        assert elapsed >= 0.9

    def test_disabled_limiter(self):
        limiter_zero = SlidingWindowRateLimiter(requests_per_minute=0)
        limiter_neg = SlidingWindowRateLimiter(requests_per_minute=-5)

        start = time.time()
        limiter_zero.wait_sync()
        limiter_neg.wait_sync()
        assert time.time() - start < 0.1

    def test_disabled_limiter_async(self):
        limiter = SlidingWindowRateLimiter(requests_per_minute=0)

        async def run_test():
            start = time.time()
            await limiter.wait_async()
            return time.time() - start

        elapsed = asyncio.run(run_test())
        assert elapsed < 0.1
