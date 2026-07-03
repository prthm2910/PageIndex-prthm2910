"""Tests for concurrency limiter (semaphore + settings)."""
import asyncio

import pytest

from pageindex.utils import (
    set_concurrency_limit,
    set_rpm_limit,
    get_rate_limiter,
    get_llm_semaphore,
    SlidingWindowRateLimiter,
    _semaphores,
)


class TestSettingsManagement:

    def test_default_concurrency_limit(self):
        import pageindex.utils as _utils
        set_concurrency_limit(0)
        assert _utils._concurrency_limit == 0

    def test_set_concurrency_limit(self):
        import pageindex.utils as _utils
        set_concurrency_limit(10)
        assert _utils._concurrency_limit == 10

    def test_set_rpm_limit_creates_limiter(self):
        set_rpm_limit(50)
        limiter = get_rate_limiter()
        assert limiter is not None
        assert isinstance(limiter, SlidingWindowRateLimiter)
        assert limiter.requests_per_minute == 50

    def test_set_rpm_limit_zero_disables(self):
        set_rpm_limit(50)
        get_rate_limiter()
        set_rpm_limit(0)
        limiter = get_rate_limiter()
        assert limiter is None


class TestSemaphoreBehavior:

    def test_semaphore_limits_concurrency(self):
        import pageindex.utils as _utils
        _semaphores.clear()
        set_concurrency_limit(2)

        max_concurrent = 0
        current_concurrent = 0

        async def task():
            nonlocal max_concurrent, current_concurrent
            async with get_llm_semaphore():
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
                await asyncio.sleep(0.05)
                current_concurrent -= 1

        async def run():
            tasks = [task() for _ in range(6)]
            await asyncio.gather(*tasks)

        asyncio.run(run())
        assert max_concurrent <= 2

    def test_semaphore_no_deadlock(self):
        import pageindex.utils as _utils
        _semaphores.clear()
        set_concurrency_limit(1)

        completed = []

        async def task(task_id):
            async with get_llm_semaphore():
                await asyncio.sleep(0.01)
                completed.append(task_id)
                return task_id

        async def run():
            tasks = [task(i) for i in range(5)]
            results = await asyncio.gather(*tasks)
            return results

        results = asyncio.run(run())
        assert len(results) == 5
        assert len(completed) == 5
        assert set(completed) == {0, 1, 2, 3, 4}


class TestRateLimiterIntegration:

    def test_get_rate_limiter_lazy_init(self):
        import pageindex.utils as _utils
        set_rpm_limit(0)
        _utils._rate_limiter = None
        assert _utils._rate_limiter is None

        set_rpm_limit(42)

        limiter = get_rate_limiter()
        assert limiter is not None
        assert limiter.requests_per_minute == 42
