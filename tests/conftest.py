"""Shared fixtures for PageIndex tests."""
import pytest

from pageindex.utils import set_concurrency_limit, set_rpm_limit


@pytest.fixture(autouse=True)
def reset_global_settings():
    """Reset global concurrency and RPM settings after each test."""
    yield
    set_concurrency_limit(0)
    set_rpm_limit(0)
    import pageindex.utils as _utils
    _utils._semaphores.clear()
    _utils._rate_limiter = None
