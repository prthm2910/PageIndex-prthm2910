"""Tests for cascading error recovery (KeyError prevention)."""
import asyncio
from unittest.mock import patch, MagicMock

import pytest

from pageindex.page_index import (
    toc_detector_single_page,
    generate_toc_init,
    generate_toc_continue,
    add_page_number_to_toc,
)


class TestTocDetectorRecovery:

    def test_returns_false_on_empty_response(self):
        with patch("pageindex.page_index.llm_completion", return_value=""):
            result = toc_detector_single_page("some text", model="gpt-4")
            assert result is False

    def test_returns_false_on_malformed_json(self):
        with patch("pageindex.page_index.llm_completion", return_value="just plain text"):
            result = toc_detector_single_page("some text", model="gpt-4")
            assert result is False

    def test_returns_false_on_missing_key(self):
        with patch("pageindex.page_index.llm_completion", return_value='{"thinking": "test"}'):
            result = toc_detector_single_page("some text", model="gpt-4")
            assert result is False

    def test_returns_true_when_valid(self):
        with patch("pageindex.page_index.llm_completion", return_value='{"thinking": "test", "toc_detected": "yes"}'):
            result = toc_detector_single_page("some text", model="gpt-4")
            assert result == "yes"


class TestGenerateTocRecovery:

    def test_generate_toc_init_returns_empty_on_exception(self):
        with patch("pageindex.page_index.llm_completion", side_effect=Exception("API error")):
            result = generate_toc_init("some text", model="gpt-4")
            assert result == []

    def test_generate_toc_continue_returns_empty_on_exception(self):
        with patch("pageindex.page_index.llm_completion", side_effect=Exception("API error")):
            result = generate_toc_continue([], "some text", model="gpt-4")
            assert result == []

    def test_generate_toc_init_returns_empty_on_length(self):
        def mock_completion(*args, return_finish_reason=False, **kwargs):
            if return_finish_reason:
                return '{"structure": "1", "title": "Test"', "length"
            return '{"structure": "1", "title": "Test"'

        with patch("pageindex.page_index.llm_completion", side_effect=mock_completion):
            result = generate_toc_init("some text", model="gpt-4")
            assert result == []


class TestAddPageNumberRecovery:

    def test_handles_non_list_json(self):
        with patch("pageindex.page_index.llm_completion", return_value='{"error": "something went wrong"}'):
            result = add_page_number_to_toc("text", [], model="gpt-4")
            assert result == []

    def test_handles_valid_list(self):
        response = (
            '[{"structure": "1", "title": "Test", "physical_index": '
            '"<physical_index_5>", "start": "yes"}]'
        )
        with patch("pageindex.page_index.llm_completion", return_value=response):
            result = add_page_number_to_toc("text", [], model="gpt-4")
            assert isinstance(result, list)
            assert "start" not in result[0]
