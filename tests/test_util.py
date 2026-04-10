"""Tests for abliterix.util — helper functions."""

from abliterix.util import (
    chunk_batches,
    flush_memory,
    humanize_duration,
    slugify_model_name,
)


# ---------------------------------------------------------------------------
# slugify_model_name
# ---------------------------------------------------------------------------


def test_slugify_simple():
    # '.' is not alphanumeric, so it becomes '--'
    assert slugify_model_name("Qwen-0.8B") == "Qwen-0--8B"


def test_slugify_slash():
    assert slugify_model_name("Qwen/Qwen3.5-0.8B") == "Qwen--Qwen3--5-0--8B"


def test_slugify_preserves_alphanumeric():
    assert slugify_model_name("abc123") == "abc123"


def test_slugify_replaces_special_chars():
    result = slugify_model_name("a/b c:d")
    assert "/" not in result
    assert " " not in result
    assert ":" not in result


# ---------------------------------------------------------------------------
# chunk_batches
# ---------------------------------------------------------------------------


def test_chunk_exact_division():
    result = chunk_batches([1, 2, 3, 4], 2)
    assert result == [[1, 2], [3, 4]]


def test_chunk_remainder():
    result = chunk_batches([1, 2, 3, 4, 5], 2)
    assert result == [[1, 2], [3, 4], [5]]


def test_chunk_single_batch():
    result = chunk_batches([1, 2, 3], 10)
    assert result == [[1, 2, 3]]


def test_chunk_empty():
    result = chunk_batches([], 5)
    assert result == []


def test_chunk_size_one():
    result = chunk_batches([1, 2, 3], 1)
    assert result == [[1], [2], [3]]


# ---------------------------------------------------------------------------
# humanize_duration
# ---------------------------------------------------------------------------


def test_humanize_seconds():
    assert humanize_duration(45) == "45s"


def test_humanize_minutes():
    assert humanize_duration(125) == "2m 5s"


def test_humanize_hours():
    assert humanize_duration(3661) == "1h 1m"


def test_humanize_zero():
    assert humanize_duration(0) == "0s"


# ---------------------------------------------------------------------------
# flush_memory
# ---------------------------------------------------------------------------


def test_flush_memory_no_error():
    """flush_memory should not raise even without GPU."""
    flush_memory()
