"""
Unit tests for KV cache helpers used by manual speculative decoding.
"""

from pathlib import Path
import sys

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    create_kv_cache,
    get_logits_with_cache,
    rewind_cache,
    get_cache_length,
)


def test_create_kv_cache_prefers_model_helper():
    """Model-provided make_cache should be used when available."""

    class MockModel:
        def __init__(self):
            self.called = False

        def make_cache(self):
            self.called = True
            return {"cache": "created"}

    model = MockModel()

    cache = create_kv_cache(model)

    assert model.called is True
    assert cache == {"cache": "created"}


def test_create_kv_cache_falls_back_to_layers_list():
    """When make_cache is missing, fall back to layer-count list."""

    class MockLayer:
        pass

    class MockModel:
        def __init__(self):
            self.layers = [MockLayer(), MockLayer(), MockLayer()]

    model = MockModel()

    cache = create_kv_cache(model)

    assert isinstance(cache, list)
    assert len(cache) == 3
    assert all(entry is None for entry in cache)


def test_get_logits_with_cache_preserves_cache_and_shapes():
    """Forward pass should reuse cache and normalize token shape."""

    class MockModel:
        def __init__(self):
            self.calls = []

        def __call__(self, tokens, cache=None):
            self.calls.append((tokens, cache))
            return tokens  # Echo logits with same shape as tokens

    model = MockModel()
    tokens = mx.array([1, 2, 3])
    cache = {"state": "existing"}

    logits, returned_cache = get_logits_with_cache(model, tokens, cache)

    # Tokens should have been expanded to [1, seq_len]
    assert logits.shape == (1, 3)
    # Cache object should flow through unchanged
    assert returned_cache is cache
    # Model should have been called exactly once with normalized tokens
    assert len(model.calls) == 1
    called_tokens, called_cache = model.calls[0]
    assert called_tokens.shape == (1, 3)
    assert called_cache is cache


def test_rewind_cache_truncates_layer_data():
    """Rewinding should drop positions at and after the cutoff."""
    keys = mx.arange(16).reshape(1, 1, 4, 4)
    values = mx.arange(100, 116).reshape(1, 1, 4, 4)
    layer_cache = (keys, values)

    truncated = rewind_cache([layer_cache], position=2)

    # Should keep only the first two positions along the sequence dimension
    truncated_keys, truncated_values = truncated[0]
    assert truncated_keys.shape == (1, 1, 2, 4)
    assert truncated_values.shape == (1, 1, 2, 4)
    # Verify data actually truncated, not just shapes
    assert truncated_keys[0, 0, -1, 0] == keys[0, 0, 1, 0]


def test_get_cache_length_handles_common_structures():
    """Cache length should be derived from keys, values, or list wrappers."""
    tuple_cache = (
        mx.zeros((1, 1, 5, 2)),  # keys shape => length 5
        mx.zeros((1, 1, 5, 2)),
    )
    list_cache = [tuple_cache, None]

    # Tuple cache
    assert get_cache_length(tuple_cache) == 5
    # List of layer caches should report the first non-empty length
    assert get_cache_length(list_cache) == 5
    # Empty cache should report zero
    assert get_cache_length(None) == 0

