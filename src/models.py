"""
Model Loading and Management for Speculative Decoding

This module handles loading and managing both the target (large) and draft (small)
models using MLX. The target model is kept quantized (4-bit) while the draft model
can be loaded in FP16 for training.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx.utils import tree_flatten

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages loading and caching of target and draft models.
    
    The target model is the larger, more accurate model that verifies
    draft tokens. The draft model is smaller and faster, used to
    generate candidate tokens speculatively.
    """
    
    def __init__(
        self,
        target_model_name: str,
        draft_model_name: str,
        lora_path: Optional[str] = None,
    ):
        """
        Initialize the model manager.
        
        Args:
            target_model_name: HuggingFace model path for target model
            draft_model_name: HuggingFace model path for draft model
            lora_path: Optional path to LoRA adapter weights for draft model
        """
        self.target_model_name = target_model_name
        self.draft_model_name = draft_model_name
        self.lora_path = lora_path
        
        self.target_model = None
        self.target_tokenizer = None
        self.draft_model = None
        self.draft_tokenizer = None
        
        self._loaded = False
    
    def load_models(self) -> None:
        """
        Load both target and draft models into memory.
        
        This uses MLX's efficient memory mapping to minimize RAM usage.
        The target model should be 4-bit quantized.
        """
        logger.info(f"Loading target model: {self.target_model_name}")
        
        # Load target model (quantized)
        self.target_model, self.target_tokenizer = load(
            self.target_model_name,
            lazy=True,  # Lazy loading for memory efficiency
        )
        
        # Ensure model weights are evaluated
        mx.eval(self.target_model.parameters())
        
        logger.info(f"Loading draft model: {self.draft_model_name}")
        
        # Load draft model (can be FP16 for training)
        self.draft_model, self.draft_tokenizer = load(
            self.draft_model_name,
            lazy=True,
        )
        
        mx.eval(self.draft_model.parameters())
        
        # Load LoRA adapter if specified
        if self.lora_path and Path(self.lora_path).exists():
            self.load_lora_adapter(self.lora_path)
        
        self._loaded = True
        logger.info("Models loaded successfully")
    
    def load_lora_adapter(self, adapter_path: str) -> None:
        """
        Load a LoRA adapter into the draft model.
        
        Args:
            adapter_path: Path to the LoRA adapter weights
        """
        if self.draft_model is None:
            raise RuntimeError("Draft model must be loaded first")
        
        logger.info(f"Loading LoRA adapter from: {adapter_path}")
        
        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        
        # Load adapter weights
        adapter_weights = mx.load(str(adapter_path / "adapters.safetensors"))
        
        # Apply adapter weights to model
        self.draft_model.load_weights(list(adapter_weights.items()))
        mx.eval(self.draft_model.parameters())
        
        logger.info("LoRA adapter loaded successfully")
    
    def save_lora_adapter(self, save_path: str) -> None:
        """
        Save the current LoRA adapter weights.
        
        Args:
            save_path: Directory to save the adapter
        """
        if self.draft_model is None:
            raise RuntimeError("Draft model must be loaded first")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Extract LoRA weights (weights with 'lora' in name)
        lora_weights = {
            k: v for k, v in self.draft_model.parameters().items()
            if 'lora' in k.lower()
        }
        
        if lora_weights:
            mx.save_safetensors(str(save_path / "adapters.safetensors"), lora_weights)
            logger.info(f"LoRA adapter saved to: {save_path}")
        else:
            logger.warning("No LoRA weights found to save")
    
    def get_target_model(self) -> Tuple[nn.Module, Any]:
        """Get the target model and tokenizer."""
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self.target_model, self.target_tokenizer
    
    def get_draft_model(self) -> Tuple[nn.Module, Any]:
        """Get the draft model and tokenizer."""
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self.draft_model, self.draft_tokenizer
    
    def clear_cache(self) -> None:
        """Clear MLX memory cache to free up RAM."""
        mx.clear_cache()
        logger.debug("Memory cache cleared")
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size (should be same for both models)."""
        if not self._loaded:
            raise RuntimeError("Models not loaded")
        return self.target_tokenizer.vocab_size
    
    def estimate_memory_usage(self) -> Dict[str, float]:
        """
        Estimate memory usage of loaded models.
        
        Returns:
            Dictionary with memory estimates in GB
        """
        def count_params(model):
            total_params = 0
            for _, v in tree_flatten(model.parameters()):
                if hasattr(v, "size"):
                    # Handle 4-bit quantized weights (packed into uint32)
                    if v.dtype == mx.uint32:
                        total_params += v.size * 8
                    else:
                        total_params += v.size
            return total_params
        
        target_params = count_params(self.target_model) if self.target_model else 0
        draft_params = count_params(self.draft_model) if self.draft_model else 0
        
        # Rough estimates: 4-bit = 0.5 bytes, FP16 = 2 bytes per param
        target_memory = target_params * 0.5 / (1024**3)  # 4-bit quantized
        draft_memory = draft_params * 2 / (1024**3)      # FP16
        
        return {
            "target_model_gb": target_memory,
            "draft_model_gb": draft_memory,
            "total_gb": target_memory + draft_memory,
            "target_params_m": target_params / 1e6,
            "draft_params_m": draft_params / 1e6,
        }


def sample_token(
    logits: mx.array,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> mx.array:
    """
    Sample a token from logits using temperature and top-p sampling.
    
    Args:
        logits: Model output logits [vocab_size] or [1, vocab_size]
        temperature: Sampling temperature (0 = greedy)
        top_p: Nucleus sampling threshold
        
    Returns:
        Sampled token ID
    """
    # Handle batch dimension
    if logits.ndim == 2:
        logits = logits[0]
    
    # Greedy sampling
    if temperature == 0:
        return mx.argmax(logits)
    
    # Apply temperature
    logits = logits / temperature
    
    # Top-p (nucleus) sampling
    if top_p < 1.0:
        # Sort logits in descending order
        sorted_indices = mx.argsort(-logits)
        sorted_logits = logits[sorted_indices]
        
        # Compute cumulative probabilities
        probs = mx.softmax(sorted_logits)
        cumsum_probs = mx.cumsum(probs)
        
        # Find cutoff index
        cutoff_mask = cumsum_probs <= top_p
        # Always include at least one token
        cutoff_mask = mx.concatenate([mx.array([True]), cutoff_mask[:-1]])
        
        # Zero out tokens beyond cutoff
        sorted_logits = mx.where(cutoff_mask, sorted_logits, mx.array(float('-inf')))
        
        # Restore original order
        logits = mx.zeros_like(logits)
        logits[sorted_indices] = sorted_logits
    
    # Sample from distribution
    probs = mx.softmax(logits)
    token = mx.random.categorical(logits)
    
    return token


def get_logits(
    model: nn.Module,
    input_ids: mx.array,
    cache: Optional[Any] = None,
) -> Tuple[mx.array, Optional[Any]]:
    """
    Get logits from a model for the given input.
    
    Args:
        model: The language model
        input_ids: Input token IDs [seq_len] or [batch, seq_len]
        cache: Optional KV cache for efficient generation
        
    Returns:
        Tuple of (logits, updated_cache)
    """
    # Ensure proper shape [batch, seq_len]
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]
    
    if cache is not None:
        logits = model(input_ids, cache=cache)
    else:
        logits = model(input_ids)
    
    return logits, cache


# ============================================================================
# KV Cache Management Helpers for Manual Speculative Decoding
# ============================================================================

def create_kv_cache(model: nn.Module) -> Any:
    """
    Initialize an empty KV cache for a model.
    
    This creates a fresh cache that can be used for autoregressive generation.
    The cache structure depends on the model architecture (typically a list
    of tuples containing key and value tensors for each layer).
    
    Args:
        model: The language model (must have make_cache method)
        
    Returns:
        Initialized empty KV cache
    """
    # MLX models typically have a make_cache method
    if hasattr(model, 'make_cache'):
        return model.make_cache()
    
    # Fallback: create cache manually based on model structure
    # Most transformer models have a layers attribute
    if hasattr(model, 'layers'):
        num_layers = len(model.layers)
        # Return empty cache list - will be populated during forward pass
        return [None] * num_layers
    
    # If model doesn't support caching, return None
    logger.warning("Model doesn't support KV caching")
    return None


def get_logits_with_cache(
    model: nn.Module,
    tokens: mx.array,
    cache: Optional[Any] = None,
) -> Tuple[mx.array, Any]:
    """
    Forward pass that returns logits and updates the KV cache.
    
    This function handles the model forward pass with proper cache management.
    It's designed for use in autoregressive generation where we want to
    maintain and update the KV cache across generation steps.
    
    Args:
        model: The language model
        tokens: Input token IDs [seq_len] or [batch, seq_len]
        cache: Optional existing KV cache to use and update
        
    Returns:
        Tuple of (logits, updated_cache)
        - logits: Shape [batch, seq_len, vocab_size]
        - updated_cache: Cache with new key/value pairs appended
    """
    # Ensure proper shape [batch, seq_len]
    if tokens.ndim == 1:
        tokens = tokens[None, :]
    
    if cache is not None:
        # Forward pass with cache - cache is updated in-place for most MLX models
        logits = model(tokens, cache=cache)
    else:
        # First pass without cache - model may return cache
        logits = model(tokens)
    
    return logits, cache


def rewind_cache(cache: Any, position: int) -> Any:
    """
    Truncate the KV cache to the specified position.
    
    When speculative decoding rejects a draft token, we need to "rewind"
    the cache to the point of rejection, discarding cached values for
    rejected tokens.
    
    Args:
        cache: The KV cache to truncate
        position: Position to truncate to (exclusive, keeps 0 to position-1)
        
    Returns:
        Truncated cache (may be same object if modified in-place)
    """
    if cache is None:
        return None
    
    # Handle list of layer caches (most common format)
    if isinstance(cache, list):
        for i, layer_cache in enumerate(cache):
            if layer_cache is not None:
                cache[i] = _rewind_layer_cache(layer_cache, position)
        return cache
    
    # Handle single cache object (for simpler models)
    return _rewind_layer_cache(cache, position)


def _rewind_layer_cache(layer_cache: Any, position: int) -> Any:
    """
    Rewind a single layer's KV cache to the specified position.
    
    Args:
        layer_cache: Cache for one transformer layer (typically tuple of K, V)
        position: Position to truncate to
        
    Returns:
        Truncated layer cache
    """
    if layer_cache is None:
        return None
    
    # Handle tuple of (keys, values) - most common format
    if isinstance(layer_cache, tuple) and len(layer_cache) == 2:
        keys, values = layer_cache
        # Keys and values typically have shape [batch, num_heads, seq_len, head_dim]
        # or [batch, seq_len, num_heads, head_dim]
        if keys is not None and values is not None:
            # Determine sequence dimension (usually 2 or 1)
            if keys.ndim == 4:
                # [batch, num_heads, seq_len, head_dim] format
                truncated_keys = keys[:, :, :position, :]
                truncated_values = values[:, :, :position, :]
            elif keys.ndim == 3:
                # [batch, seq_len, hidden] format
                truncated_keys = keys[:, :position, :]
                truncated_values = values[:, :position, :]
            else:
                # Unknown format, return as-is
                return layer_cache
            return (truncated_keys, truncated_values)

    # Handle MLX-LM's KVCache object which has update/rewind methods
    # IMPORTANT: Check this BEFORE checking for keys/values alone, since MLX-LM's
    # KVCache objects have all three attributes (keys, values, AND offset)
    if hasattr(layer_cache, 'offset'):
        # MLX-LM uses offset tracking, adjust it
        layer_cache.offset = position
        if hasattr(layer_cache, 'keys') and layer_cache.keys is not None:
            if layer_cache.keys.ndim == 4:
                layer_cache.keys = layer_cache.keys[:, :, :position, :]
                layer_cache.values = layer_cache.values[:, :, :position, :]
            elif layer_cache.keys.ndim == 3:
                layer_cache.keys = layer_cache.keys[:, :position, :]
                layer_cache.values = layer_cache.values[:, :position, :]
        return layer_cache

    # Handle object with keys/values attributes (but no offset)
    if hasattr(layer_cache, 'keys') and hasattr(layer_cache, 'values'):
        if layer_cache.keys is not None:
            if layer_cache.keys.ndim == 4:
                layer_cache.keys = layer_cache.keys[:, :, :position, :]
                layer_cache.values = layer_cache.values[:, :, :position, :]
            elif layer_cache.keys.ndim == 3:
                layer_cache.keys = layer_cache.keys[:, :position, :]
                layer_cache.values = layer_cache.values[:, :position, :]

    return layer_cache


def get_cache_length(cache: Any) -> int:
    """
    Get the current sequence length stored in the cache.
    
    Args:
        cache: The KV cache
        
    Returns:
        Number of positions currently cached
    """
    if cache is None:
        return 0
    
    # Handle list of layer caches
    if isinstance(cache, list):
        for layer_cache in cache:
            if layer_cache is not None:
                length = _get_layer_cache_length(layer_cache)
                if length > 0:
                    return length
        return 0
    
    return _get_layer_cache_length(cache)


def _get_layer_cache_length(layer_cache: Any) -> int:
    """Get length from a single layer's cache."""
    if layer_cache is None:
        return 0
    
    # Handle tuple of (keys, values)
    if isinstance(layer_cache, tuple) and len(layer_cache) >= 1:
        keys = layer_cache[0]
        if keys is not None:
            if keys.ndim == 4:
                return keys.shape[2]  # [batch, heads, seq, dim]
            elif keys.ndim == 3:
                return keys.shape[1]  # [batch, seq, hidden]
    
    # Handle object with keys attribute
    if hasattr(layer_cache, 'keys') and layer_cache.keys is not None:
        if layer_cache.keys.ndim == 4:
            return layer_cache.keys.shape[2]
        elif layer_cache.keys.ndim == 3:
            return layer_cache.keys.shape[1]
    
    # Handle MLX-LM's offset tracking
    if hasattr(layer_cache, 'offset'):
        return layer_cache.offset
    
    return 0
