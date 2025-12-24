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

    def load_lora_adapter(self, adapter_path: str, fuse: bool = True) -> None:
        """
        Load a LoRA adapter into the draft model.

        Loads saved LoRA weights and either:
        - fuse=True (default): Directly fuses weights into base model (efficient inference)
        - fuse=False: Wraps target layers with LoRALinear for continued training

        Args:
            adapter_path: Path to the LoRA adapter directory (must contain
                          adapters.safetensors and adapter_config.json)
            fuse: If True (default), immediately fuse LoRA weights into base
                  model for efficient inference. If False, keep LoRA layers
                  (useful for continued training).
        """
        import json
        from .training import LoRAConfig, apply_lora_to_model

        if self.draft_model is None:
            raise RuntimeError("Draft model must be loaded first")

        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

        config_path = adapter_path / "adapter_config.json"
        weights_path = adapter_path / "adapters.safetensors"

        if not config_path.exists():
            raise FileNotFoundError(
                f"adapter_config.json not found in {adapter_path}. "
                "Please ensure the adapter was saved in MLX-LM compatible format."
            )

        if not weights_path.exists():
            raise FileNotFoundError(
                f"adapters.safetensors not found in {adapter_path}."
            )

        logger.info(f"Loading LoRA adapter from: {adapter_path}")

        # Load adapter config
        with open(config_path, "r") as f:
            adapter_config = json.load(f)

        # Load adapter weights
        lora_weights = mx.load(str(weights_path))

        # Normalize weight names (handle both lora_A/lora_B and lora_a/lora_b)
        normalized_weights = {}
        for key, value in lora_weights.items():
            # Convert MLX-LM format (lora_a, lora_b) to our format (lora_A, lora_B)
            new_key = key.replace(".lora_a", ".lora_A").replace(".lora_b", ".lora_B")
            normalized_weights[new_key] = value

        if fuse:
            # Directly fuse LoRA weights into base model
            self._fuse_lora_weights_direct(normalized_weights, adapter_config)
        else:
            # Apply LoRA wrappers and load weights for continued training
            lora_params = adapter_config.get("lora_parameters", {})
            config = LoRAConfig(
                rank=lora_params.get("rank", 8),
                alpha=lora_params.get("scale", 16),  # MLX-LM uses 'scale'
                dropout=lora_params.get("dropout", 0.0),
            )
            self.draft_model = apply_lora_to_model(self.draft_model, config)
            self._load_lora_weights_into_model(normalized_weights)

        mx.eval(self.draft_model.parameters())
        logger.info(f"LoRA adapter loaded successfully (fused={fuse})")

    def _fuse_lora_weights_direct(
        self, lora_weights: Dict[str, mx.array], adapter_config: Dict[str, Any]
    ) -> None:
        """
        Directly fuse LoRA weights into the base model without LoRA wrappers.

        This is more efficient as it doesn't create intermediate LoRALinear layers.

        Args:
            lora_weights: Dictionary of LoRA weight tensors (lora_A, lora_B)
            adapter_config: Adapter configuration with scaling info
        """
        lora_params = adapter_config.get("lora_parameters", {})
        rank = lora_params.get("rank", 8)
        alpha = lora_params.get("scale", 16)  # MLX-LM uses 'scale' not 'alpha'
        scaling = alpha / rank

        # Group lora_A and lora_B weights by layer path
        lora_pairs = {}
        for key, value in lora_weights.items():
            if ".lora_A" in key:
                base_path = key.replace(".lora_A", "")
                if base_path not in lora_pairs:
                    lora_pairs[base_path] = {}
                lora_pairs[base_path]["A"] = value
            elif ".lora_B" in key:
                base_path = key.replace(".lora_B", "")
                if base_path not in lora_pairs:
                    lora_pairs[base_path] = {}
                lora_pairs[base_path]["B"] = value

        # Fuse each LoRA pair into the corresponding layer
        fused_count = 0
        for layer_path, pair in lora_pairs.items():
            if "A" not in pair or "B" not in pair:
                logger.warning(f"Incomplete LoRA pair for {layer_path}, skipping")
                continue

            # Navigate to the layer
            try:
                parts = layer_path.split(".")
                module = self.draft_model
                for part in parts[:-1]:
                    try:
                        # Try to treat as integer index first
                        idx = int(part)
                        module = module[idx]
                    except (ValueError, TypeError):
                        # Fall back to attribute access
                        module = getattr(module, part)
                layer_name = parts[-1]
                try:
                    # Try to treat as integer index first
                    idx = int(layer_name)
                    layer = module[idx]
                except (ValueError, TypeError):
                    # Fall back to attribute access
                    layer = getattr(module, layer_name)
            except (AttributeError, IndexError, KeyError) as e:
                logger.warning(f"Could not find layer {layer_path}: {e}")
                continue

            # Compute delta: (B @ A) * scaling
            lora_A = pair["A"]  # (rank, in_features)
            lora_B = pair["B"]  # (out_features, rank)
            delta = (lora_B @ lora_A) * scaling

            # Handle quantized layers
            if hasattr(layer, "scales"):
                # Dequantize, add delta, re-quantize
                weight = mx.dequantize(
                    layer.weight,
                    layer.scales,
                    layer.biases,
                    layer.group_size,
                    layer.bits,
                )
                dtype = layer.scales.dtype
                fused_weight = weight + delta.astype(dtype)

                # Re-quantize
                has_bias = hasattr(layer, "bias") and layer.bias is not None
                out_features, in_features = fused_weight.shape
                temp_linear = nn.Linear(in_features, out_features, bias=has_bias)
                temp_linear.weight = fused_weight
                if has_bias:
                    temp_linear.bias = layer.bias

                new_layer = nn.QuantizedLinear.from_linear(
                    temp_linear, layer.group_size, layer.bits
                )
            else:
                # Regular linear layer
                dtype = layer.weight.dtype
                new_weight = layer.weight + delta.astype(dtype)

                has_bias = hasattr(layer, "bias") and layer.bias is not None
                out_features, in_features = new_weight.shape
                new_layer = nn.Linear(in_features, out_features, bias=has_bias)
                new_layer.weight = new_weight
                if has_bias:
                    new_layer.bias = layer.bias

            # Set the new layer
            try:
                # Try to treat as integer index first
                idx = int(layer_name)
                module[idx] = new_layer
            except (ValueError, TypeError):
                # Fall back to attribute access
                setattr(module, layer_name, new_layer)
            fused_count += 1

        # Validate fusion results
        total_pairs = len(lora_pairs)
        skipped = total_pairs - fused_count

        if fused_count == 0:
            raise ValueError(
                f"No LoRA layers were fused from {total_pairs} pairs. Check adapter format."
            )

        if skipped > 0:
            logger.warning(
                f"Fusion complete: {fused_count}/{total_pairs} layers fused, {skipped} skipped"
            )
        else:
            logger.info(f"Fused {fused_count} LoRA layers directly into base model")

    def _load_lora_weights_into_model(self, lora_weights: Dict[str, mx.array]) -> None:
        """
        Load LoRA weights into existing LoRALinear layers in the model.

        Args:
            lora_weights: Dictionary of LoRA weight tensors
        """
        from .training import LoRALinear

        loaded_count = 0
        for name, module in self.draft_model.named_modules():
            if isinstance(module, LoRALinear):
                lora_a_key = f"{name}.lora_A"
                lora_b_key = f"{name}.lora_B"

                if lora_a_key in lora_weights:
                    module.lora_A = lora_weights[lora_a_key]
                    loaded_count += 1
                if lora_b_key in lora_weights:
                    module.lora_B = lora_weights[lora_b_key]

        logger.info(f"Loaded weights into {loaded_count} LoRA layers")

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
            k: v
            for k, v in self.draft_model.parameters().items()
            if "lora" in k.lower()
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
        draft_memory = draft_params * 2 / (1024**3)  # FP16

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
        sorted_logits = mx.where(cutoff_mask, sorted_logits, mx.array(float("-inf")))

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
    of KVCache objects for each layer).

    Args:
        model: The language model

    Returns:
        Initialized empty KV cache (list of KVCache objects for MLX-LM models)
    """
    # Try MLX-LM's make_prompt_cache first (works with mlx_lm.load models)
    try:
        from mlx_lm.models.cache import make_prompt_cache

        return make_prompt_cache(model)
    except ImportError:
        pass

    # MLX models may have their own make_cache method
    if hasattr(model, "make_cache"):
        return model.make_cache()

    # Fallback: create cache manually based on model structure
    # Most transformer models have a layers attribute
    if hasattr(model, "layers"):
        num_layers = len(model.layers)
        # Return list of empty KVCache objects
        try:
            from mlx_lm.models.cache import KVCache

            return [KVCache() for _ in range(num_layers)]
        except ImportError:
            # Last resort - return list of None (will be populated during forward pass)
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
    if hasattr(layer_cache, "offset"):
        # MLX-LM uses offset tracking, adjust it
        layer_cache.offset = position
        if hasattr(layer_cache, "keys") and layer_cache.keys is not None:
            if layer_cache.keys.ndim == 4:
                layer_cache.keys = layer_cache.keys[:, :, :position, :]
                layer_cache.values = layer_cache.values[:, :, :position, :]
            elif layer_cache.keys.ndim == 3:
                layer_cache.keys = layer_cache.keys[:, :position, :]
                layer_cache.values = layer_cache.values[:, :position, :]
        return layer_cache

    # Handle object with keys/values attributes (but no offset)
    if hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
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

    # PRIORITY 1: Handle MLX-LM's KVCache with offset tracking
    # This MUST be checked first because KVCache pre-allocates keys/values
    # to a larger size (e.g., 256), but offset tracks actual content length
    if hasattr(layer_cache, "offset"):
        return layer_cache.offset

    # PRIORITY 2: Handle tuple of (keys, values)
    if isinstance(layer_cache, tuple) and len(layer_cache) >= 1:
        keys = layer_cache[0]
        if keys is not None:
            if keys.ndim == 4:
                return keys.shape[2]  # [batch, heads, seq, dim]
            elif keys.ndim == 3:
                return keys.shape[1]  # [batch, seq, hidden]

    # PRIORITY 3: Handle object with keys attribute but no offset
    if hasattr(layer_cache, "keys") and layer_cache.keys is not None:
        if layer_cache.keys.ndim == 4:
            return layer_cache.keys.shape[2]
        elif layer_cache.keys.ndim == 3:
            return layer_cache.keys.shape[1]

    return 0
