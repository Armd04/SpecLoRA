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
from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache, KVCache
from mlx_lm.utils import get_model_path
from huggingface_hub import snapshot_download

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
    
    def create_kv_cache(self, model: nn.Module) -> KVCache:
        """
        Create a KV cache for the given model.
        
        Args:
            model: The model to create cache for
            
        Returns:
            KVCache object for efficient generation
        """
        return make_prompt_cache(model)
    
    def clear_cache(self) -> None:
        """Clear MLX memory cache to free up RAM."""
        mx.metal.clear_cache()
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
            return sum(p.size for p in model.parameters().values())
        
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
        logits = logits.at[sorted_indices].set(sorted_logits)
    
    # Sample from distribution
    probs = mx.softmax(logits)
    token = mx.random.categorical(logits)
    
    return token


def get_logits(
    model: nn.Module,
    input_ids: mx.array,
    cache: Optional[KVCache] = None,
) -> Tuple[mx.array, Optional[KVCache]]:
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
