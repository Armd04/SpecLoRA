"""
Speculative Decoding Implementation using MLX-LM

This module wraps MLX-LM's built-in speculative decoding with tracking
for acceptance rates, failure cases, and training data collection.

Uses MLX-LM's optimized implementation which includes proper KV caching,
cache rewinding on rejections, and efficient parallel verification.

For detailed data collection (token-level disagreements), use the
ManualSpeculativeDecoder from speculative_manual.py instead.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable, TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import stream_generate, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler

if TYPE_CHECKING:
    from .speculative_manual import ManualSpeculativeDecoder, ManualSpeculativeResult

logger = logging.getLogger(__name__)


@dataclass
class GenerationMetrics:
    """Tracks metrics during speculative decoding generation."""
    
    # Token counts
    total_tokens_generated: int = 0
    draft_tokens_proposed: int = 0
    draft_tokens_accepted: int = 0
    
    # Timing
    total_time_seconds: float = 0.0
    draft_time_seconds: float = 0.0
    verify_time_seconds: float = 0.0
    
    # Per-step acceptance rates
    step_acceptance_rates: List[float] = field(default_factory=list)
    
    @property
    def acceptance_rate(self) -> float:
        """Overall acceptance rate of draft tokens."""
        if self.draft_tokens_proposed == 0:
            return 0.0
        return self.draft_tokens_accepted / self.draft_tokens_proposed
    
    @property
    def tokens_per_second(self) -> float:
        """Generation speed in tokens per second."""
        if self.total_time_seconds == 0:
            return 0.0
        return self.total_tokens_generated / self.total_time_seconds
    
    @property
    def speedup_factor(self) -> float:
        """
        Estimated speedup vs standard decoding.
        
        In standard decoding, each token requires a full forward pass.
        With speculative decoding, we do (1 draft pass + 1 verify pass)
        to potentially generate multiple tokens.
        """
        if self.draft_tokens_proposed == 0:
            return 1.0
        # Simplified estimate: speedup ≈ avg_accepted + 1 per iteration
        avg_accepted = self.acceptance_rate * 4  # Assuming K=4
        return max(1.0, avg_accepted + 1) / 2  # Divide by 2 for draft+verify
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            "total_tokens": self.total_tokens_generated,
            "draft_proposed": self.draft_tokens_proposed,
            "draft_accepted": self.draft_tokens_accepted,
            "acceptance_rate": self.acceptance_rate,
            "tokens_per_second": self.tokens_per_second,
            "total_time": self.total_time_seconds,
        }


@dataclass
class SpeculativeResult:
    """Result from a speculative decoding generation."""
    
    # Generated text
    text: str
    
    # Token IDs generated
    tokens: List[int]
    
    # Generation metrics
    metrics: GenerationMetrics
    
    # Input prompt (for failure case collection)
    prompt: str
    
    # Whether this is considered a "failure" case (low acceptance)
    is_failure_case: bool = False
    
    # Draft model's outputs (for training)
    draft_outputs: Optional[List[int]] = None
    
    # Target model's outputs (for training)
    target_outputs: Optional[List[int]] = None


class SpeculativeDecoder:
    """
    Wrapper around MLX-LM's speculative decoding with tracking and metrics.

    Uses MLX-LM's optimized implementation which properly handles:
    - KV cache creation and management
    - Cache rewinding on rejections
    - Parallel token verification
    - Memory efficiency
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        tokenizer: Any,
        num_draft_tokens: int = 4,
        temperature: float = 0.7,
        top_p: float = 0.9,
        acceptance_threshold: float = 0.5,
        chat_template: Optional[str] = None,
        system_message: Optional[str] = None,
        use_tokenizer_chat_template: bool = True,
    ):
        """
        Initialize the speculative decoder.

        Args:
            target_model: Large target model for verification
            draft_model: Small draft model for speculation
            tokenizer: Tokenizer (shared between models)
            num_draft_tokens: Number of tokens to draft (K)
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            acceptance_threshold: Below this = failure case
            chat_template: Fallback chat template string with {system} and {prompt} placeholders
            system_message: System message for the chat template
            use_tokenizer_chat_template: Whether to use tokenizer's apply_chat_template (recommended)
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.num_draft_tokens = num_draft_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.acceptance_threshold = acceptance_threshold
        self.chat_template = chat_template
        self.system_message = system_message or "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        self.use_tokenizer_chat_template = use_tokenizer_chat_template

        # Check if tokenizer supports apply_chat_template
        self._has_chat_template = hasattr(tokenizer, 'apply_chat_template') and callable(tokenizer.apply_chat_template)
        if self.use_tokenizer_chat_template and not self._has_chat_template:
            logger.warning("Tokenizer doesn't support apply_chat_template, falling back to manual template")

        # Get EOS token
        self.eos_token_id = tokenizer.eos_token_id
        if self.eos_token_id is None:
            self.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")

    def _create_sampler(self) -> Callable[[mx.array], mx.array]:
        """
        Create a sampler function that applies temperature and top_p sampling.

        Returns:
            Callable that takes logits and returns a sampled token
        """
        return make_sampler(temp=self.temperature, top_p=self.top_p)

    def _format_prompt(self, prompt: str) -> str:
        """
        Format the prompt using the appropriate chat template.

        Uses the tokenizer's apply_chat_template if available (recommended for Qwen2.5),
        otherwise falls back to the manual template.

        Args:
            prompt: The user's input prompt

        Returns:
            Formatted prompt string ready for tokenization
        """
        # Try to use tokenizer's built-in chat template (recommended)
        if self.use_tokenizer_chat_template and self._has_chat_template:
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt},
            ]
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return formatted
            except Exception as e:
                logger.warning(f"apply_chat_template failed: {e}, falling back to manual template")

        # Fallback to manual template
        if self.chat_template:
            return self.chat_template.format(
                system=self.system_message,
                prompt=prompt
            )

        # Last resort: raw prompt
        return prompt

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        collect_training_data: bool = True,
    ) -> SpeculativeResult:
        """
        Generate text using MLX-LM's built-in speculative decoding.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            collect_training_data: Whether to collect draft/target outputs

        Returns:
            SpeculativeResult with generated text and metrics
        """
        start_time = time.time()
        metrics = GenerationMetrics()

        generated_tokens = []
        generated_text = ""
        draft_count = 0
        accepted_count = 0

        # Format prompt using the appropriate chat template
        formatted_prompt = self._format_prompt(prompt)

        # Create sampler with temperature and top_p
        sampler = self._create_sampler()

        # Use MLX-LM's stream_generate with built-in speculative decoding
        for response in stream_generate(
            model=self.target_model,
            tokenizer=self.tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            draft_model=self.draft_model,
            num_draft_tokens=self.num_draft_tokens,
            sampler=sampler,
        ):
            generated_tokens.append(response.token)
            generated_text += response.text

            # Track which tokens came from draft model
            if response.from_draft:
                accepted_count += 1
            draft_count += 1

        metrics.total_time_seconds = time.time() - start_time
        metrics.total_tokens_generated = len(generated_tokens)
        metrics.draft_tokens_accepted = accepted_count
        metrics.draft_tokens_proposed = draft_count  # Approximation

        # Determine if this is a failure case
        is_failure = metrics.acceptance_rate < self.acceptance_threshold

        # Note: We don't collect draft/target outputs in this implementation
        # since MLX-LM's generator doesn't expose them
        draft_all_outputs = None if not collect_training_data else []
        target_all_outputs = None if not collect_training_data else []

        # Clear memory
        mx.clear_cache()

        return SpeculativeResult(
            text=generated_text,
            tokens=generated_tokens,
            metrics=metrics,
            prompt=prompt,
            is_failure_case=is_failure,
            draft_outputs=draft_all_outputs,
            target_outputs=target_all_outputs,
        )
    
    def generate_standard(
        self,
        prompt: str,
        max_tokens: int = 256,
        use_target: bool = True,
    ) -> Tuple[str, float]:
        """
        Generate text using standard (non-speculative) decoding with MLX-LM.

        Useful for comparison/benchmarking.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            use_target: Use target model (True) or draft model (False)

        Returns:
            Tuple of (generated_text, time_seconds)
        """
        model = self.target_model if use_target else self.draft_model

        start_time = time.time()

        text = mlx_generate(
            model=model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )

        elapsed = time.time() - start_time

        mx.clear_cache()

        return text, elapsed
    
    def create_manual_decoder(self) -> "ManualSpeculativeDecoder":
        """
        Create a ManualSpeculativeDecoder with the same configuration.
        
        The manual decoder implements speculative decoding from scratch to
        capture detailed token-level disagreements for training. It's slower
        than the built-in version (~20%) but provides much more training signal.
        
        Use this for data collection runs, not production inference.
        
        Returns:
            ManualSpeculativeDecoder configured with same models and settings
        """
        from .speculative_manual import ManualSpeculativeDecoder
        
        return ManualSpeculativeDecoder(
            target_model=self.target_model,
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            num_draft_tokens=self.num_draft_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            acceptance_threshold=self.acceptance_threshold,
            system_message=self.system_message,
        )
    
    def generate_detailed(
        self,
        prompt: str,
        max_tokens: int = 256,
    ) -> "ManualSpeculativeResult":
        """
        Generate text using manual speculative decoding with token-level data.
        
        This method uses the ManualSpeculativeDecoder internally to capture
        detailed disagreement information. It's slower than generate() but
        provides more valuable training data.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            
        Returns:
            ManualSpeculativeResult with token-level disagreement data
        """
        manual_decoder = self.create_manual_decoder()
        return manual_decoder.generate_with_data_collection(prompt, max_tokens)


def run_acceptance_benchmark(
    decoder: SpeculativeDecoder,
    prompts: List[str],
    max_tokens: int = 128,
) -> Dict[str, Any]:
    """
    Run a benchmark to measure acceptance rates across prompts.
    
    Args:
        decoder: Configured speculative decoder
        prompts: List of test prompts
        max_tokens: Max tokens per generation
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        "per_prompt": [],
        "total_tokens": 0,
        "total_accepted": 0,
        "total_proposed": 0,
        "total_time": 0.0,
    }
    
    for prompt in prompts:
        result = decoder.generate(prompt, max_tokens=max_tokens)
        
        prompt_result = {
            "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "acceptance_rate": result.metrics.acceptance_rate,
            "tokens_generated": result.metrics.total_tokens_generated,
            "tokens_per_second": result.metrics.tokens_per_second,
            "is_failure": result.is_failure_case,
        }
        results["per_prompt"].append(prompt_result)
        
        results["total_tokens"] += result.metrics.total_tokens_generated
        results["total_accepted"] += result.metrics.draft_tokens_accepted
        results["total_proposed"] += result.metrics.draft_tokens_proposed
        results["total_time"] += result.metrics.total_time_seconds
    
    # Calculate aggregates
    if results["total_proposed"] > 0:
        results["overall_acceptance_rate"] = (
            results["total_accepted"] / results["total_proposed"]
        )
    else:
        results["overall_acceptance_rate"] = 0.0
    
    if results["total_time"] > 0:
        results["overall_tokens_per_second"] = (
            results["total_tokens"] / results["total_time"]
        )
    else:
        results["overall_tokens_per_second"] = 0.0
    
    return results
