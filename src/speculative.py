"""
Speculative Decoding Implementation for MLX

Speculative decoding is an inference optimization technique where:
1. A small "draft" model generates K candidate tokens quickly
2. A large "target" model verifies all K tokens in parallel (single forward pass)
3. Accepted tokens are kept; first rejected token triggers re-sampling

This provides speedups when:
- Draft model is much faster than target model
- Draft model has high acceptance rate (agrees with target often)

The key insight is that verifying K tokens in parallel takes roughly the same
time as generating 1 token, due to GPU parallelism.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import mlx.core as mx
import mlx.nn as nn

from .models import sample_token, get_logits

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
    Implements speculative decoding with acceptance tracking.
    
    Algorithm:
    1. Generate K tokens with draft model
    2. Run target model on all K+1 positions in parallel
    3. Compare draft and target predictions:
       - If draft token matches target: ACCEPT
       - If draft token differs: REJECT, sample from target
    4. All tokens after first rejection are discarded
    5. Repeat until max_tokens or EOS
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
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.num_draft_tokens = num_draft_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.acceptance_threshold = acceptance_threshold
        
        # Get EOS token
        self.eos_token_id = tokenizer.eos_token_id
        if self.eos_token_id is None:
            self.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
    
    def _draft_tokens(
        self,
        input_ids: mx.array,
        num_tokens: int,
    ) -> Tuple[List[int], List[mx.array]]:
        """
        Generate draft tokens using the small model.
        
        Args:
            input_ids: Current token sequence
            num_tokens: Number of tokens to draft
            
        Returns:
            Tuple of (draft_token_ids, draft_logits_list)
        """
        draft_tokens = []
        draft_logits = []
        current_ids = input_ids
        
        for _ in range(num_tokens):
            # Get logits from draft model
            logits, _ = get_logits(self.draft_model, current_ids)
            last_logits = logits[0, -1, :]  # [vocab_size]
            
            # Sample token
            token = sample_token(
                last_logits,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            
            token_id = token.item()
            draft_tokens.append(token_id)
            draft_logits.append(last_logits)
            
            # Stop if EOS
            if token_id == self.eos_token_id:
                break
            
            # Append token for next iteration
            current_ids = mx.concatenate([
                current_ids,
                mx.array([[token_id]])
            ], axis=1)
        
        return draft_tokens, draft_logits
    
    def _verify_tokens(
        self,
        input_ids: mx.array,
        draft_tokens: List[int],
    ) -> Tuple[List[int], int, List[mx.array]]:
        """
        Verify draft tokens using the target model.
        
        This is the key efficiency: we verify ALL draft tokens in a single
        forward pass by running the target model on the full sequence.
        
        Args:
            input_ids: Original input sequence (before drafting)
            draft_tokens: Tokens proposed by draft model
            
        Returns:
            Tuple of (accepted_tokens, num_accepted, target_logits)
        """
        if not draft_tokens:
            return [], 0, []
        
        # Create full sequence with draft tokens
        draft_tensor = mx.array([draft_tokens])
        full_sequence = mx.concatenate([input_ids, draft_tensor], axis=1)
        
        # Run target model on full sequence (parallel verification)
        target_logits, _ = get_logits(self.target_model, full_sequence)
        
        # Verify each draft token
        accepted_tokens = []
        target_logits_list = []
        num_accepted = 0
        
        # Starting position: where draft tokens begin
        start_pos = input_ids.shape[1] - 1
        
        for i, draft_token in enumerate(draft_tokens):
            # Get target's prediction at this position
            pos = start_pos + i
            logits_at_pos = target_logits[0, pos, :]
            target_logits_list.append(logits_at_pos)
            
            # Sample what target would have generated
            target_token = sample_token(
                logits_at_pos,
                temperature=self.temperature,
                top_p=self.top_p,
            ).item()
            
            # Check if draft matches target
            if draft_token == target_token:
                accepted_tokens.append(draft_token)
                num_accepted += 1
                
                # Stop if EOS
                if draft_token == self.eos_token_id:
                    break
            else:
                # Rejection: use target's token instead
                accepted_tokens.append(target_token)
                break
        
        # If all draft tokens accepted, sample one more from target
        if num_accepted == len(draft_tokens) and draft_tokens[-1] != self.eos_token_id:
            final_logits = target_logits[0, -1, :]
            target_logits_list.append(final_logits)
            bonus_token = sample_token(
                final_logits,
                temperature=self.temperature,
                top_p=self.top_p,
            ).item()
            accepted_tokens.append(bonus_token)
        
        return accepted_tokens, num_accepted, target_logits_list
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        collect_training_data: bool = True,
    ) -> SpeculativeResult:
        """
        Generate text using speculative decoding.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            collect_training_data: Whether to collect draft/target outputs
            
        Returns:
            SpeculativeResult with generated text and metrics
        """
        start_time = time.time()
        metrics = GenerationMetrics()
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        input_ids = mx.array(input_ids)
        
        generated_tokens = []
        draft_all_outputs = [] if collect_training_data else None
        target_all_outputs = [] if collect_training_data else None
        
        while len(generated_tokens) < max_tokens:
            # Current sequence
            if generated_tokens:
                current_ids = mx.concatenate([
                    input_ids,
                    mx.array([generated_tokens])[None, :]
                ], axis=1)
            else:
                current_ids = input_ids
            
            # Phase 1: Draft K tokens
            draft_start = time.time()
            draft_tokens, draft_logits = self._draft_tokens(
                current_ids,
                self.num_draft_tokens,
            )
            mx.eval(draft_tokens)  # Force evaluation
            metrics.draft_time_seconds += time.time() - draft_start
            
            if not draft_tokens:
                break
            
            metrics.draft_tokens_proposed += len(draft_tokens)
            
            # Collect draft outputs for training
            if collect_training_data:
                draft_all_outputs.extend(draft_tokens)
            
            # Phase 2: Verify with target model
            verify_start = time.time()
            accepted, num_accepted, target_logits = self._verify_tokens(
                current_ids,
                draft_tokens,
            )
            mx.eval(accepted)  # Force evaluation
            metrics.verify_time_seconds += time.time() - verify_start
            
            metrics.draft_tokens_accepted += num_accepted
            
            # Calculate step acceptance rate
            if draft_tokens:
                step_rate = num_accepted / len(draft_tokens)
                metrics.step_acceptance_rates.append(step_rate)
            
            # Collect target outputs for training
            if collect_training_data and target_logits:
                # Sample from target logits to get what target would output
                for logits in target_logits[:len(draft_tokens)]:
                    target_token = sample_token(
                        logits,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    ).item()
                    target_all_outputs.append(target_token)
            
            # Add accepted tokens to output
            generated_tokens.extend(accepted)
            metrics.total_tokens_generated += len(accepted)
            
            # Check for EOS
            if accepted and accepted[-1] == self.eos_token_id:
                break
        
        metrics.total_time_seconds = time.time() - start_time
        
        # Decode generated text
        generated_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )
        
        # Determine if this is a failure case
        is_failure = metrics.acceptance_rate < self.acceptance_threshold
        
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
        Generate text using standard (non-speculative) decoding.
        
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
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        input_ids = mx.array(input_ids)
        
        generated_tokens = []
        
        for _ in range(max_tokens):
            if generated_tokens:
                current_ids = mx.concatenate([
                    input_ids,
                    mx.array([generated_tokens])[None, :]
                ], axis=1)
            else:
                current_ids = input_ids
            
            logits, _ = get_logits(model, current_ids)
            last_logits = logits[0, -1, :]
            
            token = sample_token(
                last_logits,
                temperature=self.temperature,
                top_p=self.top_p,
            ).item()
            
            generated_tokens.append(token)
            
            if token == self.eos_token_id:
                break
        
        elapsed = time.time() - start_time
        
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return text, elapsed


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
