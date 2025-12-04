"""
Manual Speculative Decoding for Token-Level Data Collection

This module implements speculative decoding manually (rather than using MLX-LM's
built-in version) to capture detailed token-level disagreements between the
draft and target models.

The key difference from the built-in version:
- Built-in: Fast, optimized, but only exposes overall acceptance rates
- Manual: Slightly slower (~20%), but captures every token decision for training

This enables more effective LoRA training by focusing gradients on specific
positions where the draft model fails, rather than uniformly across sequences.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable

import mlx.core as mx
import mlx.nn as nn

from .data_collector import TokenLevelDisagreement
from .models import (
    create_kv_cache,
    get_logits_with_cache,
    rewind_cache,
    get_cache_length,
    sample_token,
)

logger = logging.getLogger(__name__)


@dataclass
class ManualSpeculativeMetrics:
    """Metrics from manual speculative decoding generation."""
    
    # Token counts
    total_tokens_generated: int = 0
    draft_tokens_proposed: int = 0
    draft_tokens_accepted: int = 0
    
    # Timing
    total_time_seconds: float = 0.0
    draft_time_seconds: float = 0.0
    verify_time_seconds: float = 0.0
    
    # Disagreement tracking
    num_disagreements: int = 0
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            "total_tokens": self.total_tokens_generated,
            "draft_proposed": self.draft_tokens_proposed,
            "draft_accepted": self.draft_tokens_accepted,
            "acceptance_rate": self.acceptance_rate,
            "tokens_per_second": self.tokens_per_second,
            "total_time": self.total_time_seconds,
            "draft_time": self.draft_time_seconds,
            "verify_time": self.verify_time_seconds,
            "num_disagreements": self.num_disagreements,
        }


@dataclass
class ManualSpeculativeResult:
    """Result from manual speculative decoding with token-level data."""
    
    # Generated text
    text: str
    
    # All generated token IDs
    tokens: List[int]
    
    # Generation metrics
    metrics: ManualSpeculativeMetrics
    
    # Input prompt
    prompt: str
    
    # Token-level disagreements (the key data for training)
    disagreements: List[TokenLevelDisagreement] = field(default_factory=list)
    
    # Whether this is considered a failure case
    is_failure_case: bool = False
    
    @property
    def has_detailed_data(self) -> bool:
        """Check if we have token-level disagreement data."""
        return len(self.disagreements) > 0


class ManualSpeculativeDecoder:
    """
    Manual speculative decoding implementation for data collection.
    
    This class implements the speculative decoding algorithm manually to
    capture every token-level decision between the draft and target models.
    
    Algorithm overview:
    1. Draft model generates K tokens autoregressively
    2. Target model verifies all K tokens in ONE forward pass
    3. For each position, compare target's choice vs draft's prediction
    4. Record disagreements with full context (tokens, confidences)
    5. Accept tokens until first rejection, resample at rejection point
    6. Rewind caches and continue
    
    This is ~20% slower than MLX-LM's built-in speculative decoding but
    provides much more valuable training signal.
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
        context_window: int = 16,
        system_message: Optional[str] = None,
    ):
        """
        Initialize the manual speculative decoder.
        
        Args:
            target_model: Large target model for verification
            draft_model: Small draft model for speculation
            tokenizer: Tokenizer (shared between models)
            num_draft_tokens: Number of tokens to draft per step (K)
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold
            acceptance_threshold: Below this acceptance rate = failure case
            context_window: Number of context tokens to include in disagreements
            system_message: System message for chat formatting
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.num_draft_tokens = num_draft_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.acceptance_threshold = acceptance_threshold
        self.context_window = context_window
        self.system_message = system_message or "You are a helpful assistant."
        
        # Get EOS token
        self.eos_token_id = tokenizer.eos_token_id
        if self.eos_token_id is None:
            self.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
        
        # Check for chat template support
        self._has_chat_template = (
            hasattr(tokenizer, 'apply_chat_template') and 
            callable(tokenizer.apply_chat_template)
        )
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt using chat template if available."""
        if self._has_chat_template:
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt},
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                logger.warning(f"apply_chat_template failed: {e}")
        return prompt
    
    def _get_softmax_probs(self, logits: mx.array) -> mx.array:
        """Convert logits to softmax probabilities."""
        # Handle different shapes
        if logits.ndim == 3:
            logits = logits[0]  # Remove batch dim
        if logits.ndim == 2:
            # [seq_len, vocab_size] - return as-is
            return mx.softmax(logits, axis=-1)
        # [vocab_size]
        return mx.softmax(logits, axis=-1)
    
    def _sample_from_logits(
        self,
        logits: mx.array,
    ) -> Tuple[int, float]:
        """
        Sample a token from logits and return the token and its probability.
        
        Args:
            logits: Logits for vocabulary [vocab_size] or [1, vocab_size]
            
        Returns:
            Tuple of (sampled_token_id, probability)
        """
        if logits.ndim == 2:
            logits = logits[0]
        
        probs = mx.softmax(logits, axis=-1)
        
        if self.temperature == 0:
            # Greedy sampling
            token = mx.argmax(logits).item()
            prob = probs[token].item()
        else:
            # Temperature sampling
            scaled_logits = logits / self.temperature
            
            # Top-p sampling
            if self.top_p < 1.0:
                sorted_indices = mx.argsort(-scaled_logits)
                sorted_probs = mx.softmax(scaled_logits[sorted_indices], axis=-1)
                cumsum = mx.cumsum(sorted_probs)
                
                # Find cutoff
                mask = cumsum <= self.top_p
                # Always include at least one token
                mask = mx.concatenate([mx.array([True]), mask[:-1]])
                
                # Zero out tokens beyond cutoff
                scaled_logits_masked = mx.where(
                    mx.zeros_like(scaled_logits).at[sorted_indices].add(
                        mask.astype(scaled_logits.dtype)
                    ) > 0,
                    scaled_logits,
                    mx.full_like(scaled_logits, float('-inf'))
                )
                scaled_logits = scaled_logits_masked
            
            # Sample
            token = mx.random.categorical(scaled_logits).item()
            prob = probs[token].item()
        
        return token, prob
    
    def _accept_token(
        self,
        draft_token: int,
        draft_prob: float,
        target_logits: mx.array,
    ) -> Tuple[bool, int, float]:
        """
        Decide whether to accept a draft token using speculative sampling.
        
        For greedy (temp=0): Accept if target argmax matches draft
        For sampling (temp>0): Accept with probability min(1, p_target/p_draft)
        
        Args:
            draft_token: Token proposed by draft model
            draft_prob: Probability draft assigned to this token
            target_logits: Target model's logits for this position
            
        Returns:
            Tuple of (accepted, final_token, target_prob)
            - accepted: Whether draft token was accepted
            - final_token: Token to use (draft if accepted, else resampled)
            - target_prob: Target's probability for the final token
        """
        if target_logits.ndim == 2:
            target_logits = target_logits[0]
        
        target_probs = mx.softmax(target_logits, axis=-1)
        target_prob_draft = target_probs[draft_token].item()
        
        if self.temperature == 0:
            # Greedy: accept if target agrees
            target_token = mx.argmax(target_logits).item()
            accepted = (target_token == draft_token)
            if accepted:
                return True, draft_token, target_prob_draft
            else:
                return False, target_token, target_probs[target_token].item()
        else:
            # Probabilistic acceptance: r < p_target / p_draft
            acceptance_prob = min(1.0, target_prob_draft / max(draft_prob, 1e-10))
            r = mx.random.uniform().item()
            
            if r < acceptance_prob:
                return True, draft_token, target_prob_draft
            else:
                # Rejection: sample from adjusted distribution
                # p_adjusted = max(0, p_target - p_draft) normalized
                adjusted_probs = mx.maximum(
                    target_probs - draft_prob * mx.ones_like(target_probs),
                    mx.zeros_like(target_probs)
                )
                adjusted_probs = adjusted_probs / (mx.sum(adjusted_probs) + 1e-10)
                
                # Sample from adjusted distribution
                final_token = mx.random.categorical(mx.log(adjusted_probs + 1e-10)).item()
                final_prob = target_probs[final_token].item()
                
                return False, final_token, final_prob
    
    def generate_with_data_collection(
        self,
        prompt: str,
        max_tokens: int = 256,
        K: Optional[int] = None,
    ) -> ManualSpeculativeResult:
        """
        Generate text using manual speculative decoding with full data collection.
        
        This is the main entry point for data collection. It implements speculative
        decoding from scratch to capture every token-level decision.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            K: Number of draft tokens per step (defaults to self.num_draft_tokens)
            
        Returns:
            ManualSpeculativeResult with generated text and detailed disagreement data
        """
        K = K or self.num_draft_tokens
        
        start_time = time.time()
        metrics = ManualSpeculativeMetrics()
        disagreements: List[TokenLevelDisagreement] = []
        
        # Format and encode prompt
        formatted_prompt = self._format_prompt(prompt)
        prompt_tokens = self.tokenizer.encode(formatted_prompt)
        prompt_tensor = mx.array(prompt_tokens)[None, :]  # [1, seq_len]
        
        # Initialize: run prompt through both models to populate caches
        draft_cache = create_kv_cache(self.draft_model)
        target_cache = create_kv_cache(self.target_model)
        
        # Process prompt with both models
        draft_logits, draft_cache = get_logits_with_cache(
            self.draft_model, prompt_tensor, draft_cache
        )
        target_logits, target_cache = get_logits_with_cache(
            self.target_model, prompt_tensor, target_cache
        )
        mx.eval(draft_logits, target_logits)
        
        # Track generated tokens
        generated_tokens: List[int] = []
        current_position = len(prompt_tokens)
        
        # Main generation loop
        while len(generated_tokens) < max_tokens:
            draft_start = time.time()
            
            # ============================================================
            # DRAFT PHASE: Generate K tokens from draft model
            # ============================================================
            draft_tokens: List[int] = []
            draft_probs: List[float] = []
            
            for _ in range(K):
                # Get next token prediction from draft
                if len(draft_tokens) == 0:
                    # First draft token uses last position's logits
                    next_logits = draft_logits[:, -1, :]
                else:
                    # Subsequent tokens: run draft model on previous draft token
                    prev_token = mx.array([[draft_tokens[-1]]])
                    next_logits, draft_cache = get_logits_with_cache(
                        self.draft_model, prev_token, draft_cache
                    )
                    next_logits = next_logits[:, -1, :]
                
                mx.eval(next_logits)
                
                # Sample token
                token, prob = self._sample_from_logits(next_logits)
                draft_tokens.append(token)
                draft_probs.append(prob)
                
                # Stop if EOS
                if token == self.eos_token_id:
                    break
            
            metrics.draft_time_seconds += time.time() - draft_start
            metrics.draft_tokens_proposed += len(draft_tokens)
            
            if not draft_tokens:
                break
            
            # ============================================================
            # VERIFY PHASE: Run target model on all draft tokens at once
            # ============================================================
            verify_start = time.time()
            
            # Create tensor of draft tokens for parallel verification
            draft_tensor = mx.array([draft_tokens])  # [1, K]
            
            # Get target logits for all draft positions in one forward pass
            verify_logits, target_cache = get_logits_with_cache(
                self.target_model, draft_tensor, target_cache
            )
            mx.eval(verify_logits)
            
            metrics.verify_time_seconds += time.time() - verify_start
            
            # ============================================================
            # COMPARE & RECORD: Check each position and record disagreements
            # ============================================================
            accepted_count = 0
            final_tokens: List[int] = []
            
            for i, (draft_token, draft_prob) in enumerate(zip(draft_tokens, draft_probs)):
                # Get target logits for this position
                target_pos_logits = verify_logits[0, i, :]
                
                # Decide acceptance
                accepted, final_token, target_prob = self._accept_token(
                    draft_token, draft_prob, target_pos_logits
                )
                
                if accepted:
                    accepted_count += 1
                    final_tokens.append(final_token)
                else:
                    # Record disagreement with context
                    context_start = max(0, len(prompt_tokens) + len(generated_tokens) - self.context_window)
                    context_end = len(prompt_tokens) + len(generated_tokens)
                    
                    all_tokens = prompt_tokens + generated_tokens
                    context = all_tokens[context_start:context_end]
                    
                    disagreement = TokenLevelDisagreement(
                        position=current_position + i,
                        draft_token=draft_token,
                        target_token=final_token,
                        draft_confidence=draft_prob,
                        target_confidence=target_prob,
                        context_tokens=context,
                    )
                    disagreements.append(disagreement)
                    metrics.num_disagreements += 1
                    
                    # Add the corrected token and stop
                    final_tokens.append(final_token)
                    break
                
                # Stop at EOS
                if final_token == self.eos_token_id:
                    break
            
            metrics.draft_tokens_accepted += accepted_count
            
            # ============================================================
            # UPDATE: Add accepted tokens and rewind caches if needed
            # ============================================================
            if final_tokens:
                generated_tokens.extend(final_tokens)
                current_position += len(final_tokens)
                
                # If we rejected some tokens, we need to rewind both caches
                if len(final_tokens) < len(draft_tokens):
                    # Rewind draft cache to rejection point
                    # The draft cache advanced by K tokens, but we only kept len(final_tokens)
                    num_to_rewind = len(draft_tokens) - len(final_tokens)
                    rewind_to = get_cache_length(draft_cache) - num_to_rewind
                    draft_cache = rewind_cache(draft_cache, rewind_to)
                    
                    # Target cache also needs adjustment
                    # It verified K tokens but we only kept len(final_tokens)
                    rewind_to = get_cache_length(target_cache) - num_to_rewind
                    target_cache = rewind_cache(target_cache, rewind_to)
                    
                    # Re-process the last accepted token to set up for next iteration
                    if final_tokens:
                        last_token = mx.array([[final_tokens[-1]]])
                        draft_logits, draft_cache = get_logits_with_cache(
                            self.draft_model, last_token, draft_cache
                        )
                        mx.eval(draft_logits)
            
            # Check for EOS
            if generated_tokens and generated_tokens[-1] == self.eos_token_id:
                break
        
        # ============================================================
        # FINALIZE: Prepare result
        # ============================================================
        metrics.total_time_seconds = time.time() - start_time
        metrics.total_tokens_generated = len(generated_tokens)
        
        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_tokens)
        
        # Determine if this is a failure case
        is_failure = metrics.acceptance_rate < self.acceptance_threshold
        
        # Clear memory
        mx.clear_cache()
        
        logger.debug(
            f"Manual speculative decoding: {metrics.total_tokens_generated} tokens, "
            f"{metrics.acceptance_rate:.1%} acceptance, "
            f"{len(disagreements)} disagreements"
        )
        
        return ManualSpeculativeResult(
            text=generated_text,
            tokens=generated_tokens,
            metrics=metrics,
            prompt=prompt,
            disagreements=disagreements,
            is_failure_case=is_failure,
        )
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
    ) -> ManualSpeculativeResult:
        """
        Convenience method that calls generate_with_data_collection.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            
        Returns:
            ManualSpeculativeResult with generated text and disagreement data
        """
        return self.generate_with_data_collection(prompt, max_tokens)


def run_data_collection_batch(
    decoder: ManualSpeculativeDecoder,
    prompts: List[str],
    max_tokens: int = 256,
    verbose: bool = True,
) -> Tuple[List[ManualSpeculativeResult], Dict[str, Any]]:
    """
    Run manual speculative decoding on a batch of prompts for data collection.
    
    This function processes multiple prompts and aggregates the results,
    providing detailed statistics about the collected data.
    
    Args:
        decoder: Configured ManualSpeculativeDecoder
        prompts: List of prompts to process
        max_tokens: Maximum tokens per generation
        verbose: Whether to log progress
        
    Returns:
        Tuple of (results, aggregate_stats)
    """
    results: List[ManualSpeculativeResult] = []
    total_disagreements = 0
    total_tokens = 0
    total_accepted = 0
    total_proposed = 0
    total_time = 0.0
    
    for i, prompt in enumerate(prompts):
        if verbose:
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
        
        result = decoder.generate_with_data_collection(prompt, max_tokens)
        results.append(result)
        
        total_disagreements += len(result.disagreements)
        total_tokens += result.metrics.total_tokens_generated
        total_accepted += result.metrics.draft_tokens_accepted
        total_proposed += result.metrics.draft_tokens_proposed
        total_time += result.metrics.total_time_seconds
    
    # Compute aggregate statistics
    stats = {
        "num_prompts": len(prompts),
        "total_tokens_generated": total_tokens,
        "total_disagreements": total_disagreements,
        "avg_disagreements_per_prompt": total_disagreements / max(len(prompts), 1),
        "overall_acceptance_rate": total_accepted / max(total_proposed, 1),
        "total_time_seconds": total_time,
        "avg_tokens_per_second": total_tokens / max(total_time, 0.001),
        "failure_cases": sum(1 for r in results if r.is_failure_case),
    }
    
    # Count high-confidence failures
    high_conf_failures = sum(
        1 for r in results
        for d in r.disagreements
        if d.is_high_confidence_failure
    )
    stats["high_confidence_failures"] = high_conf_failures
    
    if verbose:
        logger.info(
            f"Data collection complete: {total_tokens} tokens, "
            f"{total_disagreements} disagreements, "
            f"{stats['overall_acceptance_rate']:.1%} acceptance rate"
        )
    
    return results, stats
