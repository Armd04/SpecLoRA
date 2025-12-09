"""
Manual Speculative Decoding for Token-Level Data Collection

This module implements speculative decoding manually (rather than using MLX-LM's
built-in version) to capture detailed token-level disagreements between the
draft and target models.

The key difference from the built-in version:
- Built-in: Fast, optimized, but only exposes overall acceptance rates
- Manual: Comparable speed (with KV caching), captures every token decision for training

This enables more effective LoRA training by focusing gradients on specific
positions where the draft model fails, rather than uniformly across sequences.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import mlx.core as mx
import mlx.nn as nn

from .data_collector import TokenLevelDisagreement
from .models import create_kv_cache, get_logits_with_cache, rewind_cache, get_cache_length

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
    1. Draft model generates K tokens autoregressively (using KV cache)
    2. Target model verifies all K tokens in ONE forward pass
    3. For each position, compare target's choice vs draft's prediction
    4. Record disagreements with full context (tokens, confidences)
    5. Accept tokens until first rejection, resample at rejection point
    6. Update caches with accepted tokens and continue

    With KV caching, performance should be comparable to MLX-LM's built-in
    speculative decoding while providing much more valuable training signal.
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        tokenizer: Any,
        num_draft_tokens: int = 4,
        temperature: float = 0.0,  # Use greedy by default for consistency
        top_p: float = 1.0,
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
            # convert_tokens_to_ids may return a list [id] instead of int
            # Ensure we always have an int for comparisons
            eos_result = tokenizer.convert_tokens_to_ids("</s>")
            if isinstance(eos_result, list):
                self.eos_token_id = eos_result[0] if eos_result else None
            else:
                self.eos_token_id = eos_result
        
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
    
    def _sample_token(self, logits: mx.array) -> Tuple[int, float]:
        """
        Sample a token from logits using temperature sampling.
        
        Args:
            logits: Logits of shape [vocab_size]
            
        Returns:
            Tuple of (token_id, probability)
        """
        # Ensure 1D
        if logits.ndim > 1:
            logits = logits.reshape(-1)
        
        probs = mx.softmax(logits, axis=-1)
        
        if self.temperature == 0:
            # Greedy
            token = mx.argmax(logits).item()
        else:
            # Temperature sampling
            scaled_logits = logits / self.temperature
            token = mx.random.categorical(scaled_logits).item()
        
        prob = probs[token].item()
        return token, prob
    
    def _verify_and_accept(
        self,
        draft_token: int,
        draft_prob: float,
        target_logits: mx.array,
    ) -> Tuple[bool, int, float]:
        """
        Verify a draft token against target model's prediction.
        
        Uses greedy comparison: accept if target's argmax matches draft.
        For temperature > 0, uses probabilistic acceptance.
        
        Args:
            draft_token: Token proposed by draft model
            draft_prob: Probability draft assigned to this token
            target_logits: Target model's logits for this position
            
        Returns:
            Tuple of (accepted, final_token, target_prob)
        """
        if target_logits.ndim > 1:
            target_logits = target_logits.reshape(-1)
        
        target_probs = mx.softmax(target_logits, axis=-1)
        
        if self.temperature == 0:
            # Greedy verification
            target_choice = mx.argmax(target_logits).item()
            accepted = (target_choice == draft_token)
            
            if accepted:
                return True, draft_token, target_probs[draft_token].item()
            else:
                return False, target_choice, target_probs[target_choice].item()
        else:
            # Probabilistic acceptance: accept with prob min(1, p_target/p_draft)
            target_prob_for_draft = target_probs[draft_token].item()
            acceptance_ratio = min(1.0, target_prob_for_draft / max(draft_prob, 1e-10))
            
            r = mx.random.uniform().item()
            if r < acceptance_ratio:
                return True, draft_token, target_prob_for_draft
            else:
                # Sample from target distribution
                scaled_logits = target_logits / self.temperature
                target_choice = mx.random.categorical(scaled_logits).item()
                return False, target_choice, target_probs[target_choice].item()
    
    def generate_with_data_collection(
        self,
        prompt: str,
        max_tokens: int = 256,
        K: Optional[int] = None,
    ) -> ManualSpeculativeResult:
        """
        Generate text using manual speculative decoding with full data collection.
        
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

        # Build the full sequence: start with prompt tokens
        all_tokens = list(prompt_tokens)
        generated_tokens: List[int] = []

        # ============================================================
        # Initialize KV cache for draft model
        # ============================================================
        draft_cache = create_kv_cache(self.draft_model)

        # Populate cache with the prompt and get initial logits
        prompt_input = mx.array(all_tokens)[None, :]
        draft_logits, draft_cache = get_logits_with_cache(self.draft_model, prompt_input, draft_cache)
        mx.eval(draft_cache)
        mx.eval(draft_logits)

        # Save logits for next position (will be used for first draft token)
        next_position_logits = draft_logits[0, -1, :]
        mx.eval(next_position_logits)  # Evaluate to avoid lazy recomputation

        # Main generation loop
        while len(generated_tokens) < max_tokens:
            draft_start = time.time()
            
            # ============================================================
            # DRAFT PHASE: Generate K tokens from draft model (WITH KV CACHE)
            # ============================================================
            draft_tokens: List[int] = []
            draft_probs: List[float] = []

            # Save current cache position - we'll rewind after draft generation
            # since draft tokens are speculative
            cache_position_before_draft = get_cache_length(draft_cache)

            # Use the saved logits for first draft token
            current_logits = next_position_logits

            # Generate K draft tokens autoregressively using cache
            for k in range(K):
                # Sample from draft model's prediction
                token, prob = self._sample_token(current_logits)
                draft_tokens.append(token)
                draft_probs.append(prob)

                if token == self.eos_token_id:
                    break

                # Get logits for next position by passing this token with cache
                # NOTE: Only for k < K-1 (not the last token). The last draft token
                # doesn't need to be added to cache since we won't need logits after it,
                # and we're about to rewind the cache anyway. This is an intentional
                # optimization to avoid unnecessary computation.
                if k < K - 1:
                    next_input = mx.array([[token]])
                    draft_output, draft_cache = get_logits_with_cache(
                        self.draft_model, next_input, draft_cache
                    )
                    mx.eval(draft_output)
                    mx.eval(draft_cache)
                    current_logits = draft_output[0, -1, :]

            # Rewind draft cache to remove speculative draft tokens
            # Cache should have grown by len(draft_tokens)-1 tokens (last token not added)
            # After rewind, cache returns to state before draft generation
            if logger.isEnabledFor(logging.DEBUG):
                cache_len = get_cache_length(draft_cache)
                expected_len = cache_position_before_draft + len(draft_tokens) - 1
                if cache_len != expected_len:
                    logger.debug(
                        f"Cache length mismatch before rewind: {cache_len} != {expected_len}. "
                        f"Draft tokens: {len(draft_tokens)}, base position: {cache_position_before_draft}"
                    )

            draft_cache = rewind_cache(draft_cache, cache_position_before_draft)
            mx.eval(draft_cache)
            
            metrics.draft_time_seconds += time.time() - draft_start
            metrics.draft_tokens_proposed += len(draft_tokens)
            
            if not draft_tokens:
                break
            
            # ============================================================
            # VERIFY PHASE: Run target model to verify draft tokens
            # ============================================================
            verify_start = time.time()
            
            # Run target on sequence + all draft tokens
            verify_sequence = all_tokens + draft_tokens
            verify_input = mx.array(verify_sequence)[None, :]
            target_output = self.target_model(verify_input)
            mx.eval(target_output)
            
            metrics.verify_time_seconds += time.time() - verify_start
            
            # ============================================================
            # COMPARE: Check each draft token against target's prediction
            # ============================================================
            # 
            # Key insight: target_output[0, i, :] contains logits for predicting
            # position i+1. So to verify draft_tokens[k], we look at
            # target_output[0, len(all_tokens) - 1 + k, :]
            #
            # Position len(all_tokens) - 1 is the last prompt token,
            # and its output predicts what should come at len(all_tokens),
            # which is where draft_tokens[0] is.
            
            accepted_count = 0
            final_tokens: List[int] = []
            
            for k, (draft_token, draft_prob) in enumerate(zip(draft_tokens, draft_probs)):
                # Target logits that predict position (len(all_tokens) + k)
                # These are at output position (len(all_tokens) - 1 + k)
                target_pos = len(all_tokens) - 1 + k
                target_logits = target_output[0, target_pos, :]
                
                # Verify
                accepted, final_token, target_prob = self._verify_and_accept(
                    draft_token, draft_prob, target_logits
                )
                
                if accepted:
                    accepted_count += 1
                    final_tokens.append(final_token)
                else:
                    # Record disagreement
                    # Include both all_tokens AND the k tokens already accepted in this iteration
                    full_sequence_before_disagreement = all_tokens + final_tokens
                    context_start = max(0, len(full_sequence_before_disagreement) - self.context_window)
                    context = full_sequence_before_disagreement[context_start:]

                    disagreement = TokenLevelDisagreement(
                        position=len(all_tokens) + k,
                        draft_token=draft_token,
                        target_token=final_token,
                        draft_confidence=draft_prob,
                        target_confidence=target_prob,
                        context_tokens=context,
                    )
                    disagreements.append(disagreement)
                    metrics.num_disagreements += 1
                    
                    # Add the corrected token and stop accepting more
                    final_tokens.append(final_token)
                    break
                
                if final_token == self.eos_token_id:
                    break
            
            metrics.draft_tokens_accepted += accepted_count
            
            # ============================================================
            # UPDATE: Add accepted/corrected tokens to sequence and update cache
            # ============================================================
            if final_tokens:
                all_tokens.extend(final_tokens)
                generated_tokens.extend(final_tokens)

                # Update draft cache with accepted tokens and get logits for next iteration
                accepted_input = mx.array(final_tokens)[None, :]
                draft_output, draft_cache = get_logits_with_cache(
                    self.draft_model, accepted_input, draft_cache
                )
                mx.eval(draft_cache)
                mx.eval(draft_output)

                # Save logits for next draft token
                next_position_logits = draft_output[0, -1, :]
                mx.eval(next_position_logits)  # Evaluate to avoid lazy recomputation

            # Check for EOS
            if generated_tokens and generated_tokens[-1] == self.eos_token_id:
                break
        
        # ============================================================
        # FINALIZE: Prepare result
        # ============================================================
        metrics.total_time_seconds = time.time() - start_time
        metrics.total_tokens_generated = len(generated_tokens)
        
        # Decode generated tokens (strip EOS token from display)
        tokens_to_decode = generated_tokens.copy()
        if tokens_to_decode and tokens_to_decode[-1] == self.eos_token_id:
            tokens_to_decode = tokens_to_decode[:-1]
        generated_text = self.tokenizer.decode(tokens_to_decode)
        
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
