"""
Manual Speculative Decoding for Token-Level Data Collection

This module implements speculative decoding manually (rather than using MLX-LM's
built-in version) to capture detailed token-level disagreements between the
draft and target models.

The key difference from the built-in version:
- Built-in: Fast, optimized, but only exposes overall acceptance rates
- Manual: Comparable speed (with KV caching), captures every token decision for training

Performance optimizations:
- Both draft AND target models use KV caching for efficient generation
- Draft model: cache is speculatively extended during drafting, then rewound
- Target model: cache is extended during verification, rewound, then updated with
  only accepted tokens. This avoids recomputing attention for the entire prefix
  on every verification step (O(K) per step instead of O(N+K)).

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
from .models import (
    create_kv_cache,
    get_logits_with_cache,
    rewind_cache,
    get_cache_length,
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
    1. Initialize KV caches for BOTH draft and target models
    2. Populate both caches with prompt tokens
    3. Draft model generates K tokens autoregressively (using KV cache)
    4. Target model verifies all K tokens in ONE forward pass (using KV cache)
       - Only processes K draft tokens, not the full sequence (O(K) vs O(N+K))
    5. For each position, compare target's choice vs draft's prediction
    6. Record disagreements with full context (tokens, confidences)
    7. Accept tokens until first rejection, resample at rejection point
    8. Rewind both caches, then update with only accepted/corrected tokens
    9. Continue until max_tokens or EOS

    With KV caching on both models, performance is significantly better than
    naive full-sequence verification while providing detailed training signal.
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
        top_k_logits: int = 10,
        distillation_temperature: float = 2.0,
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
            top_k_logits: Number of top logits to store per disagreement for KL distillation
            distillation_temperature: Temperature for extracting target logits (should match
                training.loss.temperature in config for consistent KL divergence)
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
        self.top_k_logits = top_k_logits
        self.distillation_temperature = distillation_temperature

        if not hasattr(self.tokenizer, "apply_chat_template") or not callable(
            self.tokenizer.apply_chat_template
        ):
            raise ValueError(
                "ManualSpeculativeDecoder requires a tokenizer with apply_chat_template(). "
                "Please ensure your tokenizer supports chat templates."
            )

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

    def format_prompt(self, prompt: str) -> str:
        """Public helper for generating formatted prompts."""
        return self._format_prompt(prompt)

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt using tokenizer chat template."""
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
        except Exception as exc:
            raise RuntimeError(
                "tokenizer.apply_chat_template() failed during manual decoding."
            ) from exc

    def _sample_token(self, logits: mx.array) -> Tuple[int, float, Optional[mx.array]]:
        """
        Sample a token from logits using temperature sampling.

        Args:
            logits: Logits of shape [vocab_size]

        Returns:
            Tuple of (token_id, probability, full_probs or None)
        """
        # Ensure 1D
        if logits.ndim > 1:
            logits = logits.reshape(-1)

        if self.temperature == 0:
            # Greedy: probability is 1 for the chosen token
            token = mx.argmax(logits).item()
            return token, 1.0, None

        # Temperature-scaled sampling and probability from the same distribution
        scaled_logits = logits / self.temperature
        scaled_probs = mx.softmax(scaled_logits, axis=-1)
        token = mx.random.categorical(scaled_logits).item()
        prob = scaled_probs[token].item()
        return token, prob, scaled_probs

    def _extract_top_k_logits(
        self, logits: mx.array, k: int, temperature: float = 1.0
    ) -> List[Tuple[int, float]]:
        """
        Extract top-k logits as (token_id, probability) tuples.

        Args:
            logits: Logits array of shape [vocab_size]
            k: Number of top logits to extract
            temperature: Temperature for scaling logits before softmax (default 1.0)

        Returns:
            List of (token_id, probability) tuples, sorted by probability descending
        """
        # Ensure 1D
        if logits.ndim > 1:
            logits = logits.reshape(-1)

        # Apply temperature BEFORE softmax (consistent with KL divergence computation)
        scaled_logits = logits / temperature
        probs = mx.softmax(scaled_logits, axis=-1)
        mx.eval(probs)

        # Get top-k indices and values
        k = min(k, probs.shape[-1])
        top_k_indices = mx.argsort(probs, axis=-1)[-k:][::-1]  # Descending order
        top_k_values = mx.take(probs, top_k_indices, axis=-1)

        mx.eval(top_k_indices, top_k_values)

        # Convert to list of tuples
        result = [
            (int(top_k_indices[i].item()), float(top_k_values[i].item()))
            for i in range(k)
        ]
        return result

    def _verify_and_accept(
        self,
        draft_token: int,
        draft_prob: float,
        target_logits: mx.array,
        draft_probs_full: Optional[mx.array],
        debug: bool = False,
    ) -> Tuple[bool, int, float]:
        """
        Verify a draft token against target model's prediction.

        Uses greedy comparison: accept if target's argmax matches draft.
        For temperature > 0, uses probabilistic acceptance.

        Args:
            draft_token: Token proposed by draft model
            draft_prob: Probability draft assigned to this token
            target_logits: Target model's logits for this position
            draft_probs_full: Draft probability distribution (temp-scaled) for this position
            debug: If True, print debug information

        Returns:
            Tuple of (accepted, final_token, target_prob)
        """
        # Ensure we have evaluated the logits
        mx.eval(target_logits)

        if target_logits.ndim > 1:
            target_logits = target_logits.reshape(-1)

        if self.temperature == 0:
            target_probs = mx.softmax(target_logits, axis=-1)
        else:
            scaled_target_logits = target_logits / self.temperature
            target_probs = mx.softmax(scaled_target_logits, axis=-1)
        mx.eval(target_probs)

        if self.temperature == 0:
            # Greedy verification
            target_choice = mx.argmax(target_logits).item()
            accepted = target_choice == draft_token

            if debug:
                logger.info(
                    f"  Greedy verify: draft={draft_token}, target_argmax={target_choice}, accepted={accepted}"
                )

            if accepted:
                return True, draft_token, target_probs[draft_token].item()
            else:
                return False, target_choice, target_probs[target_choice].item()
        else:
            # Probabilistic acceptance: accept with prob min(1, p_target/p_draft)
            if draft_probs_full is None:
                raise RuntimeError(
                    "Draft probabilities missing during probabilistic verification. "
                    "Ensure _sample_token returns the full distribution when temperature > 0."
                )
            mx.eval(draft_probs_full)

            target_vocab = target_probs.shape[-1]
            draft_vocab = draft_probs_full.shape[-1]

            if draft_token >= target_vocab:
                # Draft proposed a token outside target vocab; force reject
                target_prob_for_draft = 0.0
                acceptance_ratio = 0.0
            else:
                target_prob_for_draft = target_probs[draft_token].item()
                acceptance_ratio = min(
                    1.0, target_prob_for_draft / max(draft_prob, 1e-10)
                )

            r = mx.random.uniform().item()
            accepted = r < acceptance_ratio

            if debug:
                target_argmax = mx.argmax(target_logits).item()
                logger.info(
                    f"  Prob verify: draft={draft_token} (p={draft_prob:.4f}), "
                    f"target_argmax={target_argmax}, target_p_for_draft={target_prob_for_draft:.4f}, "
                    f"ratio={acceptance_ratio:.4f}, r={r:.4f}, accepted={accepted}"
                )

            if accepted:
                return True, draft_token, target_prob_for_draft
            else:
                # Sample from residual distribution to preserve target law
                if draft_vocab < target_vocab:
                    pad = mx.zeros(
                        (target_vocab - draft_vocab,), dtype=target_probs.dtype
                    )
                    draft_probs_aligned = mx.concatenate(
                        [draft_probs_full, pad], axis=0
                    )
                elif draft_vocab > target_vocab:
                    draft_probs_aligned = draft_probs_full[:target_vocab]
                    # Renormalize after truncation so distribution sums to 1
                    draft_mass = mx.sum(draft_probs_aligned)
                    mx.eval(draft_mass)
                    if draft_mass.item() > 0:
                        draft_probs_aligned = draft_probs_aligned / draft_mass
                    else:
                        draft_probs_aligned = mx.zeros_like(target_probs)
                else:
                    draft_probs_aligned = draft_probs_full

                residual = mx.maximum(target_probs - draft_probs_aligned, 0.0)

                mx.eval(residual)
                residual_mass = mx.sum(residual).item()

                if residual_mass <= 0:
                    # Fallback: sample from target distribution
                    scaled_target_logits = target_logits / self.temperature
                    target_choice = mx.random.categorical(scaled_target_logits).item()
                    return False, target_choice, target_probs[target_choice].item()

                residual_probs = residual / residual_mass
                residual_logits = mx.log(residual_probs + 1e-12)
                target_choice = mx.random.categorical(residual_logits).item()
                return False, target_choice, target_probs[target_choice].item()

    def generate_with_data_collection(
        self,
        prompt: str,
        max_tokens: int = 256,
        K: Optional[int] = None,
        debug: bool = False,
    ) -> ManualSpeculativeResult:
        """
        Generate text using manual speculative decoding with full data collection.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            K: Number of draft tokens per step (defaults to self.num_draft_tokens)
            debug: If True, print detailed debug information for first few iterations

        Returns:
            ManualSpeculativeResult with generated text and detailed disagreement data
        """
        K = K or self.num_draft_tokens
        debug_iterations = 3 if debug else 0  # Debug first 3 iterations

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
        # Initialize KV caches for both draft and target models
        # ============================================================
        draft_cache = create_kv_cache(self.draft_model)
        target_cache = create_kv_cache(self.target_model)

        # Populate both caches with the prompt
        prompt_input = mx.array(all_tokens)[None, :]

        # Draft cache: get initial logits for first draft token
        draft_logits, draft_cache = get_logits_with_cache(
            self.draft_model, prompt_input, draft_cache
        )
        mx.eval(draft_cache)
        mx.eval(draft_logits)

        # Target cache: populate and save logits for verifying first draft token
        target_logits, target_cache = get_logits_with_cache(
            self.target_model, prompt_input, target_cache
        )
        mx.eval(target_cache)
        mx.eval(target_logits)

        # Save logits for next position (will be used for first draft token)
        next_position_logits = draft_logits[0, -1, :]
        mx.eval(next_position_logits)  # Evaluate to avoid lazy recomputation

        # Save target logits for verifying first draft token of first iteration
        # target_logits[0, -1, :] predicts position len(all_tokens), i.e., where draft_tokens[0] goes
        target_next_position_logits = target_logits[0, -1, :]
        mx.eval(target_next_position_logits)

        # Main generation loop
        iteration_count = 0
        while len(generated_tokens) < max_tokens:
            iteration_count += 1
            should_debug = debug and (iteration_count <= debug_iterations)

            if should_debug:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"ITERATION {iteration_count}")
                logger.info(f"  generated_tokens so far: {len(generated_tokens)}")
                logger.info(f"  all_tokens length: {len(all_tokens)}")
                logger.info(f"{'=' * 60}")

            draft_start = time.time()

            # ============================================================
            # DRAFT PHASE: Generate K tokens from draft model (WITH KV CACHE)
            # ============================================================
            draft_tokens: List[int] = []
            draft_probs: List[float] = []
            draft_prob_dists: List[Optional[mx.array]] = []

            # Save current cache position - we'll rewind after draft generation
            # since draft tokens are speculative
            cache_position_before_draft = get_cache_length(draft_cache)

            # Use the saved logits for first draft token
            current_logits = next_position_logits

            # Generate K draft tokens autoregressively using cache
            for k in range(K):
                if should_debug:
                    mx.eval(current_logits)
                    draft_argmax = mx.argmax(current_logits).item()
                    decoded_argmax = self.tokenizer.decode([draft_argmax])
                    cache_len = get_cache_length(draft_cache)
                    logger.info(
                        f"  DRAFT k={k}: cache_len={cache_len}, logits argmax={draft_argmax} ('{decoded_argmax}')"
                    )

                # Sample from draft model's prediction
                token, prob, prob_dist = self._sample_token(current_logits)
                draft_tokens.append(token)
                draft_probs.append(prob)
                draft_prob_dists.append(prob_dist)

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
                    mx.eval(current_logits)

            if should_debug:
                logger.info(f"DRAFT PHASE: Generated {len(draft_tokens)} draft tokens")
                for idx, (dt, dp) in enumerate(zip(draft_tokens, draft_probs)):
                    decoded = self.tokenizer.decode([dt])
                    logger.info(
                        f"  draft[{idx}]: token={dt} ('{decoded}'), prob={dp:.4f}"
                    )

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
            # VERIFY PHASE: Run target model to verify draft tokens (WITH KV CACHE)
            # ============================================================
            verify_start = time.time()

            # Save target cache position - we'll rewind to keep only accepted tokens
            target_cache_position_before_verify = get_cache_length(target_cache)

            # Run target ONLY on draft tokens (cache already has all_tokens)
            # This is O(K) instead of O(N+K) where N = len(all_tokens)
            verify_input = mx.array(draft_tokens)[None, :]
            target_output, target_cache = get_logits_with_cache(
                self.target_model, verify_input, target_cache
            )
            mx.eval(target_output)
            mx.eval(target_cache)

            metrics.verify_time_seconds += time.time() - verify_start

            if should_debug:
                logger.info(
                    f"VERIFY PHASE: target_output shape = {target_output.shape}"
                )
                logger.info(
                    f"  target_next_position_logits shape = {target_next_position_logits.shape}"
                )
                # Check what target would sample vs what draft proposed
                target_argmax_0 = mx.argmax(target_next_position_logits).item()
                decoded_target = self.tokenizer.decode([target_argmax_0])
                decoded_draft = self.tokenizer.decode([draft_tokens[0]])
                logger.info(
                    f"  For k=0: target_argmax={target_argmax_0} ('{decoded_target}'), draft={draft_tokens[0]} ('{decoded_draft}')"
                )

            # ============================================================
            # COMPARE: Check each draft token against target's prediction
            # ============================================================
            #
            # With KV caching, the logit indexing is off-by-one from the full forward pass:
            # - target_next_position_logits (saved from previous iteration) predicts position N,
            #   which verifies draft_tokens[0]
            # - target_output[0, k-1, :] predicts position N+k, which verifies draft_tokens[k]
            #   for k >= 1
            #
            # This is because processing draft_tokens[j] outputs logits for position N+j+1,
            # not N+j.

            accepted_count = 0
            final_tokens: List[int] = []

            for k, (draft_token, draft_prob) in enumerate(
                zip(draft_tokens, draft_probs)
            ):
                # Get the correct logits for verifying draft_tokens[k]
                if k == 0:
                    # First draft token: use saved logits from previous iteration
                    target_logits_for_verify = target_next_position_logits
                else:
                    # Subsequent tokens: target_output[0, k-1, :] predicts position N+k
                    target_logits_for_verify = target_output[0, k - 1, :]

                if should_debug:
                    mx.eval(target_logits_for_verify)
                    verify_argmax = mx.argmax(target_logits_for_verify).item()
                    decoded_verify = self.tokenizer.decode([verify_argmax])
                    decoded_draft = self.tokenizer.decode([draft_token])
                    logger.info(
                        f"  k={k}: verify using {'saved' if k == 0 else f'target_output[0,{k - 1},:]'}, "
                        f"argmax={verify_argmax} ('{decoded_verify}'), draft={draft_token} ('{decoded_draft}')"
                    )

                # Verify
                accepted, final_token, target_prob = self._verify_and_accept(
                    draft_token,
                    draft_prob,
                    target_logits_for_verify,
                    draft_prob_dists[k],
                    debug=should_debug,
                )

                if accepted:
                    accepted_count += 1
                    final_tokens.append(final_token)
                else:
                    # Record disagreement
                    # Include both all_tokens AND the k tokens already accepted in this iteration
                    full_sequence_before_disagreement = all_tokens + final_tokens
                    context_start = max(
                        0, len(full_sequence_before_disagreement) - self.context_window
                    )
                    context = full_sequence_before_disagreement[context_start:]

                    # Extract top-k target logits for KL distillation
                    # Use distillation_temperature (default 2.0) to match training-time temperature scaling
                    # This ensures consistent probability distributions for KL divergence computation
                    target_logits_topk = self._extract_top_k_logits(
                        target_logits_for_verify,
                        self.top_k_logits,
                        temperature=self.distillation_temperature,
                    )

                    disagreement = TokenLevelDisagreement(
                        position=len(all_tokens) + k,
                        draft_token=draft_token,
                        target_token=final_token,
                        draft_confidence=draft_prob,
                        target_confidence=target_prob,
                        context_tokens=context,
                        target_logits=target_logits_topk,
                    )
                    disagreements.append(disagreement)
                    metrics.num_disagreements += 1

                    # Add the corrected token and stop accepting more
                    final_tokens.append(final_token)
                    break

                if final_token == self.eos_token_id:
                    break

            metrics.draft_tokens_accepted += accepted_count

            if should_debug:
                logger.info(
                    f"COMPARE RESULT: accepted={accepted_count}/{len(draft_tokens)}"
                )
                logger.info(f"  final_tokens: {final_tokens}")
                for idx, ft in enumerate(final_tokens):
                    decoded = self.tokenizer.decode([ft])
                    logger.info(f"    [{idx}]: token={ft} ('{decoded}')")

            # ============================================================
            # UPDATE: Add accepted/corrected tokens to sequence and update caches
            # ============================================================
            #
            # OPTIMIZATION: The target cache already contains the draft tokens from
            # verification. We only need to:
            # 1. Keep the accepted tokens (rewind to N + accepted_count)
            # 2. If there's a corrected token, process just that one token
            # 3. If all tokens were accepted, use logits from verification output

            # Truncate final_tokens if they would exceed max_tokens
            remaining_capacity = max_tokens - len(generated_tokens)
            if len(final_tokens) > remaining_capacity:
                final_tokens = final_tokens[:remaining_capacity]
                # Recount accepted tokens after truncation
                # (truncation might cut off the corrected token or some accepted ones)
                accepted_count = min(accepted_count, len(final_tokens))

            if final_tokens:
                all_tokens.extend(final_tokens)
                generated_tokens.extend(final_tokens)

                # Update draft cache with all final tokens
                accepted_input = mx.array(final_tokens)[None, :]
                draft_output, draft_cache = get_logits_with_cache(
                    self.draft_model, accepted_input, draft_cache
                )
                mx.eval(draft_cache)
                mx.eval(draft_output)
                next_position_logits = draft_output[0, -1, :]
                mx.eval(next_position_logits)

                # For target cache: optimize based on whether all tokens were accepted
                all_accepted = accepted_count == len(final_tokens)

                if all_accepted:
                    # All draft tokens accepted - cache already has correct tokens!
                    # Just rewind to keep exactly len(final_tokens) and use existing logits
                    target_cache = rewind_cache(
                        target_cache,
                        target_cache_position_before_verify + len(final_tokens),
                    )
                    mx.eval(target_cache)

                    # Get next logits from the verification output
                    # target_output[0, K-1, :] predicts position N+K (next position)
                    # But we need to handle the case where K tokens were drafted
                    # and target_output has shape [1, K, vocab]
                    target_next_position_logits = target_output[
                        0, len(final_tokens) - 1, :
                    ]
                    mx.eval(target_next_position_logits)
                else:
                    # Some tokens rejected - need to fix the cache
                    # Rewind to keep only the accepted tokens (not the corrected one yet)
                    target_cache = rewind_cache(
                        target_cache,
                        target_cache_position_before_verify + accepted_count,
                    )
                    mx.eval(target_cache)

                    # Process only the corrected token to update cache and get next logits
                    corrected_token = final_tokens[-1]
                    corrected_input = mx.array([[corrected_token]])
                    target_correction_output, target_cache = get_logits_with_cache(
                        self.target_model, corrected_input, target_cache
                    )
                    mx.eval(target_cache)
                    mx.eval(target_correction_output)

                    target_next_position_logits = target_correction_output[0, -1, :]
                    mx.eval(target_next_position_logits)

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
            logger.info(f"Processing prompt {i + 1}/{len(prompts)}")

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
        1 for r in results for d in r.disagreements if d.is_high_confidence_failure
    )
    stats["high_confidence_failures"] = high_conf_failures

    if verbose:
        logger.info(
            f"Data collection complete: {total_tokens} tokens, "
            f"{total_disagreements} disagreements, "
            f"{stats['overall_acceptance_rate']:.1%} acceptance rate"
        )

    return results, stats
