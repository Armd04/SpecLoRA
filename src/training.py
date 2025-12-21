"""
LoRA Training Pipeline for Adaptive Draft Model Improvement

This module implements LoRA (Low-Rank Adaptation) fine-tuning for the draft model.
When the draft model has low acceptance rates on certain types of prompts,
we fine-tune it to better match the target model's outputs.

Key features:
- Memory-efficient LoRA with small rank (r=8) for 16GB RAM
- Gradient checkpointing for reduced memory usage
- Mixed training with failure cases and replay buffer
- Periodic checkpointing and evaluation
"""

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from .data_collector import TrainingExample

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""

    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]

    @property
    def scaling(self) -> float:
        """LoRA scaling factor."""
        return self.alpha / self.rank


class LoRALinear(nn.Module):
    """
    LoRA adapter for linear layers.

    LoRA decomposes weight updates into two low-rank matrices:
    W' = W + BA where B is (out_features, rank) and A is (rank, in_features)

    This dramatically reduces trainable parameters while maintaining
    model capacity for specific tasks.

    Supports both regular nn.Linear and nn.QuantizedLinear layers.
    For quantized layers, the base weights are dequantized and stored
    for the forward pass, then can be re-quantized during fusion.
    """

    def __init__(
        self,
        original_layer: nn.Module,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
    ):
        """
        Initialize LoRA adapter.

        Args:
            original_layer: The original linear layer to adapt (Linear or QuantizedLinear)
            rank: LoRA rank (lower = fewer params, less capacity)
            alpha: LoRA alpha for scaling
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = dropout
        self._training = True

        # Check if the layer is quantized
        self.is_quantized = hasattr(original_layer, "scales")

        if self.is_quantized:
            # Dequantize and store as regular weights for training
            # Store quantization parameters for later re-quantization
            self.quant_group_size = original_layer.group_size
            self.quant_bits = original_layer.bits

            # Dequantize the weights
            weight = mx.dequantize(
                original_layer.weight,
                original_layer.scales,
                original_layer.biases,
                original_layer.group_size,
                original_layer.bits,
            )
            self.weight_dtype = original_layer.scales.dtype

            # Store dequantized weight as frozen - use object.__setattr__ to bypass
            # MLX's parameter tracking so these won't be included in parameters()
            object.__setattr__(self, "_frozen_weight", weight)

            # Handle bias
            self._has_bias = hasattr(original_layer, "bias") and original_layer.bias is not None
            if self._has_bias:
                object.__setattr__(self, "_frozen_bias", original_layer.bias)
            else:
                object.__setattr__(self, "_frozen_bias", None)

            in_features = weight.shape[1]
            out_features = weight.shape[0]
        else:
            # Regular linear layer - extract and freeze weights
            self.quant_group_size = None
            self.quant_bits = None
            self.weight_dtype = original_layer.weight.dtype

            # Store frozen weights - use object.__setattr__ to bypass
            # MLX's parameter tracking so these won't be included in parameters()
            object.__setattr__(self, "_frozen_weight", original_layer.weight)

            self._has_bias = hasattr(original_layer, "bias") and original_layer.bias is not None
            if self._has_bias:
                object.__setattr__(self, "_frozen_bias", original_layer.bias)
            else:
                object.__setattr__(self, "_frozen_bias", None)

            in_features = original_layer.weight.shape[1]
            out_features = original_layer.weight.shape[0]

        self.in_features = in_features
        self.out_features = out_features

        # LoRA matrices: A projects down, B projects up
        # Initialize A with small random values scaled by 1/sqrt(rank)
        # Initialize B with zeros so adapter starts as identity (no change)
        self.lora_A = mx.random.normal((rank, in_features)) * (1.0 / math.sqrt(rank))
        self.lora_B = mx.zeros((out_features, rank))

        # Ensure LoRA parameters are evaluated
        mx.eval(self.lora_A, self.lora_B)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass with LoRA adaptation.

        Args:
            x: Input tensor

        Returns:
            Output with LoRA adaptation applied
        """
        # Base layer computation using frozen weights
        # Use stop_gradient to ensure no gradients flow through base weights
        frozen_weight = mx.stop_gradient(self._frozen_weight)
        original_output = x @ frozen_weight.T
        if self._frozen_bias is not None:
            frozen_bias = mx.stop_gradient(self._frozen_bias)
            original_output = original_output + frozen_bias

        # LoRA adaptation: x @ A.T @ B.T * scaling
        # Gradients only flow through lora_A and lora_B
        if self._training and self.dropout > 0:
            # Apply dropout during training
            mask = mx.random.bernoulli(1 - self.dropout, x.shape)
            x_dropped = x * mask / (1 - self.dropout)
            lora_output = (x_dropped @ self.lora_A.T) @ self.lora_B.T
        else:
            lora_output = (x @ self.lora_A.T) @ self.lora_B.T

        return original_output + lora_output * self.scaling

    def get_fused_weight(self) -> mx.array:
        """
        Get the fused weight (base + LoRA delta).

        Returns:
            Fused weight matrix
        """
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return self._frozen_weight + delta.astype(self.weight_dtype)

    def get_bias(self) -> Optional[mx.array]:
        """Get the bias if present."""
        return self._frozen_bias


def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
) -> nn.Module:
    """
    Apply LoRA adapters to specified layers in the model.

    Supports both regular Linear and QuantizedLinear layers.
    For QuantizedLinear, the weights are dequantized for training,
    then can be re-quantized during fusion.

    Args:
        model: The model to adapt
        config: LoRA configuration

    Returns:
        Model with LoRA adapters applied
    """
    applied_layers = []

    def apply_lora_recursive(module: nn.Module, path: str = "") -> None:
        """Recursively apply LoRA to matching layers."""
        for name, child in module.named_modules():
            full_path = f"{path}.{name}" if path else name

            # Check if this is a target module
            is_target = any(target in name for target in config.target_modules)

            # Support both Linear and QuantizedLinear
            is_linear = isinstance(child, nn.Linear)
            is_quantized_linear = isinstance(child, nn.QuantizedLinear)

            if is_target and (is_linear or is_quantized_linear):
                # Replace with LoRA-wrapped version
                lora_layer = LoRALinear(
                    child,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )

                # Set the layer in the parent module
                # Handle both attribute access and list indexing
                parent = module
                parts = name.split(".")
                for part in parts[:-1]:
                    if part.isdigit():
                        # Handle list indexing (e.g., layers.23 -> layers[23])
                        parent = parent[int(part)]
                    else:
                        parent = getattr(parent, part)

                # Set the final layer
                final_part = parts[-1]
                if final_part.isdigit():
                    parent[int(final_part)] = lora_layer
                else:
                    setattr(parent, final_part, lora_layer)

                layer_type = "QuantizedLinear" if is_quantized_linear else "Linear"
                logger.debug(f"Applied LoRA to {layer_type}: {full_path}")
                applied_layers.append(full_path)

    # Count original parameters
    original_params = 0
    for _, p in tree_flatten(model.parameters()):
        if hasattr(p, "size"):
            if p.dtype == mx.uint32:
                original_params += p.size * 8
            else:
                original_params += p.size

    apply_lora_recursive(model)

    # Force evaluation of new LoRA parameters
    mx.eval(model.parameters())

    # Count LoRA parameters
    lora_params = 0
    lora_layer_count = 0
    for _, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_params += module.lora_A.size + module.lora_B.size
            lora_layer_count += 1

    if lora_layer_count == 0:
        logger.warning(
            f"No LoRA layers were applied! Target modules: {config.target_modules}. "
            "Check that target module names match layer names in the model."
        )
    else:
        logger.info(
            f"LoRA applied to {lora_layer_count} layers. "
            f"Original params: {original_params / 1e6:.1f}M, "
            f"LoRA params: {lora_params / 1e6:.2f}M "
            f"({100 * lora_params / max(1, original_params):.2f}%)"
        )

    return model


def get_lora_parameters(model: nn.Module) -> Dict[str, mx.array]:
    """
    Extract only LoRA parameters from the model.

    Args:
        model: Model with LoRA adapters

    Returns:
        Dictionary of LoRA parameter names to values
    """
    lora_params = {}

    # Use named_modules() for recursive traversal - it yields (name, module) tuples
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_params[f"{name}.lora_A"] = module.lora_A
            lora_params[f"{name}.lora_B"] = module.lora_B

    return lora_params


@dataclass
class TrainingMetrics:
    """Metrics from a training run."""

    total_steps: int = 0
    total_loss: float = 0.0
    avg_loss: float = 0.0
    learning_rate: float = 0.0
    training_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "avg_loss": self.avg_loss,
            "learning_rate": self.learning_rate,
            "training_time": self.training_time_seconds,
        }


class LoRATrainer:
    """
    Trainer for LoRA fine-tuning of the draft model.

    Trains the draft model to better match the target model's outputs
    on cases where the draft model previously had low acceptance rates.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        lora_config: LoRAConfig,
        learning_rate: float = 1e-4,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 10,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "data/checkpoints",
    ):
        """
        Initialize the trainer.

        Args:
            model: Draft model to train
            tokenizer: Tokenizer for encoding
            lora_config: LoRA configuration
            learning_rate: Base learning rate
            batch_size: Batch size (keep small for 16GB RAM)
            gradient_accumulation_steps: Steps to accumulate gradients
            warmup_steps: LR warmup steps
            max_grad_norm: Maximum gradient norm for clipping
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model
        self.tokenizer = tokenizer
        self.lora_config = lora_config
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Apply LoRA to model
        self.model = apply_lora_to_model(model, lora_config)

        # Get trainable parameters (LoRA only)
        self.trainable_params = get_lora_parameters(self.model)

        # Log trainable parameter count
        lora_param_count = sum(p.size for p in self.trainable_params.values())
        logger.info(f"Trainable LoRA parameters: {lora_param_count:,}")

        # Setup optimizer (AdamW works well for LoRA)
        # Use lower weight decay for stability
        self.optimizer = optim.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        self.global_step = 0
        self.best_loss = float("inf")

    def _get_lr(self, step: int, total_steps: int) -> float:
        """
        Get learning rate with warmup and cosine decay.

        Args:
            step: Current step
            total_steps: Total training steps

        Returns:
            Learning rate for this step
        """
        if self.warmup_steps <= 0:
            # No warmup, use constant LR with cosine decay
            progress = step / max(1, total_steps)
            return self.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

        if step < self.warmup_steps:
            # Linear warmup - start from small fraction, not zero
            # Use (step + 1) to avoid zero LR at step 0
            return self.learning_rate * (step + 1) / self.warmup_steps
        else:
            # Cosine decay after warmup
            decay_steps = max(1, total_steps - self.warmup_steps)
            progress = (step - self.warmup_steps) / decay_steps
            return self.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    def _prepare_batch(
        self,
        examples: List[TrainingExample],
    ) -> Tuple[mx.array, mx.array]:
        """
        Prepare a training batch from examples.

        The training objective is to make the draft model output the same
        tokens as the target model. We use teacher forcing with cross-entropy loss.

        For each example with prompt [p0, p1, p2] and generation [g0, g1, g2]:
        - Full sequence: [p0, p1, p2, g0, g1, g2]
        - Input:  [p0, p1, p2, g0, g1]  (all but last)
        - Target: [p1, p2, g0, g1, g2]  (all but first, shifted)
        - Masked: [-100, -100, g0, g1, g2]  (mask prompt→prompt, keep prompt→gen transition!)

        Key insight: We mask positions 0 to len(prompt)-2 to avoid training on
        prompt reconstruction, BUT we keep position len(prompt)-1 which represents
        the critical transition from the last prompt token to the first generation
        token. This is essential for the model to learn how to start generating.

        Args:
            examples: List of training examples

        Returns:
            Tuple of (input_ids, target_ids) as MLX arrays
        """
        max_length = 512  # Configurable

        input_ids_list = []
        target_ids_list = []

        for ex in examples:
            # Prefer the exact prompt tokens used during generation (formatted chat prompt).
            # This avoids mismatch between "raw user prompt" and "chat-templated prompt",
            # and prevents tokenizer/template drift causing invalid training batches.
            if getattr(ex, "prompt_tokens", None):
                prompt_ids = list(ex.prompt_tokens)
            else:
                # Backward-compat: older datasets only store raw prompt text.
                prompt_ids = self.tokenizer.encode(ex.prompt)
            prompt_len = len(prompt_ids)

            # Target tokens are what the target model generated (not including prompt)
            # Take as many as will fit: max_length = prompt_len + generation_len
            available_space = (
                max_length - prompt_len - 1
            )  # -1 for at least one generation token
            if available_space <= 0:
                # Prompt is too long, skip this example
                logger.warning(
                    f"Skipping example {ex.id}: prompt length {prompt_len} "
                    f"exceeds max_length {max_length}"
                )
                continue

            # Get target tokens (the generation, not the prompt)
            target_tokens = ex.target_output[:available_space]

            # Skip examples with no target tokens (nothing to learn from)
            if len(target_tokens) == 0:
                logger.warning(
                    f"Skipping example {ex.id}: no target tokens to learn from"
                )
                continue

            # Teacher forcing with autoregressive next-token prediction:
            #
            # Full sequence: [prompt[0], prompt[1], ..., prompt[n-1], gen[0], gen[1], ..., gen[m-1]]
            #
            # Input:  [prompt[0], prompt[1], ..., prompt[n-1], gen[0], ..., gen[m-2]]
            # Target: [prompt[1], prompt[2], ..., gen[0],      gen[1], ..., gen[m-1]]
            #
            # Masking strategy:
            # - Mask positions 0 to n-2: prompt tokens predicting other prompt tokens (don't train)
            # - Position n-1 (last prompt) predicting gen[0]: TRAIN on this! (critical transition)
            # - Positions n onwards: generation tokens (train on all)
            #
            # Result after masking:
            # Target: [-100,      -100,      ..., gen[0], gen[1], ..., gen[m-1]]
            #          └─────── masked ─────────┘  └───── train on these ──────┘

            # Create full sequence (prompt + generation)
            full_sequence = prompt_ids + target_tokens

            # Standard teacher forcing: input is all but last, target is all but first
            full_input = full_sequence[:-1]
            targets = full_sequence[1:]

            # Mask prompt-to-prompt predictions (positions 0 to prompt_len-2)
            # But DO NOT mask position prompt_len-1, which is the critical prompt→generation transition
            for i in range(prompt_len - 1):
                targets[i] = -100

            # Safety: ensure there is at least one supervised target token.
            # If not, skip this example rather than producing a NaN loss later.
            num_valid_local = sum(1 for t in targets if t != -100)
            if num_valid_local == 0:
                logger.warning(
                    f"Skipping example {ex.id}: no valid target labels after masking "
                    f"(prompt_len={prompt_len}, target_len={len(target_tokens)})"
                )
                continue

            # Safety: validate token ID ranges to avoid undefined behavior / NaNs
            # if training data was collected with a different tokenizer.
            vocab_size = getattr(self.tokenizer, "vocab_size", None)
            if isinstance(vocab_size, int) and vocab_size > 0:
                max_input = max(full_input) if full_input else -1
                min_input = min(full_input) if full_input else 0
                # Only consider non-masked targets
                valid_targets = [t for t in targets if t != -100]
                max_tgt = max(valid_targets) if valid_targets else -1
                min_tgt = min(valid_targets) if valid_targets else 0
                if (
                    min_input < 0
                    or max_input >= vocab_size
                    or min_tgt < 0
                    or max_tgt >= vocab_size
                ):
                    logger.warning(
                        f"Skipping example {ex.id}: token id out of range for vocab_size={vocab_size} "
                        f"(input min/max={min_input}/{max_input}, target min/max={min_tgt}/{max_tgt}). "
                        "This usually indicates a tokenizer mismatch between data collection and training."
                    )
                    continue

            # Verify lengths match
            assert len(full_input) == len(targets), (
                f"Sequence length mismatch: input={len(full_input)}, "
                f"targets={len(targets)}, prompt_len={prompt_len}, "
                f"target_len={len(target_tokens)}"
            )

            # Pad to max length
            pad_length = max_length - len(full_input)
            if pad_length > 0:
                full_input = (
                    full_input + [self.tokenizer.pad_token_id or 0] * pad_length
                )
                targets = targets + [-100] * pad_length  # -100 = ignore in loss
            else:
                # Truncate if needed (shouldn't happen given our available_space check)
                full_input = full_input[:max_length]
                targets = targets[:max_length]

            input_ids_list.append(full_input)
            target_ids_list.append(targets)

        # Handle case where all examples were skipped
        if len(input_ids_list) == 0:
            logger.warning("All examples were skipped during batch preparation")
            # Return empty arrays with correct shape
            return (
                mx.array([], dtype=mx.int32).reshape(0, max_length),
                mx.array([], dtype=mx.int32).reshape(0, max_length),
            )

        return (
            mx.array(input_ids_list),
            mx.array(target_ids_list),
        )

    def _compute_loss(
        self,
        model: nn.Module,
        input_ids: mx.array,
        target_ids: mx.array,
    ) -> mx.array:
        """
        Compute cross-entropy loss.

        Args:
            model: The model to compute loss for (passed by nn.value_and_grad)
            input_ids: Input token IDs [batch, seq_len]
            target_ids: Target token IDs [batch, seq_len]

        Returns:
            Scalar loss value
        """
        # Forward pass
        logits = model(input_ids)

        # Reshape for cross-entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size)
        targets = target_ids.reshape(-1)

        # Create mask for non-padding tokens (targets == -100 are masked)
        mask = targets != -100

        # Check if we have any valid targets
        num_valid = mask.sum()
        if num_valid == 0:
            logger.warning(
                "No valid targets in batch (all masked with -100). "
                "This batch will be skipped."
            )
            # Return NaN so the training loop skips this batch
            # (consistent with how NaN losses from numerical instability are handled)
            return mx.array(float("nan"))

        # Replace masked positions with 0 (arbitrary valid index for gathering)
        targets = mx.where(mask, targets, mx.zeros_like(targets))

        # Cross-entropy loss with numerical stability
        # Subtract max for numerical stability before softmax
        logits_max = mx.max(logits, axis=-1, keepdims=True)
        logits_stable = logits - mx.stop_gradient(logits_max)
        log_probs = logits_stable - mx.logsumexp(logits_stable, axis=-1, keepdims=True)

        # Gather the log probabilities for target tokens
        target_log_probs = mx.take_along_axis(
            log_probs,
            targets[:, None],
            axis=-1,
        ).squeeze(-1)

        # Clamp to avoid extreme values
        target_log_probs = mx.clip(target_log_probs, a_min=-100.0, a_max=0.0)

        # Apply mask and compute mean loss over valid positions only
        masked_loss = -target_log_probs * mask
        loss = masked_loss.sum() / num_valid

        return loss

    def _filter_lora_gradients(self, grads: Dict[str, Any]) -> Dict[str, Any]:
        """
        Zero out non-LoRA gradients while preserving tree structure.

        This ensures only LoRA weights are updated, while maintaining
        the gradient tree structure that the optimizer expects.

        Args:
            grads: Full gradient tree from value_and_grad

        Returns:
            Gradient tree with non-LoRA gradients zeroed out
        """

        def zero_non_lora(grad_tree, path=""):
            """Recursively zero out non-LoRA gradients."""
            if isinstance(grad_tree, dict):
                result = {}
                for key, value in grad_tree.items():
                    new_path = f"{path}.{key}" if path else key
                    if key in ("lora_A", "lora_B"):
                        # Keep LoRA gradients as-is
                        result[key] = value
                    elif isinstance(value, dict):
                        # Recurse into nested dicts
                        result[key] = zero_non_lora(value, new_path)
                    elif isinstance(value, mx.array):
                        # Zero out non-LoRA array gradients
                        result[key] = mx.zeros_like(value)
                    else:
                        # Keep other values as-is
                        result[key] = value
                return result
            elif isinstance(grad_tree, mx.array):
                # Zero out any standalone arrays that aren't LoRA
                return mx.zeros_like(grad_tree)
            return grad_tree

        return zero_non_lora(grads)

    def _training_step(
        self,
        input_ids: mx.array,
        target_ids: mx.array,
    ) -> Tuple[float, Dict[str, mx.array]]:
        """
        Perform a single training step.

        Args:
            input_ids: Input batch
            target_ids: Target batch

        Returns:
            Tuple of (loss_value, gradients)
        """
        # Create a loss function that we can differentiate
        def loss_fn(model):
            return self._compute_loss(model, input_ids, target_ids)

        # Compute loss and gradients
        loss, grads = nn.value_and_grad(self.model, loss_fn)(self.model)

        # Evaluate loss and gradients to ensure computation is complete
        mx.eval(loss)
        grad_arrays = [v for _, v in tree_flatten(grads) if isinstance(v, mx.array)]
        if grad_arrays:
            mx.eval(*grad_arrays)

        # Debug: Check gradient structure on first step
        if self.global_step == 0:
            grad_names = [name for name, _ in tree_flatten(grads)]
            lora_grads = [n for n in grad_names if "lora" in n.lower()]
            logger.debug(f"Total gradient entries: {len(grad_names)}, LoRA gradients: {len(lora_grads)}")

        return loss.item(), grads

    def train(
        self,
        training_examples: List[TrainingExample],
        num_epochs: int = 3,
        eval_callback: Optional[Callable] = None,
        save_every_n_steps: int = 50,
    ) -> TrainingMetrics:
        """
        Train the draft model on collected examples.

        Args:
            training_examples: List of training examples
            num_epochs: Number of training epochs
            eval_callback: Optional callback for evaluation
            save_every_n_steps: How often to save checkpoints

        Returns:
            Training metrics
        """
        start_time = time.time()
        metrics = TrainingMetrics()

        num_examples = len(training_examples)
        steps_per_epoch = math.ceil(num_examples / self.batch_size)
        total_steps = steps_per_epoch * num_epochs

        logger.info(
            f"Starting LoRA training: {num_examples} examples, "
            f"{num_epochs} epochs, {total_steps} total steps"
        )

        accumulated_grads = None
        accumulated_loss = 0.0
        valid_steps_in_accumulation = 0  # Track how many valid steps we've accumulated

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            skipped_batches = 0

            # Shuffle examples each epoch
            import random

            shuffled = training_examples.copy()
            random.shuffle(shuffled)

            for step in range(steps_per_epoch):
                # Get batch
                start_idx = step * self.batch_size
                end_idx = min(start_idx + self.batch_size, num_examples)
                batch_examples = shuffled[start_idx:end_idx]

                if not batch_examples:
                    continue

                # Prepare batch
                input_ids, target_ids = self._prepare_batch(batch_examples)

                # Skip empty batches (can happen if all examples were filtered out)
                if input_ids.shape[0] == 0:
                    continue

                # Training step
                loss, grads = self._training_step(input_ids, target_ids)

                # Check for NaN/Inf in loss
                if not mx.isfinite(mx.array(loss)).item():
                    logger.warning(
                        f"Non-finite loss detected: {loss}. Skipping this batch. "
                        f"This may indicate numerical instability."
                    )
                    skipped_batches += 1
                    # Skip this batch
                    continue

                # Check for NaN/Inf in gradients
                has_nan_grad = False
                for name, g in tree_flatten(grads):
                    if hasattr(g, "size") and g.size > 0:
                        if not mx.all(mx.isfinite(g)).item():
                            logger.warning(
                                f"Non-finite gradient detected in {name}. Skipping this batch."
                            )
                            has_nan_grad = True
                            break

                if has_nan_grad:
                    skipped_batches += 1
                    # Skip this batch if gradients are invalid
                    continue

                # Accumulate gradients
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = tree_map(
                        lambda a, b: a + b,
                        accumulated_grads,
                        grads,
                    )
                accumulated_loss += loss
                valid_steps_in_accumulation += 1

                # Force evaluation to prevent computation graph from growing
                grad_arrays = [v for _, v in tree_flatten(accumulated_grads) if isinstance(v, mx.array)]
                if grad_arrays:
                    mx.eval(*grad_arrays)

                # Update weights after accumulation steps
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Only update if we have valid gradients
                    if valid_steps_in_accumulation == 0:
                        logger.warning(
                            "No valid gradients in accumulation window. Skipping update."
                        )
                        accumulated_grads = None
                        accumulated_loss = 0.0
                        continue

                    # Average gradients over valid steps only
                    accumulated_grads = tree_map(
                        lambda g: g / valid_steps_in_accumulation,
                        accumulated_grads,
                    )

                    # Gradient clipping
                    grad_sq_sum = 0.0
                    for _, g in tree_flatten(accumulated_grads):
                        if hasattr(g, "size"):
                            grad_sq_sum += mx.sum(g**2).item()
                    grad_norm = grad_sq_sum**0.5

                    if grad_norm > self.max_grad_norm:
                        scale = self.max_grad_norm / grad_norm
                        accumulated_grads = tree_map(
                            lambda g: g * scale,
                            accumulated_grads,
                        )

                    # Update learning rate
                    current_lr = self._get_lr(self.global_step, total_steps)
                    self.optimizer.learning_rate = current_lr

                    # Apply gradients - only update LoRA parameters
                    # Filter gradients to only include LoRA parameters
                    lora_grads = self._filter_lora_gradients(accumulated_grads)
                    self.optimizer.update(self.model, lora_grads)
                    mx.eval(self.model.parameters())

                    # Validate LoRA weights are still finite after update
                    lora_weights_valid = True
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoRALinear):
                            if not mx.all(mx.isfinite(module.lora_A)).item():
                                logger.warning(f"NaN detected in lora_A of {name} after update")
                                lora_weights_valid = False
                            if not mx.all(mx.isfinite(module.lora_B)).item():
                                logger.warning(f"NaN detected in lora_B of {name} after update")
                                lora_weights_valid = False

                    if not lora_weights_valid:
                        logger.error("LoRA weights became NaN after update. Training may be unstable.")

                    # Log progress
                    avg_accumulated_loss = (
                        accumulated_loss / valid_steps_in_accumulation
                    )

                    if self.global_step % 10 == 0:
                        # Count non-zero gradients for diagnostics
                        lora_grad_count = 0
                        for name, g in tree_flatten(lora_grads):
                            if isinstance(g, mx.array) and "lora" in name.lower():
                                lora_grad_count += 1

                        logger.info(
                            f"Epoch {epoch + 1}/{num_epochs}, "
                            f"Step {self.global_step}/{total_steps}, "
                            f"Loss: {avg_accumulated_loss:.4f}, "
                            f"Grad Norm: {grad_norm:.4f}, "
                            f"LR: {current_lr:.2e}, "
                            f"LoRA grads: {lora_grad_count}"
                        )

                    epoch_loss += avg_accumulated_loss
                    metrics.total_loss += avg_accumulated_loss

                    # Reset accumulation
                    accumulated_grads = None
                    accumulated_loss = 0.0
                    valid_steps_in_accumulation = 0

                    self.global_step += 1
                    metrics.total_steps = self.global_step

                    # Save checkpoint
                    if self.global_step % save_every_n_steps == 0:
                        self.save_checkpoint(f"step_{self.global_step}")

            # End of epoch
            avg_epoch_loss = epoch_loss / max(
                1, steps_per_epoch // self.gradient_accumulation_steps
            )

            # Report skipped batches
            if skipped_batches > 0:
                logger.warning(
                    f"Epoch {epoch + 1}: Skipped {skipped_batches} batches due to "
                    f"NaN/Inf values ({skipped_batches / steps_per_epoch * 100:.1f}%)"
                )

            logger.info(
                f"Epoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}"
            )

            # Evaluation callback
            if eval_callback:
                eval_callback(epoch, avg_epoch_loss)

            # Save best model
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                self.save_checkpoint("best")

        # Final metrics
        metrics.training_time_seconds = time.time() - start_time
        metrics.avg_loss = metrics.total_loss / max(1, metrics.total_steps)
        metrics.learning_rate = self.learning_rate

        # Save final checkpoint
        self.save_checkpoint("final")

        logger.info(
            f"Training complete. Total time: {metrics.training_time_seconds:.1f}s, "
            f"Average loss: {metrics.avg_loss:.4f}"
        )

        return metrics

    def save_checkpoint(self, name: str) -> str:
        """
        Save a training checkpoint in MLX-LM compatible format.

        Saves adapters.safetensors with MLX-LM naming conventions (lora_a, lora_b)
        and adapter_config.json for native MLX-LM loading.

        Args:
            name: Checkpoint name

        Returns:
            Path to saved checkpoint
        """
        import json

        checkpoint_path = self.checkpoint_dir / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Get LoRA parameters
        lora_params = get_lora_parameters(self.model)

        # Convert naming to MLX-LM format: lora_A -> lora_a, lora_B -> lora_b
        mlx_lora_params = {}
        for key, value in lora_params.items():
            new_key = key.replace(".lora_A", ".lora_a").replace(".lora_B", ".lora_b")
            mlx_lora_params[new_key] = value

        mx.save_safetensors(
            str(checkpoint_path / "adapters.safetensors"),
            mlx_lora_params,
        )

        # Save MLX-LM compatible adapter config
        adapter_config = {
            "fine_tune_type": "lora",
            "num_layers": -1,  # All layers
            "lora_parameters": {
                "rank": self.lora_config.rank,
                "scale": float(
                    self.lora_config.alpha
                ),  # MLX-LM uses 'scale' not 'alpha'
                "dropout": self.lora_config.dropout,
            },
        }

        with open(checkpoint_path / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=2)

        # Also save trainer state for resume capability
        state = {
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "lora_config": {
                "rank": self.lora_config.rank,
                "alpha": self.lora_config.alpha,
                "dropout": self.lora_config.dropout,
                "target_modules": self.lora_config.target_modules,
            },
        }

        with open(checkpoint_path / "trainer_state.json", "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a training checkpoint.

        Handles backward compatibility for both old (lora_A/lora_B) and
        new (lora_a/lora_b) naming conventions.

        Args:
            checkpoint_path: Path to checkpoint directory
        """
        checkpoint_path = Path(checkpoint_path)

        # Load LoRA parameters
        lora_params = mx.load(str(checkpoint_path / "adapters.safetensors"))

        # Normalize parameter names for backward compatibility
        # Support both MLX-LM format (lora_a/lora_b) and our format (lora_A/lora_B)
        normalized_params = {}
        for name, param in lora_params.items():
            # Convert MLX-LM format to our format
            new_name = name.replace(".lora_a", ".lora_A").replace(".lora_b", ".lora_B")
            normalized_params[new_name] = param

        # Apply to LoRALinear modules in the model
        loaded_count = 0
        for module_name, module in self.model.named_modules():
            if isinstance(module, LoRALinear):
                lora_a_key = f"{module_name}.lora_A"
                lora_b_key = f"{module_name}.lora_B"

                if lora_a_key in normalized_params:
                    module.lora_A = normalized_params[lora_a_key]
                    loaded_count += 1
                if lora_b_key in normalized_params:
                    module.lora_B = normalized_params[lora_b_key]

        if loaded_count == 0:
            logger.warning(
                "No LoRA parameters were loaded. Ensure the model has LoRA layers "
                "applied before loading a checkpoint."
            )

        mx.eval(self.model.parameters())

        # Load training state
        import json

        state_path = checkpoint_path / "trainer_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                state = json.load(f)

            self.global_step = state.get("global_step", 0)
            self.best_loss = state.get("best_loss", float("inf"))
        else:
            logger.warning(f"trainer_state.json not found in {checkpoint_path}")

        logger.info(
            f"Loaded checkpoint from: {checkpoint_path} ({loaded_count} LoRA layers)"
        )

    def fuse_and_get_model(self) -> nn.Module:
        """
        Fuse LoRA weights into base model and return a clean model for inference.

        This method:
        1. Merges LoRA adaptations into original layer weights
        2. Replaces LoRALinear wrappers with clean nn.Linear layers
        3. Returns model ready for efficient inference (no LoRA overhead)

        Returns:
            nn.Module: Clean model with fused weights, no LoRA wrappers
        """
        from mlx.utils import tree_unflatten

        fused_layers = []

        # Use named_modules() for recursive traversal - yields (name, module) tuples
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear):
                # Create fused linear layer
                fused_linear = self._fuse_lora_linear(module)
                fused_layers.append((name, fused_linear))

        # Replace LoRA layers with fused clean layers
        if fused_layers:
            self.model.update_modules(tree_unflatten(fused_layers))

        # Ensure model weights are evaluated
        mx.eval(self.model.parameters())

        logger.info(f"Fused {len(fused_layers)} LoRA layers into base model")

        return self.model

    def _fuse_lora_linear(self, lora_layer: LoRALinear) -> nn.Module:
        """
        Fuse a single LoRALinear layer into a clean nn.Linear or nn.QuantizedLinear.

        Handles both quantized and non-quantized layers.

        Args:
            lora_layer: The LoRA-wrapped linear layer

        Returns:
            nn.Linear or nn.QuantizedLinear: Clean linear layer with fused weights
        """
        # Get fused weight using the LoRALinear's method
        fused_weight = lora_layer.get_fused_weight()
        bias = lora_layer.get_bias()
        has_bias = bias is not None

        # Create clean linear layer
        out_features, in_features = fused_weight.shape
        fused_linear = nn.Linear(in_features, out_features, bias=has_bias)
        fused_linear.weight = fused_weight

        if has_bias:
            fused_linear.bias = bias

        # Re-quantize if original was quantized
        if lora_layer.is_quantized:
            fused_linear = nn.QuantizedLinear.from_linear(
                fused_linear,
                lora_layer.quant_group_size,
                lora_layer.quant_bits,
            )

        return fused_linear


def quick_lora_finetune(
    model: nn.Module,
    tokenizer: Any,
    examples: List[TrainingExample],
    lora_rank: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    checkpoint_dir: str = "data/checkpoints",
) -> Tuple[nn.Module, TrainingMetrics]:
    """
    Quick helper function to run LoRA fine-tuning.

    Args:
        model: Model to fine-tune
        tokenizer: Tokenizer
        examples: Training examples
        lora_rank: LoRA rank
        num_epochs: Training epochs
        learning_rate: Learning rate
        checkpoint_dir: Checkpoint directory

    Returns:
        Tuple of (trained_model, metrics)
    """
    config = LoRAConfig(
        rank=lora_rank,
        alpha=lora_rank * 2,
        dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    trainer = LoRATrainer(
        model=model,
        tokenizer=tokenizer,
        lora_config=config,
        learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir,
    )

    metrics = trainer.train(examples, num_epochs=num_epochs)

    return trainer.model, metrics
