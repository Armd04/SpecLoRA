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
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
    ):
        """
        Initialize LoRA adapter.
        
        Args:
            original_layer: The original linear layer to adapt
            rank: LoRA rank (lower = fewer params, less capacity)
            alpha: LoRA alpha for scaling
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_layer.weight.shape[1]
        out_features = original_layer.weight.shape[0]
        
        # LoRA matrices: A projects down, B projects up
        # Initialize A with small random values, B with zeros
        # This means the adapter starts as identity (no change)
        self.lora_A = mx.random.normal((rank, in_features)) * 0.01
        self.lora_B = mx.zeros((out_features, rank))
        
        self.dropout = dropout
        self._training = True
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass with LoRA adaptation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output with LoRA adaptation applied
        """
        # Original layer output
        original_output = self.original_layer(x)
        
        # LoRA adaptation: x @ A.T @ B.T * scaling
        if self._training and self.dropout > 0:
            # Apply dropout during training
            mask = mx.random.bernoulli(1 - self.dropout, x.shape)
            x_dropped = x * mask / (1 - self.dropout)
            lora_output = (x_dropped @ self.lora_A.T) @ self.lora_B.T
        else:
            lora_output = (x @ self.lora_A.T) @ self.lora_B.T
        
        return original_output + lora_output * self.scaling
    
    def merge_weights(self) -> None:
        """
        Merge LoRA weights into the original layer.
        
        This is useful for inference when you want to use the adapted
        model without LoRA overhead.
        """
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        self.original_layer.weight = self.original_layer.weight + delta_w
    
    def parameters(self) -> Dict[str, mx.array]:
        """Return only the trainable LoRA parameters."""
        return {
            "lora_A": self.lora_A,
            "lora_B": self.lora_B,
        }


def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
) -> nn.Module:
    """
    Apply LoRA adapters to specified layers in the model.
    
    Args:
        model: The model to adapt
        config: LoRA configuration
        
    Returns:
        Model with LoRA adapters applied
    """
    def apply_lora_recursive(module: nn.Module, path: str = "") -> None:
        """Recursively apply LoRA to matching layers."""
        for name, child in module.named_modules():
            full_path = f"{path}.{name}" if path else name
            
            # Check if this is a target module
            is_target = any(
                target in name for target in config.target_modules
            )
            
            if is_target and isinstance(child, nn.Linear):
                # Replace with LoRA-wrapped version
                lora_layer = LoRALinear(
                    child,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
                
                # Set the layer in the parent module
                parent = module
                parts = name.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], lora_layer)
                
                logger.debug(f"Applied LoRA to: {full_path}")
    
    # Count original parameters
    original_params = sum(p.size for p in tree_flatten(model.parameters())[0])
    
    apply_lora_recursive(model)
    
    # Count LoRA parameters
    lora_params = 0
    for name, param in tree_flatten(model.parameters())[0]:
        if "lora" in str(name).lower():
            lora_params += param.size
    
    logger.info(
        f"LoRA applied. Original params: {original_params/1e6:.1f}M, "
        f"LoRA params: {lora_params/1e6:.2f}M ({100*lora_params/original_params:.2f}%)"
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
    
    def extract_recursive(module: nn.Module, prefix: str = "") -> None:
        for name, child in module._modules.items():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, LoRALinear):
                lora_params[f"{full_name}.lora_A"] = child.lora_A
                lora_params[f"{full_name}.lora_B"] = child.lora_B
            elif hasattr(child, "_modules"):
                extract_recursive(child, full_name)
    
    extract_recursive(model)
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
        
        # Setup optimizer (AdamW works well for LoRA)
        self.optimizer = optim.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01,
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
        if step < self.warmup_steps:
            # Linear warmup
            return self.learning_rate * step / self.warmup_steps
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (total_steps - self.warmup_steps)
            return self.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
    
    def _prepare_batch(
        self,
        examples: List[TrainingExample],
    ) -> Tuple[mx.array, mx.array]:
        """
        Prepare a training batch from examples.
        
        The training objective is to make the draft model output the same
        tokens as the target model. We use teacher forcing with cross-entropy loss.
        
        Args:
            examples: List of training examples
            
        Returns:
            Tuple of (input_ids, target_ids) as MLX arrays
        """
        max_length = 512  # Configurable
        
        input_ids_list = []
        target_ids_list = []
        
        for ex in examples:
            # Encode prompt
            prompt_ids = self.tokenizer.encode(ex.prompt)
            
            # Target is what the target model would have generated
            target_tokens = ex.target_output[:max_length - len(prompt_ids)]
            
            # Input is prompt + target (for teacher forcing)
            full_input = prompt_ids + target_tokens[:-1]  # Shift right
            targets = prompt_ids[1:] + target_tokens  # Shift left
            
            # Pad to max length
            pad_length = max_length - len(full_input)
            if pad_length > 0:
                full_input = full_input + [self.tokenizer.pad_token_id or 0] * pad_length
                targets = targets + [-100] * pad_length  # -100 = ignore in loss
            else:
                full_input = full_input[:max_length]
                targets = targets[:max_length]
            
            input_ids_list.append(full_input)
            target_ids_list.append(targets)
        
        return (
            mx.array(input_ids_list),
            mx.array(target_ids_list),
        )
    
    def _compute_loss(
        self,
        input_ids: mx.array,
        target_ids: mx.array,
    ) -> mx.array:
        """
        Compute cross-entropy loss.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            target_ids: Target token IDs [batch, seq_len]
            
        Returns:
            Scalar loss value
        """
        # Forward pass
        logits = self.model(input_ids)
        
        # Reshape for cross-entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size)
        targets = target_ids.reshape(-1)
        
        # Create mask for non-padding tokens
        mask = targets != -100
        targets = mx.where(mask, targets, mx.zeros_like(targets))
        
        # Cross-entropy loss
        log_probs = nn.log_softmax(logits, axis=-1)
        
        # Gather the log probabilities for target tokens
        target_log_probs = mx.take_along_axis(
            log_probs,
            targets[:, None],
            axis=-1,
        ).squeeze(-1)
        
        # Apply mask and compute mean loss
        masked_loss = -target_log_probs * mask
        loss = masked_loss.sum() / mask.sum()
        
        return loss
    
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
        # Compute loss and gradients
        loss_and_grad_fn = nn.value_and_grad(self.model, self._compute_loss)
        loss, grads = loss_and_grad_fn(input_ids, target_ids)
        
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
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
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
                
                # Training step
                loss, grads = self._training_step(input_ids, target_ids)
                
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
                
                # Update weights after accumulation steps
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Average gradients
                    accumulated_grads = tree_map(
                        lambda g: g / self.gradient_accumulation_steps,
                        accumulated_grads,
                    )
                    
                    # Gradient clipping
                    grad_norm = sum(
                        mx.sum(g ** 2).item()
                        for g in tree_flatten(accumulated_grads)[0]
                    ) ** 0.5
                    
                    if grad_norm > self.max_grad_norm:
                        scale = self.max_grad_norm / grad_norm
                        accumulated_grads = tree_map(
                            lambda g: g * scale,
                            accumulated_grads,
                        )
                    
                    # Update learning rate
                    current_lr = self._get_lr(self.global_step, total_steps)
                    self.optimizer.learning_rate = current_lr
                    
                    # Apply gradients
                    self.optimizer.update(self.model, accumulated_grads)
                    mx.eval(self.model.parameters())
                    
                    # Log progress
                    avg_accumulated_loss = accumulated_loss / self.gradient_accumulation_steps
                    
                    if self.global_step % 10 == 0:
                        logger.info(
                            f"Epoch {epoch+1}/{num_epochs}, "
                            f"Step {self.global_step}/{total_steps}, "
                            f"Loss: {avg_accumulated_loss:.4f}, "
                            f"LR: {current_lr:.2e}"
                        )
                    
                    epoch_loss += avg_accumulated_loss
                    metrics.total_loss += avg_accumulated_loss
                    
                    # Reset accumulation
                    accumulated_grads = None
                    accumulated_loss = 0.0
                    
                    self.global_step += 1
                    metrics.total_steps = self.global_step
                    
                    # Save checkpoint
                    if self.global_step % save_every_n_steps == 0:
                        self.save_checkpoint(f"step_{self.global_step}")
            
            # End of epoch
            avg_epoch_loss = epoch_loss / max(1, steps_per_epoch // self.gradient_accumulation_steps)
            logger.info(f"Epoch {epoch+1} complete. Average loss: {avg_epoch_loss:.4f}")
            
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
        Save a training checkpoint.
        
        Args:
            name: Checkpoint name
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA parameters
        lora_params = get_lora_parameters(self.model)
        mx.save_safetensors(
            str(checkpoint_path / "adapters.safetensors"),
            lora_params,
        )
        
        # Save training state
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
        
        import json
        with open(checkpoint_path / "trainer_state.json", "w") as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Load LoRA parameters
        lora_params = mx.load(str(checkpoint_path / "adapters.safetensors"))
        
        # Apply to model
        for name, param in lora_params.items():
            parts = name.split(".")
            module = self.model
            for part in parts[:-1]:
                module = getattr(module, part)
            setattr(module, parts[-1], param)
        
        mx.eval(self.model.parameters())
        
        # Load training state
        import json
        with open(checkpoint_path / "trainer_state.json", "r") as f:
            state = json.load(f)
        
        self.global_step = state["global_step"]
        self.best_loss = state["best_loss"]
        
        logger.info(f"Loaded checkpoint from: {checkpoint_path}")


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
