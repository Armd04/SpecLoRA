"""
Quick test to diagnose the NaN loss issue in training.
"""

import logging
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from src.training import LoRATrainer, LoRAConfig
from src.data_collector import TrainingExample
from datetime import datetime

logging.basicConfig(level=logging.DEBUG)

# Create some dummy training examples
print("Creating dummy training examples...")
examples = [
    TrainingExample(
        id="test_1",
        prompt="What is Python?",
        draft_output=[128000, 3923, 374, 13325, 30, 271, 31380, 374, 264, 1579],  # dummy tokens
        target_output=[128000, 3923, 374, 13325, 30, 271, 31380, 374, 264, 1579, 11036, 15840],  # slightly different
        acceptance_rate=0.4,
        timestamp=datetime.now().isoformat(),
        is_failure=True,
    ),
    TrainingExample(
        id="test_2",
        prompt="Explain AI.",
        draft_output=[128000, 849, 21435, 15592, 13, 271, 15836, 11478],  # dummy tokens
        target_output=[128000, 849, 21435, 15592, 13, 271, 15836, 11478, 374, 279],  # slightly different
        acceptance_rate=0.3,
        timestamp=datetime.now().isoformat(),
        is_failure=True,
    ),
]

# Load a small model for testing
print("Loading draft model...")
model_name = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
model, tokenizer = load(model_name, lazy=True)
mx.eval(model.parameters())

print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
print(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}")

# Setup LoRA config
lora_config = LoRAConfig(
    rank=8,
    alpha=16,
    dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)

# Initialize trainer
print("\nInitializing trainer...")
trainer = LoRATrainer(
    model=model,
    tokenizer=tokenizer,
    lora_config=lora_config,
    learning_rate=1e-4,
    batch_size=1,
    gradient_accumulation_steps=2,
    warmup_steps=5,
    checkpoint_dir="data/checkpoints/test",
)

print("\nPreparing first batch to debug...")
# Prepare a batch to see what happens
input_ids, target_ids = trainer._prepare_batch([examples[0]])

print(f"Input IDs shape: {input_ids.shape}")
print(f"Target IDs shape: {target_ids.shape}")
print(f"Input IDs sample: {input_ids[0, :20]}")
print(f"Target IDs sample: {target_ids[0, :20]}")
print(f"Input IDs min/max: {input_ids.min()}/{input_ids.max()}")
print(f"Target IDs min/max (excluding -100): {mx.where(target_ids != -100, target_ids, mx.zeros_like(target_ids)).max()}")

# Try forward pass
print("\nTrying forward pass...")
try:
    logits = trainer.model(input_ids)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits dtype: {logits.dtype}")
    print(f"Logits has NaN: {mx.any(mx.isnan(logits)).item()}")
    print(f"Logits has Inf: {mx.any(mx.isinf(logits)).item()}")
    print(f"Logits min/max: {logits.min()}/{logits.max()}")

    # Compute loss
    print("\nComputing loss...")
    loss = trainer._compute_loss(input_ids, target_ids)
    print(f"Loss: {loss.item()}")
    print(f"Loss is NaN: {mx.isnan(loss).item()}")
    print(f"Loss is Inf: {mx.isinf(loss).item()}")

except Exception as e:
    print(f"Error during forward pass or loss computation: {e}")
    import traceback
    traceback.print_exc()

# Try a full training step
print("\nTrying full training...")
try:
    metrics = trainer.train(training_examples=examples, num_epochs=1)
    print(f"\nTraining complete!")
    print(f"Average loss: {metrics.avg_loss}")
    print(f"Total steps: {metrics.total_steps}")
except Exception as e:
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()
