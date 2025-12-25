# API Reference

Using SpecLoRA programmatically in your Python code.

## Quick Start

```python
from src.main import SpeculativeDecodingSystem
import yaml

# Load config
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

# Initialize system
system = SpeculativeDecodingSystem(config)
system.initialize()

# Generate text
response = system.generate("What is machine learning?")
print(response)
```

## SpeculativeDecodingSystem

Main class that orchestrates all components.

### Initialization

```python
system = SpeculativeDecodingSystem(config: dict)
system.initialize()
```

**Parameters:**
- `config`: Dictionary from `config.yaml`

**Methods:**
- `initialize()`: Loads models, creates decoders. Call once before use.

### Text Generation

```python
response = system.generate(
    prompt: str,
    mode: str = "fast",
    max_tokens: int = None,
    temperature: float = None
) -> str
```

**Parameters:**
- `prompt`: User prompt (will be formatted with chat template)
- `mode`: "fast" or "detailed"
- `max_tokens`: Override config value
- `temperature`: Override config value

**Returns:**
- Generated text (string)

**Example:**
```python
# Fast mode (production)
answer = system.generate("Explain Python in one sentence.", mode="fast")

# Detailed mode (collect training data)
answer = system.generate(
    "Explain Python in detail.",
    mode="detailed",
    max_tokens=1024
)
```

### Training

```python
system.train(
    num_epochs: int = None,
    learning_rate: float = None
) -> dict
```

**Parameters:**
- `num_epochs`: Override config value
- `learning_rate`: Override config value

**Returns:**
- Dictionary with training results:
  ```python
  {
      'final_loss': 0.543,
      'best_loss': 0.521,
      'global_step': 150,
      'checkpoint_path': 'data/checkpoints/best/'
  }
  ```

**Example:**
```python
# Train with defaults from config
results = system.train()
print(f"Final loss: {results['final_loss']:.3f}")

# Override hyperparameters
results = system.train(num_epochs=5, learning_rate=5e-5)
```

### Statistics

```python
stats = system.get_stats() -> dict
```

**Returns:**
- Dictionary with current stats:
  ```python
  {
      'generation_count': 42,
      'tracker': {
          'recent_average': 0.67,  # Recent acceptance rate
          'overall_average': 0.65,
          'total_samples': 42
      },
      'failure_count': 15,
      'replay_buffer_size': 8,
      'adapter_loaded': True,
      'adapter_path': 'data/checkpoints/best/'
  }
  ```

**Example:**
```python
stats = system.get_stats()
print(f"Acceptance rate: {stats['tracker']['recent_average']:.1%}")
print(f"Failures collected: {stats['failure_count']}")
```

### Adapter Management

```python
# Load adapter
system.load_adapter(path: str = "best") -> bool

# Unload adapter
system.unload_adapter() -> bool

# Get adapter info
info = system.get_adapter_info() -> dict
```

**Parameters:**
- `path`: Path to checkpoint directory or alias ("best", "final", "latest")

**Returns:**
- Boolean success status for load/unload
- Dictionary with adapter info:
  ```python
  {
      'loaded': True,
      'path': '/full/path/to/checkpoint',
      'rank': 8,
      'alpha': 16,
      'target_modules': ['q_proj', 'v_proj']
  }
  ```

**Example:**
```python
# Load best checkpoint
if system.load_adapter("best"):
    print("Adapter loaded successfully")

# Get info
info = system.get_adapter_info()
if info['loaded']:
    print(f"Using rank-{info['rank']} adapter from {info['path']}")

# Unload to test base model
system.unload_adapter()
```

### Batch Processing

```python
results = system.batch_generate(
    prompts: list[str],
    mode: str = "fast",
    show_progress: bool = True
) -> list[str]
```

**Parameters:**
- `prompts`: List of prompts to process
- `mode`: "fast" or "detailed"
- `show_progress`: Show progress bar (requires tqdm)

**Returns:**
- List of generated responses

**Example:**
```python
prompts = [
    "What is Python?",
    "What is JavaScript?",
    "What is Rust?"
]

responses = system.batch_generate(prompts, mode="detailed")
for prompt, response in zip(prompts, responses):
    print(f"Q: {prompt}")
    print(f"A: {response}\n")
```

## Lower-Level APIs

### ModelManager

Direct model and adapter management.

```python
from src.models import ModelManager

manager = ModelManager(config)
target_model, draft_model = manager.load_models()

# Load LoRA adapter
manager.load_lora_adapter(
    model=draft_model,
    adapter_path="data/checkpoints/best/",
    fuse=True  # True for inference, False for training
)
```

### SpeculativeDecoder (Fast Mode)

```python
from src.speculative import SpeculativeDecoder

decoder = SpeculativeDecoder(
    target_model=target_model,
    draft_model=draft_model,
    tokenizer=tokenizer,
    num_draft_tokens=4,
    temperature=0.7
)

result = decoder.generate(
    prompt="What is AI?",
    max_tokens=512
)

print(result.text)
print(f"Acceptance rate: {result.acceptance_rate:.1%}")
print(f"Draft tokens: {result.num_draft_tokens}")
print(f"Accepted: {result.num_accepted_tokens}")
```

### ManualSpeculativeDecoder (Detailed Mode)

```python
from src.speculative_manual import ManualSpeculativeDecoder

decoder = ManualSpeculativeDecoder(
    target_model=target_model,
    draft_model=draft_model,
    tokenizer=tokenizer,
    num_draft_tokens=4,
    temperature=0.7
)

result = decoder.generate(
    prompt="What is AI?",
    max_tokens=512
)

print(result.text)
print(f"Acceptance rate: {result.acceptance_rate:.1%}")

# Access token-level disagreements
for d in result.disagreements:
    print(f"Position {d.position}: draft={d.draft_token}, target={d.target_token}")
    print(f"  Confidences: draft={d.draft_confidence:.2f}, target={d.target_confidence:.2f}")
```

### DataCollector

```python
from src.data_collector import DataCollector, TrainingExample

collector = DataCollector(
    data_dir="data/failures/",
    replay_buffer_max_size=100,
    acceptance_threshold=0.5
)

# Add failure case
example = TrainingExample(
    id="unique_id",
    prompt="User prompt",
    target_output=result.text,
    acceptance_rate=result.acceptance_rate,
    prompt_tokens=tokenizer.encode(prompt),
    disagreements=result.disagreements,
    timestamp="2024-01-01T00:00:00"
)

collector.add_failure(example)

# Get training data
failures = collector.load_failures()
replay_buffer = collector.load_replay_buffer()

print(f"Loaded {len(failures)} failures, {len(replay_buffer)} replay cases")
```

### LoRATrainer

```python
from src.training import LoRATrainer

trainer = LoRATrainer(
    model=draft_model,
    tokenizer=tokenizer,
    config=config['training']
)

# Load data
failures = collector.load_failures()
replay_buffer = collector.load_replay_buffer()

# Train
results = trainer.train(
    training_examples=failures,
    replay_buffer=replay_buffer
)

print(f"Training complete. Final loss: {results['final_loss']:.3f}")
print(f"Checkpoint saved to: {results['checkpoint_path']}")
```

## Custom Workflows

### Continuous Learning Loop

```python
import time

system = SpeculativeDecodingSystem(config)
system.initialize()

while True:
    # Generate with detailed tracking
    prompt = get_next_prompt()  # Your function
    response = system.generate(prompt, mode="detailed")

    # Check stats
    stats = system.get_stats()

    # Train when enough failures collected
    if stats['failure_count'] >= 50:
        print("Training on collected failures...")
        results = system.train()
        print(f"Training done. Loss: {results['final_loss']:.3f}")

    # Periodically check acceptance rate
    if stats['generation_count'] % 10 == 0:
        acc_rate = stats['tracker']['recent_average']
        print(f"Recent acceptance rate: {acc_rate:.1%}")

        if acc_rate < 0.5:
            print("Low acceptance rate, consider training or adjusting config")

    time.sleep(1)
```

### A/B Testing: Base vs Trained

```python
system = SpeculativeDecodingSystem(config)
system.initialize()

test_prompts = [...]  # Your test set

# Test with base model (no adapter)
system.unload_adapter()
base_results = []
for prompt in test_prompts:
    response = system.generate(prompt, mode="fast")
    stats = system.get_stats()
    base_results.append(stats['tracker']['recent_average'])

base_avg = sum(base_results) / len(base_results)
print(f"Base model acceptance: {base_avg:.1%}")

# Test with trained adapter
system.load_adapter("best")
trained_results = []
for prompt in test_prompts:
    response = system.generate(prompt, mode="fast")
    stats = system.get_stats()
    trained_results.append(stats['tracker']['recent_average'])

trained_avg = sum(trained_results) / len(trained_results)
print(f"Trained model acceptance: {trained_avg:.1%}")
print(f"Improvement: {trained_avg - base_avg:.1%}")
```

### Custom Data Collection

```python
from src.speculative_manual import ManualSpeculativeDecoder
from src.data_collector import DataCollector, TrainingExample
from datetime import datetime

# Initialize
system = SpeculativeDecodingSystem(config)
system.initialize()

# Get components
decoder = system.decoder  # ManualSpeculativeDecoder
collector = DataCollector("data/custom_failures/", acceptance_threshold=0.6)

# Custom collection logic
for prompt in my_prompts:
    result = decoder.generate(prompt, max_tokens=512)

    # Custom criteria for "failure"
    if result.acceptance_rate < 0.6 or len(result.disagreements) > 10:
        example = TrainingExample(
            id=f"custom_{datetime.now().isoformat()}",
            prompt=prompt,
            target_output=result.text,
            acceptance_rate=result.acceptance_rate,
            prompt_tokens=decoder.tokenizer.encode(prompt),
            disagreements=result.disagreements,
            timestamp=datetime.now().isoformat()
        )
        collector.add_failure(example)
        print(f"Collected failure: {result.acceptance_rate:.1%} acceptance")
```

## Type Hints

All main classes and functions are typed. Use a type checker:

```python
from src.main import SpeculativeDecodingSystem
from typing import Dict, List

def process_batch(system: SpeculativeDecodingSystem, prompts: List[str]) -> Dict:
    responses = system.batch_generate(prompts)
    stats = system.get_stats()
    return {
        'responses': responses,
        'acceptance_rate': stats['tracker']['recent_average']
    }
```

## Error Handling

```python
from src.main import SpeculativeDecodingSystem

try:
    system = SpeculativeDecodingSystem(config)
    system.initialize()
except ValueError as e:
    print(f"Invalid config: {e}")
except RuntimeError as e:
    print(f"Model loading failed: {e}")

try:
    response = system.generate("Test prompt")
except RuntimeError as e:
    print(f"Generation failed: {e}")
except MemoryError:
    print("Out of memory, reduce max_tokens or model size")

try:
    results = system.train()
except ValueError as e:
    print(f"Not enough training data: {e}")
except RuntimeError as e:
    print(f"Training failed: {e}")
```
