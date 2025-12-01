# Speculative Decoding with Adaptive LoRA Training

A high-performance inference system for Apple Silicon that uses speculative decoding with an adaptive LoRA training loop. The system improves draft model performance over time by learning from cases where it disagrees with the target model.

## Overview

### What is Speculative Decoding?

Speculative decoding is an inference optimization technique that uses two models:
- **Target Model**: Large, accurate model (Qwen2.5-7B, 4-bit quantized)
- **Draft Model**: Small, fast model (Qwen2.5-0.5B)

The algorithm:
1. Draft model generates K candidate tokens quickly
2. Target model verifies all K tokens in a single forward pass (parallel)
3. Accepted tokens are kept; rejected tokens are resampled from target
4. Speedup when draft agrees with target (high acceptance rate)

### What is Adaptive LoRA Training?

When the draft model frequently disagrees with the target model (low acceptance rate), we collect these "failure cases" and fine-tune the draft model using LoRA (Low-Rank Adaptation) to better match the target's behavior.

This creates a feedback loop:
1. **Inference** → Track acceptance rates
2. **Collect** → Store failure cases
3. **Train** → LoRA fine-tune draft model
4. **Improve** → Higher acceptance rate → Faster inference

## Features

- 🚀 **Speculative Decoding**: 1.5-2x speedup over standard generation
- 🎯 **Adaptive Training**: Draft model improves over time
- 💾 **Memory Efficient**: Works on 16GB RAM MacBooks
- 📊 **Metrics Tracking**: Acceptance rates, tokens/second, training progress
- 🔧 **Configurable**: All hyperparameters in `config.yaml`
- 🖥️ **CLI Interface**: Easy-to-use commands

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10+
- 16GB+ RAM (32GB recommended)
- ~15GB disk space for models

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd speculative-decoding-lora
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify MLX Installation

```bash
python -c "import mlx.core as mx; print(mx.default_device())"
# Should output: Device(gpu, 0)
```

## Quick Start

### Run Demo

```bash
python -m src.main demo
```

This will:
1. Download and load the models (first run only)
2. Run sample generations
3. Show acceptance rate metrics

### Interactive Mode

```bash
python -m src.main interactive
```

Commands in interactive mode:
- `/stats` - Show current statistics
- `/train` - Train on collected failure cases
- `/eval` - Run evaluation benchmark
- `/quit` - Exit

### Generate Text

```bash
python -m src.main generate "Explain quantum computing in simple terms."
```

### Train the Draft Model

```bash
python -m src.main train
```

### Benchmark Performance

```bash
python -m src.main benchmark --prompt "What is machine learning?" --iterations 10
```

## Configuration

All settings are in `configs/config.yaml`:

```yaml
# Model Configuration
models:
  target:
    name: "mlx-community/Qwen2.5-7B-Instruct-4bit"
  draft:
    name: "mlx-community/Qwen2.5-0.5B-Instruct"

# Speculative Decoding
speculative:
  num_draft_tokens: 4          # K tokens to draft
  temperature: 0.7             # Sampling temperature
  acceptance_threshold: 0.5    # Below = failure case

# LoRA Training
training:
  lora:
    rank: 8                    # LoRA rank (small for 16GB RAM)
    alpha: 16                  # LoRA alpha
  learning_rate: 1.0e-4
  min_failure_cases: 50        # When to trigger training
```

## Architecture

```
project/
├── src/
│   ├── models.py          # Model loading and management
│   ├── speculative.py     # Speculative decoding algorithm
│   ├── training.py        # LoRA training pipeline
│   ├── data_collector.py  # Failure case collection
│   └── main.py            # CLI and orchestration
├── data/
│   ├── failures/          # Collected failure cases (JSONL)
│   └── checkpoints/       # LoRA checkpoints
├── configs/
│   └── config.yaml        # All hyperparameters
└── requirements.txt
```

## How It Works

### 1. Speculative Decoding Flow

```
Draft Model        Target Model
    |                   |
    v                   |
Generate K tokens       |
    |                   |
    +-------> Verify all K tokens (parallel)
                        |
                        v
              Accept/Reject each token
                        |
                        v
              Return accepted tokens + resample if needed
```

### 2. Adaptive Training Loop

```
┌─────────────────────────────────────────────────┐
│                                                 │
│   Generate with Speculative Decoding            │
│              ↓                                  │
│   Track Acceptance Rate                         │
│              ↓                                  │
│   If acceptance < threshold:                    │
│       Store as failure case                     │
│              ↓                                  │
│   When N failures collected:                    │
│       Trigger LoRA Training                     │
│              ↓                                  │
│   Draft model improves → Higher acceptance      │
│              ↓                                  │
│   Loop back to generation                       │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 3. LoRA Fine-Tuning

We use Low-Rank Adaptation to efficiently fine-tune the draft model:
- Only ~0.5% of parameters are trainable
- Fits in 16GB RAM
- Fast training (minutes, not hours)
- Preserves original model capabilities via replay buffer

## API Usage

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
response = system.generate("What is the meaning of life?")
print(response)

# Get statistics
stats = system.get_stats()
print(f"Acceptance rate: {stats['tracker']['recent_average']:.1%}")

# Train when ready
system.train()
```

## Performance Tips

### Memory Optimization

1. **Use 4-bit quantized target model** (default)
2. **Keep batch size at 1** for 16GB RAM
3. **Clear cache periodically**: Happens automatically
4. **Reduce max_tokens** if running out of memory

### Speed Optimization

1. **Increase num_draft_tokens** (K=4-8) for higher throughput
2. **Use temperature=0** (greedy) for more consistent acceptance
3. **Train frequently** to improve acceptance rate

### Acceptance Rate Tips

- Higher acceptance = faster generation
- Target 60-80% acceptance rate
- Train more frequently if acceptance drops
- Different prompt types may have different rates

## Troubleshooting

### Out of Memory

```bash
# Reduce model sizes in config.yaml
models:
  target:
    name: "mlx-community/Qwen2.5-3B-Instruct-4bit"  # Smaller target
  draft:
    name: "mlx-community/Qwen2.5-0.5B-Instruct-4bit"  # Quantized draft
```

### Slow Model Loading

First run downloads models from Hugging Face. Subsequent runs use cache:
```bash
# Models cached at: ~/.cache/huggingface/hub/
```

### Low Acceptance Rate

1. Check if prompts match training data distribution
2. Increase temperature for more diverse sampling
3. Collect more failure cases and train
4. Consider using a larger draft model

## Metrics Explained

| Metric | Description | Good Value |
|--------|-------------|------------|
| Acceptance Rate | % of draft tokens accepted | >50% |
| Tokens/Second | Generation speed | Higher = better |
| Failure Cases | Prompts with low acceptance | Use for training |
| Training Loss | LoRA training loss | Lower = better |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python -m pytest tests/`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## References

- [Speculative Decoding Paper](https://arxiv.org/abs/2211.17192)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [MLX Framework](https://github.com/ml-explore/mlx)
- [Qwen2.5 Models](https://huggingface.co/Qwen)

## Acknowledgments

- Apple MLX team for the framework
- Qwen team for the models
- HuggingFace for model hosting
