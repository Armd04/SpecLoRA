#!/usr/bin/env python3
"""
Simple Demo of Speculative Decoding with Adaptive LoRA

This script demonstrates the core functionality of the system.
Run on a Mac with Apple Silicon to use MLX.

Usage:
    python examples/simple_demo.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_environment():
    """Check if we're running on macOS with MLX available."""
    import platform

    print("=" * 60)
    print("Environment Check")
    print("=" * 60)

    print(f"Platform: {platform.system()}")
    print(f"Machine: {platform.machine()}")
    print(f"Python: {platform.python_version()}")

    if platform.system() != "Darwin":
        print("\n⚠️  Warning: This system is optimized for macOS with Apple Silicon.")
        print("   MLX is only available on macOS.")
        return False

    try:
        import mlx.core as mx

        print(f"MLX Device: {mx.default_device()}")
        print("✅ MLX is available!")
        return True
    except ImportError:
        print("\n❌ MLX is not installed. Install with: pip install mlx mlx-lm")
        return False


def run_demo():
    """Run the speculative decoding demo."""
    from src.main import SpeculativeDecodingSystem
    from src.utils import load_config

    print("\n" + "=" * 60)
    print("Speculative Decoding Demo")
    print("=" * 60)

    # Load config
    config = load_config("configs/config.yaml")

    # Override for faster demo
    config["speculative"]["max_tokens"] = 64
    config["training"]["min_failure_cases"] = 10

    # Initialize system
    print("\n📦 Loading models...")
    system = SpeculativeDecodingSystem(config)
    system.initialize()

    # Demo prompts
    demo_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "Write a haiku about programming.",
    ]

    print("\n📝 Running demo generations...\n")

    for i, prompt in enumerate(demo_prompts, 1):
        print(f"\n--- Demo {i}/{len(demo_prompts)} ---")
        print(f"Prompt: {prompt}")

        response = system.generate(prompt, collect_data=True)
        print(f"Response: {response}")

    # Show stats
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)

    stats = system.get_stats()

    if "collector" in stats:
        print("\nData Collector:")
        print(f"  Failure cases: {stats['collector']['num_failure_cases']}")
        print(f"  Replay buffer: {stats['collector']['num_replay_cases']}")

    if "tracker" in stats:
        print("\nAcceptance Rate Tracker:")
        print(f"  Total observations: {stats['tracker']['total_observations']}")
        print(f"  Recent average: {stats['tracker']['recent_average']:.1%}")
        print(f"  Trend: {stats['tracker']['trend']:+.2%}")

    print("\n✅ Demo complete!")
    print("Run 'python -m src.main interactive' for interactive mode.")


def run_mock_demo():
    """Run a mock demo without MLX (for testing structure)."""
    from src.data_collector import DataCollector, AcceptanceRateTracker, TrainingExample

    print("\n" + "=" * 60)
    print("Mock Demo (Without MLX)")
    print("=" * 60)

    print("\n📝 Simulating speculative decoding...")

    # Create mock data collector
    collector = DataCollector(
        failures_dir="data/failures",
        max_failure_cases=50,
        replay_buffer_size=25,
    )

    # Create mock tracker
    tracker = AcceptanceRateTracker(window_size=100)

    # Simulate some generations
    import random

    for i in range(20):
        # Simulate acceptance rate
        acceptance_rate = random.uniform(0.3, 0.9)
        is_failure = acceptance_rate < 0.5

        # Track rate
        tracker.add_rate(acceptance_rate)

        # Create mock example
        example = TrainingExample(
            id=f"mock_{i}",
            prompt=f"Demo prompt {i}",
            draft_output=[1, 2, 3, 4],
            target_output=[1, 2, 5, 6] if is_failure else [1, 2, 3, 4],
            acceptance_rate=acceptance_rate,
            timestamp="2024-01-01",
            is_failure=is_failure,
        )

        if is_failure:
            collector.failure_cases.append(example)
        else:
            collector.replay_buffer.append(example)

        print(
            f"  Generation {i + 1}: acceptance={acceptance_rate:.1%}, failure={is_failure}"
        )

    # Show stats
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)

    collector_stats = collector.get_stats()
    tracker_stats = tracker.get_stats()

    print("\nData Collector:")
    print(f"  Failure cases: {collector_stats['num_failure_cases']}")
    print(f"  Replay buffer: {collector_stats['num_replay_cases']}")

    print("\nAcceptance Rate Tracker:")
    print(f"  Total observations: {tracker_stats['total_observations']}")
    print(f"  Overall average: {tracker_stats['overall_average']:.1%}")
    print(f"  Min rate: {tracker_stats['min_rate']:.1%}")
    print(f"  Max rate: {tracker_stats['max_rate']:.1%}")

    print("\n✅ Mock demo complete!")
    print("Run on macOS with MLX for full functionality.")


def main():
    """Main entry point."""
    print("🚀 Speculative Decoding with Adaptive LoRA")
    print("=" * 60)

    if check_environment():
        run_demo()
    else:
        print("\nRunning mock demo without MLX...")
        run_mock_demo()


if __name__ == "__main__":
    main()
