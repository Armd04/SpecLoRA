"""
Main CLI and Orchestration for Speculative Decoding with Adaptive LoRA

This is the main entry point for the system. It provides:
1. CLI commands for inference, training, and evaluation
2. Orchestration of the full adaptive training pipeline
3. Interactive mode for testing
4. Data collection mode for token-level disagreement capture

Usage:
    python -m src.main generate "prompt" --mode fast    # Production mode
    python -m src.main generate "prompt" --mode detailed  # Data collection mode
    python -m src.main collect-data --prompts-file FILE   # Batch data collection
"""

import logging
import random
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import click
import yaml
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.table import Table
from rich.panel import Panel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        sys.exit(1)

    with open(config_path) as f:
        return yaml.safe_load(f)


class SpeculativeDecodingSystem:
    """
    Main orchestration class for the speculative decoding system.

    Coordinates between:
    - Model loading and management
    - Speculative decoding inference
    - Failure case collection
    - LoRA training
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the system with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_manager = None
        self.decoder = None
        self.data_collector = None
        self.rate_tracker = None
        self._initialized = False
        self._generation_count = 0  # Track generations for cache clearing
        self._current_adapter_path: Optional[str] = None  # Track loaded adapter
        self._current_adapter_info: Optional[Dict[str, Any]] = None  # Adapter metadata

    def initialize(self, initial_adapter_path: Optional[str] = None) -> None:
        """
        Initialize all components.

        Args:
            initial_adapter_path: Optional LoRA adapter to load during initialization
        """
        from .models import ModelManager
        from .speculative import SpeculativeDecoder
        from .data_collector import DataCollector, AcceptanceRateTracker

        console.print("[cyan]Initializing speculative decoding system...[/cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Load models
            task = progress.add_task("Loading models...", total=None)

            self.model_manager = ModelManager(
                target_model_name=self.config["models"]["target"]["name"],
                draft_model_name=self.config["models"]["draft"]["name"],
            )
            self.model_manager.load_models()

            # Load initial adapter if specified
            if initial_adapter_path:
                try:
                    adapter_path = self.resolve_adapter_path(initial_adapter_path)
                    metadata = self.validate_adapter_structure(adapter_path)

                    progress.update(
                        task, description=f"Loading adapter: {adapter_path.name}..."
                    )
                    self.model_manager.load_lora_adapter(str(adapter_path), fuse=True)

                    self._current_adapter_path = str(adapter_path)
                    self._current_adapter_info = metadata

                    progress.update(task, description="Adapter loaded!")
                except Exception as e:
                    logger.error(f"Failed to load initial adapter: {e}")
                    console.print(f"[red]Warning: Could not load adapter: {e}[/red]")
                    console.print("[yellow]Continuing with base model...[/yellow]")

            progress.update(task, description="Models loaded!")

            # Get models and tokenizer
            target_model, target_tokenizer = self.model_manager.get_target_model()
            draft_model, draft_tokenizer = self.model_manager.get_draft_model()

            # Initialize decoder
            chat_config = self.config.get("chat", {})
            self.decoder = SpeculativeDecoder(
                target_model=target_model,
                draft_model=draft_model,
                tokenizer=target_tokenizer,  # Use target tokenizer (should be same)
                num_draft_tokens=self.config["speculative"]["num_draft_tokens"],
                temperature=self.config["speculative"]["temperature"],
                top_p=self.config["speculative"]["top_p"],
                acceptance_threshold=self.config["speculative"]["acceptance_threshold"],
                system_message=chat_config.get("system_message"),
            )

            # Initialize data collector
            self.data_collector = DataCollector(
                failures_dir=self.config["data"]["failures_dir"],
                max_failure_cases=self.config["training"]["max_failure_cases"],
                replay_buffer_size=self.config["training"]["replay_buffer_size"],
                replay_ratio=self.config["training"]["replay_ratio"],
            )

            # Initialize rate tracker
            self.rate_tracker = AcceptanceRateTracker()

            # Try to load existing tracker data
            tracker_path = (
                Path(self.config["data"]["failures_dir"]) / "rate_tracker.json"
            )
            if tracker_path.exists():
                self.rate_tracker.load(str(tracker_path))

            self._initialized = True

        # Print memory usage
        memory = self.model_manager.estimate_memory_usage()
        console.print(
            Panel(
                f"Target Model: {memory['target_params_m']:.1f}M params ({memory['target_model_gb']:.2f} GB)\n"
                f"Draft Model: {memory['draft_params_m']:.1f}M params ({memory['draft_model_gb']:.2f} GB)\n"
                f"Total: {memory['total_gb']:.2f} GB",
                title="Memory Usage",
            )
        )

        # Display adapter status if loaded
        if self._current_adapter_path:
            console.print(
                Panel(
                    f"LoRA Adapter: [green]{Path(self._current_adapter_path).name}[/green]",
                    title="Adapter Loaded",
                )
            )

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        collect_data: bool = True,
        implementation: str = "manual",
    ) -> str:
        """
        Generate text using speculative decoding.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            collect_data: Whether to collect training data
            implementation: "manual" for our implementation with KV-cached target verification
                           and token-level data collection (default, recommended),
                           "builtin" for MLX-LM's built-in speculative decoding

        Returns:
            Generated text
        """
        if not self._initialized:
            self.initialize()

        max_tokens = max_tokens or self.config["speculative"]["max_tokens"]

        # Clear cache periodically to prevent memory accumulation
        cache_clear_freq = self.config.get("memory", {}).get(
            "cache_clear_frequency", 10
        )
        if (
            self._generation_count > 0
            and self._generation_count % cache_clear_freq == 0
        ):
            self.model_manager.clear_cache()
            logger.debug(f"Cleared cache after {self._generation_count} generations")

        if implementation == "manual":
            # Use manual speculative decoding for detailed data collection
            result = self.decoder.generate_detailed(
                prompt=prompt,
                max_tokens=max_tokens,
            )

            # Increment generation counter
            self._generation_count += 1

            # Track acceptance rate
            self.rate_tracker.add_rate(result.metrics.acceptance_rate)

            # Collect detailed data if enabled
            if collect_data:
                # Calculate prompt length for accurate disagreement position mapping
                # Need to format and tokenize the prompt the same way the decoder does
                # IMPORTANT: Use the decoder's actual system_message to ensure consistency
                try:
                    formatted_prompt = self.decoder.format_prompt(prompt)
                    prompt_tokens = self.decoder.tokenizer.encode(formatted_prompt)
                    prompt_length = len(prompt_tokens)
                except Exception as e:
                    logger.warning(f"Could not calculate prompt length: {e}")
                    prompt_length = None
                    prompt_tokens = None

                should_train = self.data_collector.add_detailed_result(
                    prompt=prompt,
                    generated_tokens=result.tokens,
                    disagreements=result.disagreements,
                    acceptance_rate=result.metrics.acceptance_rate,
                    metadata={
                        "total_tokens": result.metrics.total_tokens_generated,
                        "tokens_per_second": result.metrics.tokens_per_second,
                        "draft_time": result.metrics.draft_time_seconds,
                        "verify_time": result.metrics.verify_time_seconds,
                    },
                    prompt_length=prompt_length,
                    prompt_tokens=prompt_tokens,
                )

                if should_train:
                    console.print(
                        "[yellow]Training threshold reached! "
                        "Run 'train' command to fine-tune the draft model.[/yellow]"
                    )

            # Display detailed metrics
            metrics_table = Table(
                title="Generation Metrics (Manual Spec-Dec)", show_header=False
            )
            metrics_table.add_row(
                "Acceptance Rate", f"{result.metrics.acceptance_rate:.1%}"
            )
            metrics_table.add_row(
                "Tokens/Second", f"{result.metrics.tokens_per_second:.1f}"
            )
            metrics_table.add_row(
                "Total Tokens", str(result.metrics.total_tokens_generated)
            )
            metrics_table.add_row(
                "Draft Proposed", str(result.metrics.draft_tokens_proposed)
            )
            metrics_table.add_row(
                "Draft Accepted", str(result.metrics.draft_tokens_accepted)
            )
            metrics_table.add_row(
                "[bold]Disagreements[/bold]", str(len(result.disagreements))
            )
            metrics_table.add_row(
                "Is Failure Case", "Yes" if result.is_failure_case else "No"
            )

            # Show disagreement details if any
            if result.disagreements:
                high_conf = sum(
                    1 for d in result.disagreements if d.is_high_confidence_failure
                )
                metrics_table.add_row("High-Confidence Failures", str(high_conf))

            console.print(metrics_table)

            return result.text

        else:
            # Use MLX-LM's built-in speculative decoding
            result = self.decoder.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                collect_training_data=collect_data,
            )

            # Increment generation counter
            self._generation_count += 1

            # Track acceptance rate
            self.rate_tracker.add_rate(result.metrics.acceptance_rate)

            # Collect data if enabled
            if collect_data:
                should_train = self.data_collector.add_result(result)

                if should_train:
                    console.print(
                        "[yellow]Training threshold reached! "
                        "Run 'train' command to fine-tune the draft model.[/yellow]"
                    )

            # Display metrics
            metrics_table = Table(
                title="Generation Metrics (MLX-LM Built-in)", show_header=False
            )
            metrics_table.add_row(
                "Acceptance Rate", f"{result.metrics.acceptance_rate:.1%}"
            )
            metrics_table.add_row(
                "Tokens/Second", f"{result.metrics.tokens_per_second:.1f}"
            )
            metrics_table.add_row(
                "Total Tokens", str(result.metrics.total_tokens_generated)
            )
            metrics_table.add_row(
                "Draft Proposed", str(result.metrics.draft_tokens_proposed)
            )
            metrics_table.add_row(
                "Draft Accepted", str(result.metrics.draft_tokens_accepted)
            )
            metrics_table.add_row(
                "Is Failure Case", "Yes" if result.is_failure_case else "No"
            )

            console.print(metrics_table)

            return result.text

    def collect_data_batch(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run data collection on a batch of prompts using manual speculative decoding.

        This method processes multiple prompts and collects detailed token-level
        disagreement data for training.

        Args:
            prompts: List of prompts to process
            max_tokens: Maximum tokens per generation
            output_file: Optional path to save detailed results

        Returns:
            Dictionary with aggregate statistics
        """
        if not self._initialized:
            self.initialize()

        max_tokens = max_tokens or self.config["speculative"]["max_tokens"]

        # Create manual decoder
        manual_decoder = self.decoder.create_manual_decoder()

        console.print(f"[cyan]Collecting data from {len(prompts)} prompts...[/cyan]")

        # Run batch collection with progress
        results = []
        stats = {
            "total_tokens": 0,
            "total_disagreements": 0,
            "total_accepted": 0,
            "total_proposed": 0,
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing prompts...", total=len(prompts))

            for prompt in prompts:
                result = manual_decoder.generate_with_data_collection(
                    prompt, max_tokens
                )
                results.append(result)

                # Calculate prompt length for accurate disagreement position mapping
                prompt_length = None
                prompt_tokens = None
                try:
                    formatted_prompt = manual_decoder.format_prompt(prompt)
                    prompt_tokens = manual_decoder.tokenizer.encode(formatted_prompt)
                    prompt_length = len(prompt_tokens)
                except Exception as e:
                    logger.warning(f"Could not calculate prompt length: {e}")

                # Add to collector
                self.data_collector.add_detailed_result(
                    prompt=prompt,
                    generated_tokens=result.tokens,
                    disagreements=result.disagreements,
                    acceptance_rate=result.metrics.acceptance_rate,
                    prompt_length=prompt_length,
                    prompt_tokens=prompt_tokens,
                )

                # Track aggregate stats
                stats["total_tokens"] += result.metrics.total_tokens_generated
                stats["total_disagreements"] += len(result.disagreements)
                stats["total_accepted"] += result.metrics.draft_tokens_accepted
                stats["total_proposed"] += result.metrics.draft_tokens_proposed

                # Track rate
                self.rate_tracker.add_rate(result.metrics.acceptance_rate)

                progress.advance(task)

        # Calculate final stats
        stats["num_prompts"] = len(prompts)
        stats["acceptance_rate"] = stats["total_accepted"] / max(
            stats["total_proposed"], 1
        )
        stats["avg_disagreements"] = stats["total_disagreements"] / max(len(prompts), 1)
        stats["failure_cases"] = sum(1 for r in results if r.is_failure_case)
        stats["high_confidence_failures"] = sum(
            1 for r in results for d in r.disagreements if d.is_high_confidence_failure
        )

        # Save detailed results if output file specified
        if output_file:
            import json

            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            detailed_results = []
            for r in results:
                detailed_results.append(
                    {
                        "prompt": r.prompt,
                        "text": r.text,
                        "tokens": r.tokens,
                        "acceptance_rate": r.metrics.acceptance_rate,
                        "disagreements": [d.to_dict() for d in r.disagreements],
                        "is_failure": r.is_failure_case,
                    }
                )

            with open(output_path, "w") as f:
                for item in detailed_results:
                    f.write(json.dumps(item) + "\n")

            console.print(f"[green]Saved detailed results to {output_file}[/green]")

        # Display summary
        table = Table(title="Data Collection Summary")
        table.add_column("Metric")
        table.add_column("Value")

        table.add_row("Prompts Processed", str(stats["num_prompts"]))
        table.add_row("Total Tokens", str(stats["total_tokens"]))
        table.add_row("Acceptance Rate", f"{stats['acceptance_rate']:.1%}")
        table.add_row(
            "[bold]Total Disagreements[/bold]", str(stats["total_disagreements"])
        )
        table.add_row("Avg Disagreements/Prompt", f"{stats['avg_disagreements']:.1f}")
        table.add_row("Failure Cases", str(stats["failure_cases"]))
        table.add_row(
            "High-Confidence Failures", str(stats["high_confidence_failures"])
        )

        console.print(table)

        # Check if training should be triggered
        collector_stats = self.data_collector.get_stats()
        if collector_stats["ready_for_training"]:
            console.print(
                "[yellow]Training threshold reached! "
                "Run 'train' command to fine-tune the draft model.[/yellow]"
            )

        return stats

    def train(self, num_epochs: Optional[int] = None) -> None:
        """
        Train the draft model on collected failure cases.

        Args:
            num_epochs: Number of training epochs
        """
        if not self._initialized:
            self.initialize()

        from .training import LoRATrainer, LoRAConfig

        num_epochs = num_epochs or self.config["training"]["num_epochs"]

        # Check if we have enough data
        stats = self.data_collector.get_stats()
        if stats["num_failure_cases"] < self.config["training"]["min_failure_cases"]:
            console.print(
                f"[yellow]Not enough failure cases for training. "
                f"Have {stats['num_failure_cases']}, need {self.config['training']['min_failure_cases']}[/yellow]"
            )
            return

        console.print(
            Panel(
                f"Failure cases: {stats['num_failure_cases']}\n"
                f"Replay buffer: {stats['num_replay_cases']}\n"
                f"Avg failure acceptance: {stats.get('avg_failure_acceptance', 0):.1%}",
                title="Training Data Summary",
            )
        )

        # Get training data
        training_examples = self.data_collector.get_all_training_data()

        # Setup LoRA config
        lora_config = LoRAConfig(
            rank=self.config["training"]["lora"]["rank"],
            alpha=self.config["training"]["lora"]["alpha"],
            dropout=self.config["training"]["lora"]["dropout"],
            target_modules=self.config["training"]["lora"]["target_modules"],
        )

        # Get draft model
        draft_model, draft_tokenizer = self.model_manager.get_draft_model()
        # IMPORTANT: Training data collection uses the decoder tokenizer (target tokenizer).
        # If draft/target tokenizers differ, the token IDs in failure cases may be invalid
        # for the draft model (leading to NaNs or undefined behavior). Prefer the decoder's
        # tokenizer for training, and hard-fail on obvious incompatibility.
        tokenizer = self.decoder.tokenizer
        try:
            # Prefer len(tokenizer) over vocab_size because many tokenizers exclude
            # added/special tokens from vocab_size but still emit IDs >= vocab_size.
            def _effective_size(tok):
                try:
                    return int(len(tok))
                except Exception:
                    vs = getattr(tok, "vocab_size", None)
                    return int(vs) if vs is not None else None

            draft_sz = _effective_size(draft_tokenizer)
            target_sz = _effective_size(tokenizer)
            if (
                isinstance(draft_sz, int)
                and isinstance(target_sz, int)
                and draft_sz != target_sz
            ):
                raise RuntimeError(
                    "Draft/target tokenizers have different effective vocab sizes "
                    f"({draft_sz} vs {target_sz}). "
                    "Collected token IDs may not be valid for the draft model; aborting training."
                )
        except Exception as e:
            # If tokenizer objects don't expose vocab_size cleanly, we still proceed,
            # but training will additionally validate token ID ranges per example.
            logger.warning(f"Tokenizer compatibility check failed: {e}")

        # Initialize trainer
        trainer = LoRATrainer(
            model=draft_model,
            tokenizer=tokenizer,
            lora_config=lora_config,
            learning_rate=self.config["training"]["learning_rate"],
            batch_size=self.config["training"]["batch_size"],
            gradient_accumulation_steps=self.config["training"][
                "gradient_accumulation_steps"
            ],
            warmup_steps=self.config["training"]["warmup_steps"],
            checkpoint_dir=self.config["training"]["checkpoint_dir"],
        )

        console.print("[cyan]Starting LoRA training...[/cyan]")

        # Train
        metrics = trainer.train(
            training_examples=training_examples,
            num_epochs=num_epochs,
            save_every_n_steps=self.config["training"]["save_every_n_steps"],
        )

        # Fuse LoRA weights and get clean model for efficient inference
        # This removes the LoRA wrapper overhead that would otherwise slow down inference
        clean_model = trainer.fuse_and_get_model()

        # Update decoder with fused model for inference
        self.decoder.draft_model = clean_model

        # Clear failure cases
        self.data_collector.clear_failure_cases()

        # Display results
        results_table = Table(title="Training Results", show_header=False)
        results_table.add_row("Total Steps", str(metrics.total_steps))
        results_table.add_row("Average Loss", f"{metrics.avg_loss:.4f}")
        results_table.add_row("Training Time", f"{metrics.training_time_seconds:.1f}s")
        results_table.add_row("Adapter Saved To", metrics.adapter_path or "N/A")

        console.print(results_table)
        console.print("[green]Training complete! Draft model updated.[/green]")

        # Prompt user to save to 'best' folder
        existing_best_loss = trainer.get_existing_best_loss()
        if existing_best_loss is not None:
            comparison = (
                f"[yellow]Existing best loss: {existing_best_loss:.4f}[/yellow]\n"
                f"[cyan]Current loss: {metrics.avg_loss:.4f}[/cyan]"
            )
            if metrics.avg_loss < existing_best_loss:
                comparison += " [green](better)[/green]"
            else:
                comparison += " [red](worse)[/red]"
            console.print(Panel(comparison, title="Comparison with Existing Best"))
            default_choice = metrics.avg_loss < existing_best_loss
        else:
            console.print("[yellow]No existing best checkpoint found.[/yellow]")
            default_choice = True

        if click.confirm("\nSave this adapter to 'best'?", default=default_choice):
            best_path = trainer.save_to_best()
            console.print(f"[green]✓ Saved to: {best_path}[/green]")

    def evaluate(self, prompts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate current acceptance rates.

        Args:
            prompts: Optional list of evaluation prompts

        Returns:
            Evaluation results
        """
        if not self._initialized:
            self.initialize()

        from .speculative import run_acceptance_benchmark

        prompts = prompts or self.config["evaluation"]["test_prompts"]

        console.print("[cyan]Running evaluation benchmark...[/cyan]")

        results = run_acceptance_benchmark(
            decoder=self.decoder,
            prompts=prompts,
            max_tokens=128,
        )

        # Display results
        table = Table(title="Evaluation Results")
        table.add_column("Prompt")
        table.add_column("Acceptance Rate")
        table.add_column("Tokens/s")
        table.add_column("Failure?")

        for prompt_result in results["per_prompt"]:
            table.add_row(
                prompt_result["prompt"],
                f"{prompt_result['acceptance_rate']:.1%}",
                f"{prompt_result['tokens_per_second']:.1f}",
                "Yes" if prompt_result["is_failure"] else "No",
            )

        console.print(table)

        # Summary
        console.print(
            Panel(
                f"Overall Acceptance Rate: {results['overall_acceptance_rate']:.1%}\n"
                f"Overall Tokens/Second: {results['overall_tokens_per_second']:.1f}",
                title="Summary",
            )
        )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        if not self._initialized:
            console.print(
                "[yellow]System not initialized. Run a command first.[/yellow]"
            )
            return {}

        collector_stats = self.data_collector.get_stats()
        tracker_stats = self.rate_tracker.get_stats()

        return {
            "collector": collector_stats,
            "tracker": tracker_stats,
        }

    def resolve_adapter_path(self, path_or_alias: str) -> Path:
        """
        Resolve adapter path from user input (supports aliases and paths).

        Supports:
        - Aliases: 'best', 'latest'
        - Relative paths: data/checkpoints/my_adapter
        - Absolute paths: /full/path/to/adapter

        Args:
            path_or_alias: User input (path or alias)

        Returns:
            Resolved absolute Path object

        Raises:
            FileNotFoundError: If resolved path doesn't exist
        """
        checkpoint_dir = Path(self.config["training"]["checkpoint_dir"])

        # Handle aliases
        if path_or_alias.lower() == "best":
            resolved = checkpoint_dir / "best"
        elif path_or_alias.lower() == "latest":
            # Find most recently modified checkpoint directory
            if not checkpoint_dir.exists():
                raise FileNotFoundError(
                    f"Checkpoint directory not found: {checkpoint_dir}"
                )
            checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
            resolved = max(checkpoints, key=lambda p: p.stat().st_mtime)
        else:
            # Treat as path (relative or absolute)
            resolved = Path(path_or_alias)
            if not resolved.is_absolute():
                resolved = Path.cwd() / resolved

        if not resolved.exists():
            raise FileNotFoundError(f"Adapter not found: {resolved}")

        return resolved

    def validate_adapter_structure(self, adapter_path: Path) -> Dict[str, Any]:
        """
        Validate adapter directory structure and return metadata.

        Required files:
        - adapters.safetensors (LoRA weights)
        - adapter_config.json (rank, alpha, dropout)
        - trainer_state.json (training metadata)

        Args:
            adapter_path: Path to adapter directory

        Returns:
            Merged metadata from config and trainer state

        Raises:
            ValueError: If structure is invalid or files are corrupted
        """
        import json

        required_files = {
            "adapters.safetensors": "LoRA weights file",
            "adapter_config.json": "Adapter configuration",
            "trainer_state.json": "Training state metadata",
        }

        # Check all required files exist
        missing = []
        for filename, description in required_files.items():
            if not (adapter_path / filename).exists():
                missing.append(f"{filename} ({description})")

        if missing:
            raise ValueError(
                "Invalid adapter structure. Missing files:\n"
                + "\n".join(f"  - {m}" for m in missing)
            )

        # Load and validate JSON files
        try:
            with open(adapter_path / "adapter_config.json") as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid adapter_config.json: {e}")

        try:
            with open(adapter_path / "trainer_state.json") as f:
                state = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid trainer_state.json: {e}")

        # Extract metadata
        lora_params = config.get("lora_parameters", {})
        metadata = {
            "rank": lora_params.get("rank"),
            "alpha": lora_params.get("scale"),  # MLX-LM uses 'scale' not 'alpha'
            "dropout": lora_params.get("dropout"),
            "global_step": state.get("global_step"),
            "best_loss": state.get("best_loss"),
            "path": str(adapter_path),
        }

        # Validate required fields
        if metadata["rank"] is None:
            raise ValueError("adapter_config.json missing lora_parameters.rank")

        return metadata

    def _reinitialize_decoder(self) -> None:
        """
        Recreate the decoder with current models.

        Called after adapter loading/unloading to ensure decoder uses
        the updated draft model weights.

        IMPORTANT: This preserves decoder settings (temperature, top_p, etc.)
        but clears all internal state (KV caches).
        """
        from .speculative import SpeculativeDecoder

        # Get current models from model manager
        target_model, target_tokenizer = self.model_manager.get_target_model()
        draft_model, draft_tokenizer = self.model_manager.get_draft_model()

        # Create new decoder with same config
        chat_config = self.config.get("chat", {})
        self.decoder = SpeculativeDecoder(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=target_tokenizer,  # Use target tokenizer (should be same)
            num_draft_tokens=self.config["speculative"]["num_draft_tokens"],
            temperature=self.config["speculative"]["temperature"],
            top_p=self.config["speculative"]["top_p"],
            acceptance_threshold=self.config["speculative"]["acceptance_threshold"],
            system_message=chat_config.get("system_message"),
        )

        logger.info("Decoder reinitialized with updated draft model")

    def load_lora_adapter(self, adapter_path: str) -> Dict[str, Any]:
        """
        Load a LoRA adapter and reinitialize the decoder.

        This method:
        1. Validates adapter path and structure
        2. Loads adapter into draft model via ModelManager
        3. Clears MLX memory cache
        4. Reinitializes decoder with updated draft model
        5. Returns adapter metadata for display

        Args:
            adapter_path: Path to adapter directory

        Returns:
            Dictionary with adapter info (rank, alpha, dropout, training_state)

        Raises:
            FileNotFoundError: If adapter path or required files don't exist
            ValueError: If adapter structure is invalid
            RuntimeError: If loading fails
        """
        # Validate first (fast fail)
        try:
            resolved_path = self.resolve_adapter_path(adapter_path)
            metadata = self.validate_adapter_structure(resolved_path)
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Invalid adapter: {e}") from None

        # Save current state for rollback
        old_adapter_path = self._current_adapter_path

        try:
            # Load adapter (may fail with MLX errors)
            self.model_manager.load_lora_adapter(str(resolved_path), fuse=True)

            # Clear cache
            self.model_manager.clear_cache()

            # Reinitialize decoder
            self._reinitialize_decoder()

            # Update state
            self._current_adapter_path = str(resolved_path)
            self._current_adapter_info = metadata

            logger.info(f"Loaded adapter: {resolved_path}")
            return metadata

        except Exception as e:
            # System error - attempt rollback
            logger.error(f"Failed to load adapter: {e}", exc_info=True)

            # Try to restore previous state
            if old_adapter_path:
                try:
                    self.model_manager.load_lora_adapter(old_adapter_path, fuse=True)
                    self._reinitialize_decoder()
                    console.print("[yellow]Rolled back to previous adapter[/yellow]")
                except Exception as rollback_error:
                    # Rollback failed - system in bad state
                    logger.critical(
                        f"Rollback failed after load error: {rollback_error}",
                        exc_info=True,
                    )
                    console.print(
                        "[red]Critical: System in inconsistent state. "
                        "Interactive mode may be unstable. "
                        "Please restart if you encounter issues.[/red]"
                    )
                    # Mark system as potentially unstable
                    self._initialized = False

            raise RuntimeError(f"Failed to load adapter: {e}") from e

    def unload_lora_adapter(self) -> None:
        """
        Unload current adapter and revert to base model.

        This requires reloading the draft model from scratch since
        LoRA fusion is irreversible (weights are modified in-place).
        """
        if self._current_adapter_path is None:
            logger.warning("Attempted to unload adapter, but none loaded")
            return

        console.print("[cyan]Reloading base model (this may take a moment)...[/cyan]")

        try:
            # Reload draft model without adapter
            from mlx_lm import load
            import mlx.core as mx

            draft_model, draft_tokenizer = load(
                self.config["models"]["draft"]["name"],
                lazy=True,
            )
            mx.eval(draft_model.parameters())

            # Update model manager
            self.model_manager.draft_model = draft_model
            self.model_manager.draft_tokenizer = draft_tokenizer

            # Clear cache
            self.model_manager.clear_cache()

            # Reinitialize decoder
            self._reinitialize_decoder()

            # Update state
            self._current_adapter_path = None
            self._current_adapter_info = None

            logger.info("Unloaded adapter, reverted to base model")

        except Exception as e:
            logger.error(f"Failed to unload adapter: {e}", exc_info=True)
            raise RuntimeError(f"Failed to unload adapter: {e}") from e

    def get_current_adapter_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the currently loaded adapter.

        Returns:
            Adapter metadata dict if loaded, None otherwise
        """
        return self._current_adapter_info

    def _format_adapter_status(self) -> str:
        """Format current adapter status for display."""
        if self._current_adapter_path is None:
            return "Adapter: [yellow]None (base model)[/yellow]"
        else:
            adapter_name = Path(self._current_adapter_path).name
            return f"Adapter: [green]{adapter_name}[/green]"

    def _display_adapter_info(self, metadata: Dict[str, Any]) -> None:
        """Display adapter metadata in a formatted table."""
        from rich.table import Table

        table = Table(title="LoRA Adapter", show_header=False, border_style="cyan")
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        # Path (show relative to CWD if possible)
        path_str = metadata.get("path", "Unknown")
        try:
            path_rel = Path(path_str).relative_to(Path.cwd())
            path_display = str(path_rel)
        except ValueError:
            path_display = path_str

        table.add_row("Path", path_display)
        table.add_row("Name", Path(path_str).name)

        # LoRA hyperparameters
        table.add_section()
        table.add_row("Rank", str(metadata.get("rank", "Unknown")))
        table.add_row("Alpha", str(metadata.get("alpha", "Unknown")))
        table.add_row("Dropout", str(metadata.get("dropout", "Unknown")))

        # Training metadata (if available)
        if metadata.get("global_step") is not None:
            table.add_section()
            table.add_row("Training Steps", str(metadata["global_step"]))
        if metadata.get("best_loss") is not None:
            table.add_row("Best Loss", f"{metadata['best_loss']:.4f}")

        console.print(table)
        console.print(
            "\n[yellow]Note: Decoder reinitialized with new adapter. "
            "KV caches cleared.[/yellow]"
        )

    def _handle_command(self, cmd: str) -> bool:
        """
        Handle slash commands in interactive mode.

        Args:
            cmd: Command string (without the leading '/')

        Returns:
            True if should exit interactive mode, False otherwise
        """
        parts = cmd.strip().split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command in ("quit", "exit"):
            return True

        elif command == "stats":
            stats = self.get_stats()
            console.print(stats)

        elif command == "train":
            self.train()

        elif command == "eval":
            self.evaluate()

        elif command == "clear":
            console.clear()

        elif command in ("load-adapter", "load"):
            if not args:
                console.print(
                    "[red]Error: /load-adapter requires a path argument[/red]"
                )
                console.print("Usage: /load-adapter <path|best|latest>")
                return False

            try:
                # Resolve and validate path
                adapter_path = self.resolve_adapter_path(args)
                metadata = self.validate_adapter_structure(adapter_path)

                # Load adapter
                console.print(f"[cyan]Loading LoRA adapter from: {adapter_path}[/cyan]")
                self.load_lora_adapter(str(adapter_path))

                # Display success + metadata
                console.print("[green]✓ Adapter loaded successfully[/green]\n")
                self._display_adapter_info(metadata)

            except (FileNotFoundError, ValueError) as e:
                console.print(f"[red]Error loading adapter: {e}[/red]")
            except Exception as e:
                console.print(f"[red]Unexpected error: {e}[/red]")
                logger.exception("Error loading adapter")

        elif command in ("unload-adapter", "unload"):
            if self._current_adapter_path is None:
                console.print("[yellow]No adapter currently loaded[/yellow]")
                return False

            try:
                console.print(
                    "[cyan]Unloading adapter and reverting to base model...[/cyan]"
                )
                self.unload_lora_adapter()
                console.print("[green]✓ Adapter unloaded successfully[/green]")
            except Exception as e:
                console.print(f"[red]Error unloading adapter: {e}[/red]")
                logger.exception("Error unloading adapter")

        elif command in ("adapter-info", "adapter"):
            info = self.get_current_adapter_info()
            if info is None:
                console.print("[yellow]No adapter currently loaded[/yellow]")
            else:
                self._display_adapter_info(info)

        else:
            console.print(f"[red]Unknown command: {command}[/red]")
            console.print("Type a prompt to generate, or use /quit to exit")

        return False

    def interactive_mode(self, implementation: str = "manual") -> None:
        """Run interactive generation mode.

        Args:
            implementation: "manual" (default) or "builtin" for spec dec implementation
        """
        if not self._initialized:
            self.initialize()

        impl_desc = (
            "Manual Spec-Dec" if implementation == "manual" else "MLX-LM Built-in"
        )
        adapter_status = self._format_adapter_status()

        console.print(
            Panel(
                f"Implementation: [cyan]{impl_desc}[/cyan]\n"
                f"{adapter_status}\n\n"
                "Enter prompts to generate text. Commands:\n"
                "  /quit - Exit interactive mode\n"
                "  /stats - Show statistics\n"
                "  /train - Train on collected data\n"
                "  /eval - Run evaluation\n"
                "  /clear - Clear screen\n"
                "  /load-adapter <path|best|latest> - Load LoRA adapter\n"
                "  /unload-adapter - Unload adapter (revert to base model)\n"
                "  /adapter-info - Show current adapter details",
                title="Interactive Mode",
            )
        )

        while True:
            try:
                prompt = console.input("\n[bold green]Prompt:[/bold green] ")

                if not prompt:
                    continue

                if prompt.startswith("/"):
                    # Handle command and check if should exit
                    should_exit = self._handle_command(prompt[1:])
                    if should_exit:
                        break
                    continue

                # Generate response
                console.print("\n[bold blue]Response:[/bold blue]")
                response = self.generate(prompt, implementation=implementation)
                console.print(response)

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type /quit to exit.[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                logger.exception("Error during generation")

        # Save tracker data on exit
        tracker_path = Path(self.config["data"]["failures_dir"]) / "rate_tracker.json"
        self.rate_tracker.save(str(tracker_path))
        console.print("[green]Session saved. Goodbye![/green]")


# CLI Commands
@click.group()
@click.option("--config", "-c", default="configs/config.yaml", help="Config file path")
@click.pass_context
def cli(ctx, config):
    """Speculative Decoding with Adaptive LoRA Training"""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)


@cli.command()
@click.argument("prompt")
@click.option("--max-tokens", "-m", default=256, help="Maximum tokens to generate")
@click.option("--no-collect", is_flag=True, help="Disable data collection")
@click.option(
    "--implementation",
    "-i",
    type=click.Choice(["manual", "builtin"]),
    default="manual",
    help="Speculative decoding implementation: 'manual' (default) uses our implementation "
    "with KV-cached target verification and token-level data collection, "
    "'builtin' uses MLX-LM's built-in speculative decoding",
)
@click.pass_context
def generate(ctx, prompt, max_tokens, no_collect, implementation):
    """Generate text using speculative decoding.

    Implementations:
    - manual (default): Our implementation with KV-cached target verification.
      Captures token-level disagreements for training data collection.
    - builtin: MLX-LM's built-in speculative decoding.
    """
    system = SpeculativeDecodingSystem(ctx.obj["config"])
    response = system.generate(
        prompt,
        max_tokens=max_tokens,
        collect_data=not no_collect,
        implementation=implementation,
    )
    console.print(f"\n[bold]Response:[/bold]\n{response}")


@cli.command()
@click.option(
    "--epochs", "-e", default=None, type=int, help="Number of training epochs"
)
@click.pass_context
def train(ctx, epochs):
    """Train the draft model on collected failure cases."""
    system = SpeculativeDecodingSystem(ctx.obj["config"])
    system.train(num_epochs=epochs)


@cli.command()
@click.pass_context
def evaluate(ctx):
    """Evaluate acceptance rates on test prompts."""
    system = SpeculativeDecodingSystem(ctx.obj["config"])
    system.evaluate()


@cli.command()
@click.pass_context
def stats(ctx):
    """Show current system statistics."""
    system = SpeculativeDecodingSystem(ctx.obj["config"])
    system.initialize()
    stats = system.get_stats()

    # Collector stats
    if "collector" in stats:
        table = Table(title="Data Collector Stats")
        table.add_column("Metric")
        table.add_column("Value")
        for key, value in stats["collector"].items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.2%}")
            else:
                table.add_row(key, str(value))
        console.print(table)

    # Tracker stats
    if "tracker" in stats:
        table = Table(title="Acceptance Rate Tracker")
        table.add_column("Metric")
        table.add_column("Value")
        for key, value in stats["tracker"].items():
            if key != "by_category":
                if isinstance(value, float):
                    table.add_row(key, f"{value:.2%}")
                else:
                    table.add_row(key, str(value))
        console.print(table)


@cli.command()
@click.option(
    "--implementation",
    "-i",
    type=click.Choice(["manual", "builtin"]),
    default="manual",
    help="Speculative decoding implementation: 'manual' (default) or 'builtin'",
)
@click.option(
    "--lora-adapter",
    "-l",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=None,
    help="Path to LoRA adapter checkpoint directory (e.g., data/checkpoints/best)",
)
@click.pass_context
def interactive(ctx, implementation, lora_adapter):
    """Run interactive generation mode.

    Uses the manual speculative decoding implementation by default,
    which provides KV-cached target verification and token-level data collection.
    """
    system = SpeculativeDecodingSystem(ctx.obj["config"])

    # Initialize with adapter if provided
    if lora_adapter:
        system.initialize(initial_adapter_path=lora_adapter)

    system.interactive_mode(implementation=implementation)


@cli.command()
@click.option("--prompt", "-p", required=True, help="Test prompt")
@click.option("--iterations", "-n", default=5, help="Number of iterations")
@click.option("--max-tokens", "-m", default=128, help="Maximum tokens to generate")
@click.option(
    "--implementation",
    "-i",
    type=click.Choice(["manual", "builtin"]),
    default="manual",
    help="Speculative decoding implementation to benchmark: 'manual' (default) or 'builtin'",
)
@click.pass_context
def benchmark(ctx, prompt, iterations, max_tokens, implementation):
    """Benchmark speculative vs standard decoding.

    Compares the selected speculative decoding implementation against
    standard autoregressive decoding with the target model.
    """
    system = SpeculativeDecodingSystem(ctx.obj["config"])
    system.initialize()

    import time

    # Create manual decoder if needed
    manual_decoder = system.decoder.create_manual_decoder()

    # Warm up
    console.print("[cyan]Warming up...[/cyan]")
    if implementation == "manual":
        _ = manual_decoder.generate(prompt, max_tokens=32)
    else:
        _ = system.decoder.generate(prompt, max_tokens=32)

    # Benchmark speculative decoding
    impl_name = "Manual Spec-Dec" if implementation == "manual" else "MLX-LM Spec-Dec"
    console.print(f"[cyan]Benchmarking {impl_name}...[/cyan]")
    spec_times = []
    spec_tokens = []
    spec_acceptance_rates = []

    for _ in range(iterations):
        start = time.time()
        if implementation == "manual":
            result = manual_decoder.generate(prompt, max_tokens=max_tokens)
            elapsed = time.time() - start
            spec_tokens.append(result.metrics.total_tokens_generated)
            spec_acceptance_rates.append(result.metrics.acceptance_rate)
        else:
            result = system.decoder.generate(
                prompt, max_tokens=max_tokens, collect_training_data=False
            )
            elapsed = time.time() - start
            spec_tokens.append(result.metrics.total_tokens_generated)
            spec_acceptance_rates.append(result.metrics.acceptance_rate)
        spec_times.append(elapsed)

    # Benchmark standard decoding (target model)
    console.print("[cyan]Benchmarking standard decoding (target model)...[/cyan]")
    std_times = []
    std_tokens = []

    for _ in range(iterations):
        # generate_standard now returns (text, elapsed, num_generated_tokens)
        text, elapsed, generated_tokens = system.decoder.generate_standard(
            prompt, max_tokens=max_tokens, use_target=True
        )
        std_times.append(elapsed)
        std_tokens.append(generated_tokens)

    # Results
    avg_spec_time = sum(spec_times) / len(spec_times)
    avg_std_time = sum(std_times) / len(std_times)
    avg_spec_tokens = sum(spec_tokens) / len(spec_tokens)
    avg_std_tokens = sum(std_tokens) / len(std_tokens)
    avg_acceptance_rate = sum(spec_acceptance_rates) / len(spec_acceptance_rates)

    table = Table(title=f"Benchmark Results ({impl_name})")
    table.add_column("Method")
    table.add_column("Avg Time (s)")
    table.add_column("Avg Tokens")
    table.add_column("Tokens/s")

    table.add_row(
        impl_name,
        f"{avg_spec_time:.2f}",
        f"{avg_spec_tokens:.0f}",
        f"{avg_spec_tokens / avg_spec_time:.1f}",
    )
    table.add_row(
        "Standard (Target)",
        f"{avg_std_time:.2f}",
        f"{avg_std_tokens:.0f}",
        f"{avg_std_tokens / avg_std_time:.1f}",
    )

    console.print(table)

    speedup = avg_std_time / avg_spec_time
    console.print(f"\n[bold]Speedup: {speedup:.2f}x[/bold]")
    console.print(f"[cyan]Acceptance Rate: {avg_acceptance_rate:.1%}[/cyan]")

    # Warn if acceptance rate is too low for speedup
    if avg_acceptance_rate < 0.6:
        console.print(
            f"\n[yellow]⚠ Low acceptance rate ({avg_acceptance_rate:.1%}). "
            f"Speculative decoding needs >60% acceptance to provide speedup. "
            f"Consider training the draft model on failure cases.[/yellow]"
        )


@cli.command("collect-data")
@click.option(
    "--prompts-file",
    "-f",
    type=click.Path(exists=True),
    help="File containing prompts (one per line)",
)
@click.option(
    "--prompts",
    "-p",
    multiple=True,
    help="Prompts to process (can specify multiple times)",
)
@click.option("--max-tokens", "-m", default=4096, help="Maximum tokens per generation")
@click.option(
    "--limit",
    "-l",
    type=int,
    default=None,
    help="Maximum number of prompts to process (limits total prompts from all sources)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file for detailed results (JSONL format)",
)
@click.pass_context
def collect_data(ctx, prompts_file, prompts, max_tokens, limit, output):
    """Collect token-level training data using manual speculative decoding.

    This command runs manual speculative decoding on a batch of prompts
    to collect detailed token-level disagreements between draft and target
    models. This data is more valuable for training than the overall
    acceptance rates collected in normal mode.

    Examples:
        # From file
        python -m src.main collect-data -f prompts.txt

        # From command line
        python -m src.main collect-data -p "What is Python?" -p "Explain ML"

        # With output file
        python -m src.main collect-data -f prompts.txt -o data/detailed.jsonl

        # Limit number of prompts processed
        python -m src.main collect-data -f prompts.txt --limit 10
    """
    # Collect prompts from all sources
    all_prompts = list(prompts)

    if prompts_file:
        with open(prompts_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    all_prompts.append(line)

    if not all_prompts:
        console.print(
            "[red]Error: No prompts provided. Use --prompts-file or --prompts[/red]"
        )
        return

    # Apply limit if specified (randomly sample if limiting)
    original_count = len(all_prompts)
    if limit is not None and limit > 0:
        if limit < original_count:
            all_prompts = random.sample(all_prompts, limit)
            console.print(
                f"[yellow]Randomly sampled {limit} prompts (from {original_count} total)[/yellow]"
            )
        # If limit >= original_count, use all prompts (no need to sample)

    console.print(
        Panel(
            f"Running data collection on [bold]{len(all_prompts)}[/bold] prompts\n"
            f"Mode: [cyan]detailed[/cyan] (manual speculative decoding)\n"
            f"Max tokens: {max_tokens}",
            title="Data Collection Mode",
        )
    )

    system = SpeculativeDecodingSystem(ctx.obj["config"])
    system.collect_data_batch(
        prompts=all_prompts,
        max_tokens=max_tokens,
        output_file=output,
    )

    console.print("\n[green]Data collection complete![/green]")


@cli.command("benchmark-suite")
@click.option(
    "--prompts-file",
    "-f",
    type=click.Path(exists=True),
    default=None,
    help="File with prompts (one per line). If not provided, uses evaluation.test_prompts from config.",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    default=None,
    help="Limit number of prompts to benchmark (randomly samples if limiting)",
)
@click.option(
    "--max-tokens",
    "-m",
    type=int,
    default=256,
    help="Maximum tokens per generation",
)
@click.option(
    "--iterations",
    "-n",
    type=int,
    default=1,
    help="Number of iterations per prompt for averaging",
)
@click.option(
    "--lora-adapter",
    "-a",
    type=str,
    default="best",
    help="LoRA adapter path or alias (best/latest). Default: best",
)
@click.option(
    "--skip-lora",
    is_flag=True,
    help="Skip LoRA benchmarks (only run target-only and base spec dec)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output JSON file path for results",
)
@click.pass_context
def benchmark_suite(
    ctx, prompts_file, limit, max_tokens, iterations, lora_adapter, skip_lora, output
):
    """
    Run comprehensive benchmark suite comparing decoding modes.

    Compares three modes:
    1. Target-only: Standard autoregressive decoding (baseline)
    2. Spec Dec (Base): Speculative decoding with base draft model
    3. Spec Dec (LoRA): Speculative decoding with LoRA-adapted draft model

    Examples:
        # Use config test prompts (default)
        python -m src.main benchmark-suite

        # Use prompts from file, limit to 15
        python -m src.main benchmark-suite -f prompts.txt --limit 15

        # Run 3 iterations per prompt for averaging, export results
        python -m src.main benchmark-suite -f prompts.txt -n 3 -o results.json

        # Skip LoRA benchmarks
        python -m src.main benchmark-suite --skip-lora

        # Use specific adapter
        python -m src.main benchmark-suite -a data/checkpoints/best
    """
    from .benchmark import run_benchmark_suite

    # Load prompts
    if prompts_file:
        # Load from file
        prompts = []
        with open(prompts_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    prompts.append(line)
    else:
        # Use config evaluation prompts
        prompts = ctx.obj["config"].get("evaluation", {}).get("test_prompts", [])
        if not prompts:
            console.print(
                "[red]Error: No prompts found in config.evaluation.test_prompts[/red]"
            )
            console.print(
                "Either provide --prompts-file or add test_prompts to config.yaml"
            )
            return

    if not prompts:
        console.print("[red]Error: No prompts to benchmark[/red]")
        return

    # Apply limit (randomly sample)
    original_count = len(prompts)
    if limit is not None and limit > 0 and limit < len(prompts):
        prompts = random.sample(prompts, limit)
        console.print(
            f"[yellow]Randomly sampled {limit} prompts (from {original_count} total)[/yellow]"
        )

    # Initialize system
    system = SpeculativeDecodingSystem(ctx.obj["config"])

    # Resolve adapter path (only needs config, not full initialization)
    adapter_path = None
    if not skip_lora:
        try:
            # resolve_adapter_path() and validate_adapter_structure() only need self.config
            # Don't initialize yet to avoid loading models twice
            adapter_path = str(system.resolve_adapter_path(lora_adapter))

            # Validate adapter structure
            _ = system.validate_adapter_structure(Path(adapter_path))

            console.print(
                f"[cyan]LoRA adapter validated: {Path(adapter_path).name}[/cyan]"
            )
        except FileNotFoundError as e:
            console.print(f"[red]LoRA adapter not found: {e}[/red]")
            console.print("[yellow]Continuing without LoRA benchmarks[/yellow]")
            skip_lora = True
        except Exception as e:
            console.print(f"[red]Error validating adapter: {e}[/red]")
            console.print("[yellow]Continuing without LoRA benchmarks[/yellow]")
            skip_lora = True

    # Now initialize system (load models once)
    system.initialize()

    # Run benchmark suite
    run_benchmark_suite(
        system=system,
        prompts=prompts,
        max_tokens=max_tokens,
        iterations=iterations,
        lora_adapter_path=adapter_path,
        skip_lora=skip_lora,
        output_path=output,
    )

    console.print("\n[green]✓ Benchmark suite complete![/green]")


@cli.command()
@click.pass_context
def demo(ctx):
    """Run a quick demo of the system."""
    console.print(
        Panel(
            "[bold]Speculative Decoding with Adaptive LoRA Demo[/bold]\n\n"
            "This demo will:\n"
            "1. Load the target (Qwen2.5-7B-4bit) and draft (Qwen2.5-0.5B) models\n"
            "2. Generate text using speculative decoding\n"
            "3. Show acceptance rate metrics\n"
            "4. Demonstrate failure case collection",
            title="Welcome",
        )
    )

    system = SpeculativeDecodingSystem(ctx.obj["config"])
    system.initialize()

    # Demo prompts
    demo_prompts = [
        "What is the capital of France?",
        "Write a short poem about coding.",
        "Explain machine learning in one sentence.",
    ]

    console.print("\n[bold cyan]Running demo generations...[/bold cyan]\n")

    for i, prompt in enumerate(demo_prompts, 1):
        console.print(f"\n[bold]Demo {i}/{len(demo_prompts)}[/bold]")
        console.print(f"[green]Prompt:[/green] {prompt}")

        response = system.generate(prompt, max_tokens=64)
        console.print(f"[blue]Response:[/blue] {response}")

    # Show stats
    console.print("\n[bold cyan]Current Statistics[/bold cyan]")
    stats = system.get_stats()
    console.print(stats)

    console.print("\n[green]Demo complete![/green]")
    console.print(
        "Run [bold]'python -m src.main interactive'[/bold] for interactive mode."
    )


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
