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
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import click
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

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
    
    def initialize(self) -> None:
        """Initialize all components."""
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
            tracker_path = Path(self.config["data"]["failures_dir"]) / "rate_tracker.json"
            if tracker_path.exists():
                self.rate_tracker.load(str(tracker_path))
            
            self._initialized = True
        
        # Print memory usage
        memory = self.model_manager.estimate_memory_usage()
        console.print(Panel(
            f"Target Model: {memory['target_params_m']:.1f}M params ({memory['target_model_gb']:.2f} GB)\n"
            f"Draft Model: {memory['draft_params_m']:.1f}M params ({memory['draft_model_gb']:.2f} GB)\n"
            f"Total: {memory['total_gb']:.2f} GB",
            title="Memory Usage",
        ))
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        collect_data: bool = True,
        mode: str = "fast",
    ) -> str:
        """
        Generate text using speculative decoding.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            collect_data: Whether to collect training data
            mode: "fast" for built-in speculative decoding, 
                  "detailed" for manual implementation with token-level data
            
        Returns:
            Generated text
        """
        if not self._initialized:
            self.initialize()
        
        max_tokens = max_tokens or self.config["speculative"]["max_tokens"]

        # Clear cache periodically to prevent memory accumulation
        cache_clear_freq = self.config.get("memory", {}).get("cache_clear_frequency", 10)
        if self._generation_count > 0 and self._generation_count % cache_clear_freq == 0:
            self.model_manager.clear_cache()
            logger.debug(f"Cleared cache after {self._generation_count} generations")

        if mode == "detailed":
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
                )
                
                if should_train:
                    console.print(
                        "[yellow]Training threshold reached! "
                        "Run 'train' command to fine-tune the draft model.[/yellow]"
                    )
            
            # Display detailed metrics
            metrics_table = Table(title="Generation Metrics (Detailed Mode)", show_header=False)
            metrics_table.add_row("Acceptance Rate", f"{result.metrics.acceptance_rate:.1%}")
            metrics_table.add_row("Tokens/Second", f"{result.metrics.tokens_per_second:.1f}")
            metrics_table.add_row("Total Tokens", str(result.metrics.total_tokens_generated))
            metrics_table.add_row("Draft Proposed", str(result.metrics.draft_tokens_proposed))
            metrics_table.add_row("Draft Accepted", str(result.metrics.draft_tokens_accepted))
            metrics_table.add_row("[bold]Disagreements[/bold]", str(len(result.disagreements)))
            metrics_table.add_row("Is Failure Case", "Yes" if result.is_failure_case else "No")
            
            # Show disagreement details if any
            if result.disagreements:
                high_conf = sum(1 for d in result.disagreements if d.is_high_confidence_failure)
                metrics_table.add_row("High-Confidence Failures", str(high_conf))
            
            console.print(metrics_table)
            
            return result.text
        
        else:
            # Use fast built-in speculative decoding
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
            metrics_table = Table(title="Generation Metrics", show_header=False)
            metrics_table.add_row("Acceptance Rate", f"{result.metrics.acceptance_rate:.1%}")
            metrics_table.add_row("Tokens/Second", f"{result.metrics.tokens_per_second:.1f}")
            metrics_table.add_row("Total Tokens", str(result.metrics.total_tokens_generated))
            metrics_table.add_row("Draft Proposed", str(result.metrics.draft_tokens_proposed))
            metrics_table.add_row("Draft Accepted", str(result.metrics.draft_tokens_accepted))
            metrics_table.add_row("Is Failure Case", "Yes" if result.is_failure_case else "No")
            
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
        
        from .speculative_manual import run_data_collection_batch
        
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
        stats["acceptance_rate"] = (
            stats["total_accepted"] / max(stats["total_proposed"], 1)
        )
        stats["avg_disagreements"] = stats["total_disagreements"] / max(len(prompts), 1)
        stats["failure_cases"] = sum(1 for r in results if r.is_failure_case)
        stats["high_confidence_failures"] = sum(
            1 for r in results
            for d in r.disagreements
            if d.is_high_confidence_failure
        )
        
        # Save detailed results if output file specified
        if output_file:
            import json
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            detailed_results = []
            for r in results:
                detailed_results.append({
                    "prompt": r.prompt,
                    "text": r.text,
                    "tokens": r.tokens,
                    "acceptance_rate": r.metrics.acceptance_rate,
                    "disagreements": [d.to_dict() for d in r.disagreements],
                    "is_failure": r.is_failure_case,
                })
            
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
        table.add_row("[bold]Total Disagreements[/bold]", str(stats["total_disagreements"]))
        table.add_row("Avg Disagreements/Prompt", f"{stats['avg_disagreements']:.1f}")
        table.add_row("Failure Cases", str(stats["failure_cases"]))
        table.add_row("High-Confidence Failures", str(stats["high_confidence_failures"]))
        
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
        
        console.print(Panel(
            f"Failure cases: {stats['num_failure_cases']}\n"
            f"Replay buffer: {stats['num_replay_cases']}\n"
            f"Avg failure acceptance: {stats.get('avg_failure_acceptance', 0):.1%}",
            title="Training Data Summary",
        ))
        
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
        draft_model, tokenizer = self.model_manager.get_draft_model()
        
        # Initialize trainer
        trainer = LoRATrainer(
            model=draft_model,
            tokenizer=tokenizer,
            lora_config=lora_config,
            learning_rate=self.config["training"]["learning_rate"],
            batch_size=self.config["training"]["batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
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
        
        # Update decoder with trained model
        self.decoder.draft_model = trainer.model
        
        # Clear failure cases
        self.data_collector.clear_failure_cases()
        
        # Display results
        results_table = Table(title="Training Results", show_header=False)
        results_table.add_row("Total Steps", str(metrics.total_steps))
        results_table.add_row("Average Loss", f"{metrics.avg_loss:.4f}")
        results_table.add_row("Training Time", f"{metrics.training_time_seconds:.1f}s")
        
        console.print(results_table)
        console.print("[green]Training complete! Draft model updated.[/green]")
    
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
        console.print(Panel(
            f"Overall Acceptance Rate: {results['overall_acceptance_rate']:.1%}\n"
            f"Overall Tokens/Second: {results['overall_tokens_per_second']:.1f}",
            title="Summary",
        ))
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        if not self._initialized:
            console.print("[yellow]System not initialized. Run a command first.[/yellow]")
            return {}
        
        collector_stats = self.data_collector.get_stats()
        tracker_stats = self.rate_tracker.get_stats()
        
        return {
            "collector": collector_stats,
            "tracker": tracker_stats,
        }
    
    def interactive_mode(self) -> None:
        """Run interactive generation mode."""
        if not self._initialized:
            self.initialize()
        
        console.print(Panel(
            "Enter prompts to generate text. Commands:\n"
            "  /quit - Exit interactive mode\n"
            "  /stats - Show statistics\n"
            "  /train - Train on collected data\n"
            "  /eval - Run evaluation\n"
            "  /clear - Clear screen",
            title="Interactive Mode",
        ))
        
        while True:
            try:
                prompt = console.input("\n[bold green]Prompt:[/bold green] ")
                
                if not prompt:
                    continue
                
                if prompt.startswith("/"):
                    cmd = prompt[1:].lower().strip()
                    
                    if cmd == "quit" or cmd == "exit":
                        break
                    elif cmd == "stats":
                        stats = self.get_stats()
                        console.print(stats)
                    elif cmd == "train":
                        self.train()
                    elif cmd == "eval":
                        self.evaluate()
                    elif cmd == "clear":
                        console.clear()
                    else:
                        console.print(f"[red]Unknown command: {cmd}[/red]")
                    continue
                
                # Generate response
                console.print("\n[bold blue]Response:[/bold blue]")
                response = self.generate(prompt)
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
    "--mode",
    type=click.Choice(["fast", "detailed"]),
    default="fast",
    help="Generation mode: 'fast' uses MLX-LM's optimized speculative decoding, "
         "'detailed' uses manual implementation that captures token-level disagreements"
)
@click.pass_context
def generate(ctx, prompt, max_tokens, no_collect, mode):
    """Generate text using speculative decoding.
    
    Modes:
    - fast: Uses MLX-LM's built-in speculative decoding (default, recommended for production)
    - detailed: Uses manual implementation that captures every token-level disagreement 
      (slower ~20%, but provides more valuable training data)
    """
    system = SpeculativeDecodingSystem(ctx.obj["config"])
    response = system.generate(
        prompt, 
        max_tokens=max_tokens, 
        collect_data=not no_collect,
        mode=mode,
    )
    console.print(f"\n[bold]Response:[/bold]\n{response}")


@cli.command()
@click.option("--epochs", "-e", default=None, type=int, help="Number of training epochs")
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
@click.pass_context
def interactive(ctx):
    """Run interactive generation mode."""
    system = SpeculativeDecodingSystem(ctx.obj["config"])
    system.interactive_mode()


@cli.command()
@click.option("--prompt", "-p", required=True, help="Test prompt")
@click.option("--iterations", "-n", default=5, help="Number of iterations")
@click.option("--max-tokens", "-m", default=128, help="Maximum tokens to generate")
@click.pass_context
def benchmark(ctx, prompt, iterations, max_tokens):
    """Benchmark speculative vs standard decoding."""
    system = SpeculativeDecodingSystem(ctx.obj["config"])
    system.initialize()
    
    import time
    
    # Warm up
    console.print("[cyan]Warming up...[/cyan]")
    _ = system.decoder.generate(prompt, max_tokens=32)
    
    # Benchmark speculative decoding
    console.print("[cyan]Benchmarking speculative decoding...[/cyan]")
    spec_times = []
    spec_tokens = []
    
    for _ in range(iterations):
        start = time.time()
        result = system.decoder.generate(prompt, max_tokens=max_tokens, collect_training_data=False)
        elapsed = time.time() - start
        spec_times.append(elapsed)
        spec_tokens.append(result.metrics.total_tokens_generated)
    
    # Benchmark standard decoding (target model)
    console.print("[cyan]Benchmarking standard decoding (target model)...[/cyan]")
    std_times = []
    std_tokens = []
    
    for _ in range(iterations):
        start = time.time()
        text, elapsed = system.decoder.generate_standard(prompt, max_tokens=max_tokens, use_target=True)
        std_times.append(elapsed)
        std_tokens.append(len(system.decoder.tokenizer.encode(text)))
    
    # Results
    avg_spec_time = sum(spec_times) / len(spec_times)
    avg_std_time = sum(std_times) / len(std_times)
    avg_spec_tokens = sum(spec_tokens) / len(spec_tokens)
    avg_std_tokens = sum(std_tokens) / len(std_tokens)
    
    table = Table(title="Benchmark Results")
    table.add_column("Method")
    table.add_column("Avg Time (s)")
    table.add_column("Avg Tokens")
    table.add_column("Tokens/s")
    
    table.add_row(
        "Speculative",
        f"{avg_spec_time:.2f}",
        f"{avg_spec_tokens:.0f}",
        f"{avg_spec_tokens/avg_spec_time:.1f}",
    )
    table.add_row(
        "Standard",
        f"{avg_std_time:.2f}",
        f"{avg_std_tokens:.0f}",
        f"{avg_std_tokens/avg_std_time:.1f}",
    )
    
    console.print(table)
    
    speedup = avg_std_time / avg_spec_time
    console.print(f"\n[bold]Speedup: {speedup:.2f}x[/bold]")


@cli.command("collect-data")
@click.option(
    "--prompts-file", "-f",
    type=click.Path(exists=True),
    help="File containing prompts (one per line)"
)
@click.option(
    "--prompts", "-p",
    multiple=True,
    help="Prompts to process (can specify multiple times)"
)
@click.option("--max-tokens", "-m", default=256, help="Maximum tokens per generation")
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output file for detailed results (JSONL format)"
)
@click.pass_context
def collect_data(ctx, prompts_file, prompts, max_tokens, output):
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
        console.print("[red]Error: No prompts provided. Use --prompts-file or --prompts[/red]")
        return
    
    console.print(Panel(
        f"Running data collection on [bold]{len(all_prompts)}[/bold] prompts\n"
        f"Mode: [cyan]detailed[/cyan] (manual speculative decoding)\n"
        f"Max tokens: {max_tokens}",
        title="Data Collection Mode",
    ))
    
    system = SpeculativeDecodingSystem(ctx.obj["config"])
    stats = system.collect_data_batch(
        prompts=all_prompts,
        max_tokens=max_tokens,
        output_file=output,
    )
    
    console.print("\n[green]Data collection complete![/green]")


@cli.command()
@click.pass_context
def demo(ctx):
    """Run a quick demo of the system."""
    console.print(Panel(
        "[bold]Speculative Decoding with Adaptive LoRA Demo[/bold]\n\n"
        "This demo will:\n"
        "1. Load the target (Qwen2.5-7B-4bit) and draft (Qwen2.5-0.5B) models\n"
        "2. Generate text using speculative decoding\n"
        "3. Show acceptance rate metrics\n"
        "4. Demonstrate failure case collection",
        title="Welcome",
    ))
    
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
    console.print("Run [bold]'python -m src.main interactive'[/bold] for interactive mode.")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
