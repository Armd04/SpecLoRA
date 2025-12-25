"""
Benchmark Suite for SpecLoRA

This module provides comprehensive benchmarking capabilities to compare:
1. Target-only decoding (baseline)
2. Speculative decoding with base draft model
3. Speculative decoding with LoRA-adapted draft model

Key metrics: tokens/second, time-to-last-token (TTLT), acceptance rate
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import mlx.core as mx
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

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run (one prompt, one mode)."""

    mode: str  # "target_only", "spec_base", "spec_lora"
    prompt: str
    tokens_per_second: float
    time_to_last_token: float  # Total generation time (TTLT)
    acceptance_rate: Optional[float]  # Only for spec dec modes
    total_tokens: int
    iteration: int  # Which iteration this is (for averaging)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ModeSummary:
    """Aggregate statistics for one benchmark mode across all prompts."""

    mode: str
    avg_tokens_per_second: float
    avg_time_to_last_token: float
    avg_acceptance_rate: Optional[float]
    total_tokens: int
    num_prompts: int

    # Min/max for understanding variance
    min_tokens_per_second: float
    max_tokens_per_second: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class BenchmarkSummary:
    """Complete benchmark summary with per-mode aggregates and per-prompt details."""

    # Metadata
    timestamp: str
    num_prompts: int
    max_tokens: int
    iterations: int
    adapter_path: Optional[str]

    # Per-mode summaries
    target_only: Optional[ModeSummary] = None
    spec_base: Optional[ModeSummary] = None
    spec_lora: Optional[ModeSummary] = None

    # Per-prompt results (grouped by prompt)
    per_prompt: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "metadata": {
                "timestamp": self.timestamp,
                "num_prompts": self.num_prompts,
                "max_tokens": self.max_tokens,
                "iterations": self.iterations,
                "adapter_path": self.adapter_path,
            },
            "summary": {},
            "per_prompt": self.per_prompt,
        }

        if self.target_only:
            result["summary"]["target_only"] = self.target_only.to_dict()
        if self.spec_base:
            result["summary"]["spec_base"] = self.spec_base.to_dict()
        if self.spec_lora:
            result["summary"]["spec_lora"] = self.spec_lora.to_dict()

        return result


class BenchmarkRunner:
    """
    Runs benchmark comparisons across different decoding modes.

    Handles model loading, warmup, and result aggregation.
    """

    def __init__(self, system):
        """
        Initialize benchmark runner.

        Args:
            system: SpeculativeDecodingSystem instance (already initialized)
        """
        self.system = system

    def run_target_only(self, prompt: str, max_tokens: int) -> BenchmarkResult:
        """
        Run standard autoregressive decoding with target model only.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            BenchmarkResult with timing metrics
        """
        # Use the decoder's generate_standard method
        text, elapsed = self.system.decoder.generate_standard(
            prompt=prompt,
            max_tokens=max_tokens,
            use_target=True,
        )

        # Count generated tokens (exclude prompt)
        formatted_prompt = self.system.decoder.format_prompt(prompt)
        prompt_tokens = self.system.decoder.tokenizer.encode(formatted_prompt)
        total_tokens = len(self.system.decoder.tokenizer.encode(text))
        generated_tokens = max(0, total_tokens - len(prompt_tokens))

        tokens_per_second = generated_tokens / max(elapsed, 0.001)

        return BenchmarkResult(
            mode="target_only",
            prompt=prompt,
            tokens_per_second=tokens_per_second,
            time_to_last_token=elapsed,
            acceptance_rate=None,
            total_tokens=generated_tokens,
            iteration=0,
        )

    def run_spec_base(self, prompt: str, max_tokens: int) -> BenchmarkResult:
        """
        Run speculative decoding with base (unmodified) draft model.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            BenchmarkResult with timing and acceptance metrics
        """
        result = self.system.decoder.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            collect_training_data=False,
        )

        return BenchmarkResult(
            mode="spec_base",
            prompt=prompt,
            tokens_per_second=result.metrics.tokens_per_second,
            time_to_last_token=result.metrics.total_time_seconds,
            acceptance_rate=result.metrics.acceptance_rate,
            total_tokens=result.metrics.total_tokens_generated,
            iteration=0,
        )

    def run_spec_lora(self, prompt: str, max_tokens: int) -> BenchmarkResult:
        """
        Run speculative decoding with LoRA-adapted draft model.

        Note: Assumes LoRA adapter is already loaded into the system.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            BenchmarkResult with timing and acceptance metrics
        """
        # Same as spec_base - the system's decoder already has the LoRA adapter loaded
        result = self.system.decoder.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            collect_training_data=False,
        )

        return BenchmarkResult(
            mode="spec_lora",
            prompt=prompt,
            tokens_per_second=result.metrics.tokens_per_second,
            time_to_last_token=result.metrics.total_time_seconds,
            acceptance_rate=result.metrics.acceptance_rate,
            total_tokens=result.metrics.total_tokens_generated,
            iteration=0,
        )

    def warmup(self, prompt: str, max_tokens: int = 32) -> None:
        """
        Warm up models with a short generation.

        Args:
            prompt: Warmup prompt
            max_tokens: Short token limit for warmup
        """
        console.print("[cyan]Warming up models...[/cyan]")

        # Warmup target-only
        _ = self.run_target_only(prompt, max_tokens)

        # Warmup spec dec
        _ = self.run_spec_base(prompt, max_tokens)

        # Clear cache after warmup
        mx.clear_cache()

        console.print("[green]✓ Warmup complete[/green]")


def aggregate_results(results: List[BenchmarkResult], mode: str) -> ModeSummary:
    """
    Aggregate results for a single mode.

    Args:
        results: List of BenchmarkResult for this mode
        mode: Mode name (for validation)

    Returns:
        ModeSummary with aggregate statistics
    """
    mode_results = [r for r in results if r.mode == mode]

    if not mode_results:
        raise ValueError(f"No results found for mode: {mode}")

    avg_tps = sum(r.tokens_per_second for r in mode_results) / len(mode_results)
    avg_ttlt = sum(r.time_to_last_token for r in mode_results) / len(mode_results)
    total_tokens = sum(r.total_tokens for r in mode_results)

    # Acceptance rate (only for spec dec modes)
    acceptance_rates = [
        r.acceptance_rate for r in mode_results if r.acceptance_rate is not None
    ]
    avg_acceptance = (
        sum(acceptance_rates) / len(acceptance_rates) if acceptance_rates else None
    )

    return ModeSummary(
        mode=mode,
        avg_tokens_per_second=avg_tps,
        avg_time_to_last_token=avg_ttlt,
        avg_acceptance_rate=avg_acceptance,
        total_tokens=total_tokens,
        num_prompts=len(mode_results),
        min_tokens_per_second=min(r.tokens_per_second for r in mode_results),
        max_tokens_per_second=max(r.tokens_per_second for r in mode_results),
    )


def group_results_by_prompt(results: List[BenchmarkResult]) -> List[Dict[str, Any]]:
    """
    Group results by prompt for per-prompt comparison.

    Args:
        results: All benchmark results

    Returns:
        List of dicts with per-prompt results for each mode
    """
    # Get unique prompts
    prompts = list(dict.fromkeys(r.prompt for r in results))

    grouped = []
    for prompt in prompts:
        prompt_results = [r for r in results if r.prompt == prompt]

        entry = {
            "prompt": prompt[:60] + "..." if len(prompt) > 60 else prompt,
            "full_prompt": prompt,
        }

        # Add results for each mode
        for mode in ["target_only", "spec_base", "spec_lora"]:
            mode_results = [r for r in prompt_results if r.mode == mode]
            if mode_results:
                # Average across iterations
                entry[mode] = {
                    "tokens_per_second": sum(r.tokens_per_second for r in mode_results)
                    / len(mode_results),
                    "time_to_last_token": sum(
                        r.time_to_last_token for r in mode_results
                    )
                    / len(mode_results),
                    "acceptance_rate": sum(
                        r.acceptance_rate
                        for r in mode_results
                        if r.acceptance_rate is not None
                    )
                    / len(mode_results)
                    if any(r.acceptance_rate for r in mode_results)
                    else None,
                    "total_tokens": mode_results[
                        0
                    ].total_tokens,  # Should be same for all iterations
                }

        grouped.append(entry)

    return grouped


def display_results(summary: BenchmarkSummary) -> None:
    """
    Display benchmark results in rich console tables.

    Args:
        summary: BenchmarkSummary to display
    """
    # Main summary table
    table = Table(
        title="SpecLoRA Benchmark Summary", show_header=True, border_style="cyan"
    )
    table.add_column("Mode", style="cyan", no_wrap=True)
    table.add_column("Avg Tok/s", justify="right")
    table.add_column("Avg TTLT", justify="right")
    table.add_column("Avg Acceptance", justify="right")
    table.add_column("Total Tokens", justify="right")

    # Add rows for each mode
    if summary.target_only:
        table.add_row(
            "Target Only",
            f"{summary.target_only.avg_tokens_per_second:.1f}",
            f"{summary.target_only.avg_time_to_last_token:.2f}s",
            "-",
            str(summary.target_only.total_tokens),
        )

    if summary.spec_base:
        table.add_row(
            "Spec Dec (Base)",
            f"{summary.spec_base.avg_tokens_per_second:.1f}",
            f"{summary.spec_base.avg_time_to_last_token:.2f}s",
            f"{summary.spec_base.avg_acceptance_rate:.1%}"
            if summary.spec_base.avg_acceptance_rate
            else "-",
            str(summary.spec_base.total_tokens),
        )

    if summary.spec_lora:
        table.add_row(
            "Spec Dec (LoRA)",
            f"{summary.spec_lora.avg_tokens_per_second:.1f}",
            f"{summary.spec_lora.avg_time_to_last_token:.2f}s",
            f"{summary.spec_lora.avg_acceptance_rate:.1%}"
            if summary.spec_lora.avg_acceptance_rate
            else "-",
            str(summary.spec_lora.total_tokens),
        )

    console.print(table)

    # Speedup comparison
    if summary.target_only:
        speedup_lines = []

        if summary.spec_base:
            speedup = (
                summary.spec_base.avg_tokens_per_second
                / summary.target_only.avg_tokens_per_second
            )
            speedup_lines.append(f"Base: {speedup:.2f}x")

        if summary.spec_lora:
            speedup = (
                summary.spec_lora.avg_tokens_per_second
                / summary.target_only.avg_tokens_per_second
            )
            speedup_lines.append(f"LoRA: {speedup:.2f}x")

        if speedup_lines:
            console.print(
                f"\n[bold cyan]Speedup vs Target:[/bold cyan] {', '.join(speedup_lines)}"
            )

    # LoRA improvement over base
    if summary.spec_base and summary.spec_lora:
        tps_improvement = (
            summary.spec_lora.avg_tokens_per_second
            / summary.spec_base.avg_tokens_per_second
            - 1
        ) * 100
        if (
            summary.spec_base.avg_acceptance_rate
            and summary.spec_lora.avg_acceptance_rate
        ):
            acc_improvement = (
                summary.spec_lora.avg_acceptance_rate
                / summary.spec_base.avg_acceptance_rate
                - 1
            ) * 100
            console.print(
                f"[bold green]LoRA improvement over Base:[/bold green] "
                f"+{tps_improvement:.1f}% tok/s, +{acc_improvement:.1f}% acceptance"
            )
        else:
            console.print(
                f"[bold green]LoRA improvement over Base:[/bold green] "
                f"+{tps_improvement:.1f}% tok/s"
            )

    # Per-prompt details (abbreviated)
    if summary.per_prompt:
        console.print("\n[bold]Per-Prompt Results (first 5):[/bold]")
        detail_table = Table(show_header=True, border_style="dim")
        detail_table.add_column("Prompt", style="dim", no_wrap=False, max_width=40)
        detail_table.add_column("Target", justify="right")
        detail_table.add_column("Base", justify="right")
        detail_table.add_column("LoRA", justify="right")

        for i, prompt_data in enumerate(summary.per_prompt[:5]):
            target_tps = prompt_data.get("target_only", {}).get("tokens_per_second", 0)
            base_tps = prompt_data.get("spec_base", {}).get("tokens_per_second", 0)
            lora_tps = prompt_data.get("spec_lora", {}).get("tokens_per_second", 0)

            detail_table.add_row(
                prompt_data["prompt"],
                f"{target_tps:.1f}" if target_tps else "-",
                f"{base_tps:.1f}" if base_tps else "-",
                f"{lora_tps:.1f}" if lora_tps else "-",
            )

        console.print(detail_table)

        if len(summary.per_prompt) > 5:
            console.print(
                f"[dim]... and {len(summary.per_prompt) - 5} more prompts[/dim]"
            )


def export_results_json(summary: BenchmarkSummary, output_path: str) -> None:
    """
    Export benchmark results to JSON file.

    Args:
        summary: BenchmarkSummary to export
        output_path: Path to output JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)

    console.print(f"[green]✓ Results exported to {output_path}[/green]")


def run_benchmark_suite(
    system,
    prompts: List[str],
    max_tokens: int,
    iterations: int,
    lora_adapter_path: Optional[str],
    skip_lora: bool,
    output_path: Optional[str],
) -> BenchmarkSummary:
    """
    Run complete benchmark suite.

    Args:
        system: SpeculativeDecodingSystem instance
        prompts: List of prompts to benchmark
        max_tokens: Maximum tokens per generation
        iterations: Number of iterations per prompt
        lora_adapter_path: Path to LoRA adapter (or None)
        skip_lora: Skip LoRA benchmarks
        output_path: Optional JSON output path

    Returns:
        BenchmarkSummary with all results
    """
    runner = BenchmarkRunner(system)
    all_results: List[BenchmarkResult] = []

    # Warmup
    warmup_prompt = prompts[0] if prompts else "Hello, world!"
    runner.warmup(warmup_prompt)

    console.print(
        Panel(
            f"[bold]Benchmark Configuration[/bold]\n"
            f"Prompts: {len(prompts)}\n"
            f"Max tokens: {max_tokens}\n"
            f"Iterations: {iterations}\n"
            f"LoRA adapter: {lora_adapter_path if lora_adapter_path and not skip_lora else 'None'}",
            title="Starting Benchmark Suite",
            border_style="cyan",
        )
    )

    # Phase 1: Target Only
    console.print("\n[bold cyan]Phase 1: Target Only Decoding[/bold cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Running target-only benchmarks...", total=len(prompts) * iterations
        )

        for prompt in prompts:
            for iteration in range(iterations):
                result = runner.run_target_only(prompt, max_tokens)
                result.iteration = iteration
                all_results.append(result)
                progress.advance(task)

    # Phase 2: Spec Dec (Base)
    console.print("\n[bold cyan]Phase 2: Speculative Decoding (Base Draft)[/bold cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Running spec dec (base) benchmarks...", total=len(prompts) * iterations
        )

        for prompt in prompts:
            for iteration in range(iterations):
                result = runner.run_spec_base(prompt, max_tokens)
                result.iteration = iteration
                all_results.append(result)
                progress.advance(task)

    # Phase 3: Spec Dec (LoRA) - if not skipped
    if not skip_lora and lora_adapter_path:
        console.print("\n[bold cyan]Phase 3: Loading LoRA Adapter[/bold cyan]")
        try:
            # Load adapter into system
            metadata = system.load_lora_adapter(lora_adapter_path)
            console.print(
                f"[green]✓ Loaded adapter: {Path(lora_adapter_path).name}[/green]"
            )
            console.print(
                f"  Rank: {metadata.get('rank', 'unknown')}, Alpha: {metadata.get('alpha', 'unknown')}"
            )

            console.print(
                "\n[bold cyan]Phase 3: Speculative Decoding (LoRA Draft)[/bold cyan]"
            )
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Running spec dec (LoRA) benchmarks...",
                    total=len(prompts) * iterations,
                )

                for prompt in prompts:
                    for iteration in range(iterations):
                        result = runner.run_spec_lora(prompt, max_tokens)
                        result.iteration = iteration
                        all_results.append(result)
                        progress.advance(task)
        except Exception as e:
            console.print(f"[red]✗ Failed to load LoRA adapter: {e}[/red]")
            console.print("[yellow]Skipping LoRA benchmarks[/yellow]")

    # Aggregate results
    console.print("\n[bold cyan]Computing aggregate statistics...[/bold cyan]")

    summary = BenchmarkSummary(
        timestamp=datetime.now().isoformat(),
        num_prompts=len(prompts),
        max_tokens=max_tokens,
        iterations=iterations,
        adapter_path=lora_adapter_path if not skip_lora else None,
    )

    # Aggregate per mode
    if any(r.mode == "target_only" for r in all_results):
        summary.target_only = aggregate_results(all_results, "target_only")

    if any(r.mode == "spec_base" for r in all_results):
        summary.spec_base = aggregate_results(all_results, "spec_base")

    if any(r.mode == "spec_lora" for r in all_results):
        summary.spec_lora = aggregate_results(all_results, "spec_lora")

    # Group by prompt
    summary.per_prompt = group_results_by_prompt(all_results)

    # Display results
    console.print("\n")
    display_results(summary)

    # Export to JSON if requested
    if output_path:
        export_results_json(summary, output_path)

    return summary
