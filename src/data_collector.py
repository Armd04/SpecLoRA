"""
Failure Case Collection for Adaptive LoRA Training

This module collects cases where the draft model's acceptance rate is low,
indicating it disagrees significantly with the target model. These cases
are used to fine-tune the draft model via LoRA.

The collector also maintains a replay buffer of successful cases to prevent
catastrophic forgetting during training.
"""

import json
import logging
import random
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .speculative import SpeculativeResult

logger = logging.getLogger(__name__)


@dataclass
class TokenLevelDisagreement:
    """
    Captures a single token-level disagreement between draft and target models.

    Used for targeted training where we focus on specific positions where
    the draft model failed to predict correctly.
    """

    # Position in the generated sequence where disagreement occurred
    position: int

    # What the draft model predicted
    draft_token: int

    # What the target model chose (ground truth for training)
    target_token: int

    # Softmax probability of draft's choice (confidence)
    draft_confidence: float

    # Softmax probability of target's choice
    target_confidence: float

    # Last N tokens before this position (context for understanding failure)
    context_tokens: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenLevelDisagreement":
        """Create from dictionary."""
        return cls(**data)

    @property
    def is_high_confidence_failure(self) -> bool:
        """
        Check if this was a high-confidence failure (draft was confident but wrong).
        These are particularly valuable for training as they indicate systematic errors.
        """
        return self.draft_confidence > 0.4


@dataclass
class TrainingExample:
    """A single training example for LoRA fine-tuning."""

    # Unique identifier
    id: str

    # Original prompt
    prompt: str

    # Tokenized prompt as used during generation (preferred for training).
    # When present, this should be the token IDs of the FULL formatted chat prompt
    # (i.e., after tokenizer.apply_chat_template(..., add_generation_prompt=True)).
    #
    # Storing prompt_tokens avoids tokenizer/template drift between collection and training.
    prompt_tokens: Optional[List[int]] = None

    # What the draft model generated
    draft_output: List[int]

    # What the target model would have generated (ground truth)
    target_output: List[int]

    # Acceptance rate for this example
    acceptance_rate: float

    # Timestamp
    timestamp: str

    # Whether this is a failure case or replay case
    is_failure: bool = True

    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None

    # Token-level disagreements (optional, from manual speculative decoding)
    disagreements: Optional[List[TokenLevelDisagreement]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Handle nested dataclasses
        if self.disagreements is not None:
            result["disagreements"] = [d.to_dict() for d in self.disagreements]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingExample":
        """Create from dictionary."""
        # Handle nested disagreements
        if data.get("disagreements") is not None:
            data["disagreements"] = [
                TokenLevelDisagreement.from_dict(d) for d in data["disagreements"]
            ]
        return cls(**data)

    @property
    def has_detailed_data(self) -> bool:
        """Check if this example has token-level disagreement data."""
        return self.disagreements is not None and len(self.disagreements) > 0

    @property
    def high_confidence_failures(self) -> List[TokenLevelDisagreement]:
        """Get disagreements where draft was confident but wrong."""
        if not self.disagreements:
            return []
        return [d for d in self.disagreements if d.is_high_confidence_failure]


class DataCollector:
    """
    Collects failure cases and manages a replay buffer for training.

    The collector:
    1. Stores failure cases (low acceptance rate) for training
    2. Maintains a replay buffer of successful cases
    3. Provides batches mixing failure and replay cases
    4. Handles persistence to disk
    """

    def __init__(
        self,
        failures_dir: str = "data/failures",
        max_failure_cases: int = 100,
        replay_buffer_size: int = 25,
        replay_ratio: float = 0.2,
    ):
        """
        Initialize the data collector.

        Args:
            failures_dir: Directory to store failure cases
            max_failure_cases: Maximum failures before triggering training
            replay_buffer_size: Size of successful case buffer
            replay_ratio: Fraction of training batch from replay buffer
        """
        self.failures_dir = Path(failures_dir)
        self.failures_dir.mkdir(parents=True, exist_ok=True)

        self.max_failure_cases = max_failure_cases
        self.replay_buffer_size = replay_buffer_size
        self.replay_ratio = replay_ratio

        # In-memory storage
        self.failure_cases: List[TrainingExample] = []
        self.replay_buffer: List[TrainingExample] = []

        # Counter for unique IDs
        self._example_counter = 0

        # Load existing data
        self._load_existing_data()

    def _load_existing_data(self) -> None:
        """Load existing failure cases and replay buffer from disk."""
        failures_file = self.failures_dir / "failures.jsonl"
        replay_file = self.failures_dir / "replay_buffer.jsonl"

        if failures_file.exists():
            self.failure_cases = self._load_jsonl(failures_file)
            logger.info(f"Loaded {len(self.failure_cases)} existing failure cases")

        if replay_file.exists():
            self.replay_buffer = self._load_jsonl(replay_file)
            logger.info(f"Loaded {len(self.replay_buffer)} replay buffer cases")

        # Update counter
        all_examples = self.failure_cases + self.replay_buffer
        if all_examples:
            max_id = max(int(ex.id.split("_")[-1]) for ex in all_examples)
            self._example_counter = max_id + 1

    def _load_jsonl(self, filepath: Path) -> List[TrainingExample]:
        """Load examples from JSONL file."""
        examples = []
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append(TrainingExample.from_dict(data))
        return examples

    def _save_jsonl(self, examples: List[TrainingExample], filepath: Path) -> None:
        """Save examples to JSONL file."""
        with open(filepath, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict()) + "\n")

    def _generate_id(self) -> str:
        """Generate a unique example ID."""
        self._example_counter += 1
        return f"example_{self._example_counter}"

    def add_result(self, result: "SpeculativeResult") -> bool:
        """
        Add a speculative decoding result to the collector.

        Args:
            result: Result from speculative decoding

        Returns:
            True if training should be triggered
        """
        # Create training example
        example = TrainingExample(
            id=self._generate_id(),
            prompt=result.prompt,
            prompt_tokens=getattr(result, "prompt_tokens", None),
            draft_output=result.draft_outputs or result.tokens,
            target_output=result.target_outputs or result.tokens,
            acceptance_rate=result.metrics.acceptance_rate,
            timestamp=datetime.now().isoformat(),
            is_failure=result.is_failure_case,
            metadata={
                "total_tokens": result.metrics.total_tokens_generated,
                "tokens_per_second": result.metrics.tokens_per_second,
            },
        )

        if result.is_failure_case:
            # Add to failure cases
            self.failure_cases.append(example)
            logger.info(
                f"Added failure case (acceptance: {result.metrics.acceptance_rate:.2%}). "
                f"Total: {len(self.failure_cases)}/{self.max_failure_cases}"
            )
        else:
            # Add to replay buffer (circular buffer)
            self.replay_buffer.append(example)
            if len(self.replay_buffer) > self.replay_buffer_size:
                self.replay_buffer.pop(0)

        # Save to disk
        self._save_to_disk()

        # Check if training should be triggered
        return len(self.failure_cases) >= self.max_failure_cases

    def _save_to_disk(self) -> None:
        """Save current state to disk."""
        self._save_jsonl(self.failure_cases, self.failures_dir / "failures.jsonl")
        self._save_jsonl(self.replay_buffer, self.failures_dir / "replay_buffer.jsonl")

    def get_training_batch(
        self,
        batch_size: int,
        include_replay: bool = True,
    ) -> List[TrainingExample]:
        """
        Get a batch of training examples.

        Mixes failure cases with replay cases to prevent catastrophic forgetting.

        Args:
            batch_size: Number of examples to return
            include_replay: Whether to include replay buffer cases

        Returns:
            List of training examples
        """
        batch = []

        if include_replay and self.replay_buffer:
            # Calculate how many replay cases to include
            num_replay = int(batch_size * self.replay_ratio)
            num_replay = min(num_replay, len(self.replay_buffer))

            # Sample from replay buffer
            replay_samples = random.sample(self.replay_buffer, num_replay)
            batch.extend(replay_samples)

            # Fill rest with failure cases
            num_failures = batch_size - num_replay
        else:
            num_failures = batch_size

        # Sample from failure cases
        if self.failure_cases:
            num_failures = min(num_failures, len(self.failure_cases))
            failure_samples = random.sample(self.failure_cases, num_failures)
            batch.extend(failure_samples)

        # Shuffle the batch
        random.shuffle(batch)

        return batch

    def get_all_training_data(self) -> List[TrainingExample]:
        """Get all training data (failures + replay)."""
        all_data = self.failure_cases.copy()

        # Add replay buffer samples
        if self.replay_buffer:
            # Add proportional amount of replay data
            num_replay = int(len(self.failure_cases) * self.replay_ratio)
            num_replay = min(num_replay, len(self.replay_buffer))
            replay_samples = random.sample(self.replay_buffer, num_replay)
            all_data.extend(replay_samples)

        random.shuffle(all_data)
        return all_data

    def clear_failure_cases(self) -> None:
        """Clear failure cases after training (move some to replay)."""
        # Move high-acceptance failures to replay buffer for diversity
        high_acceptance = [
            ex
            for ex in self.failure_cases
            if ex.acceptance_rate > 0.3  # Somewhat successful
        ]

        # Add to replay buffer
        for ex in high_acceptance[: self.replay_buffer_size // 4]:
            ex.is_failure = False
            self.replay_buffer.append(ex)

        # Trim replay buffer
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer = self.replay_buffer[-self.replay_buffer_size :]

        # Clear failure cases
        self.failure_cases = []

        # Save to disk
        self._save_to_disk()

        logger.info("Cleared failure cases after training")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about collected data."""
        stats = {
            "num_failure_cases": len(self.failure_cases),
            "num_replay_cases": len(self.replay_buffer),
            "max_failure_cases": self.max_failure_cases,
            "ready_for_training": len(self.failure_cases) >= self.max_failure_cases,
        }

        if self.failure_cases:
            acceptance_rates = [ex.acceptance_rate for ex in self.failure_cases]
            stats["avg_failure_acceptance"] = sum(acceptance_rates) / len(
                acceptance_rates
            )
            stats["min_failure_acceptance"] = min(acceptance_rates)
            stats["max_failure_acceptance"] = max(acceptance_rates)

        if self.replay_buffer:
            acceptance_rates = [ex.acceptance_rate for ex in self.replay_buffer]
            stats["avg_replay_acceptance"] = sum(acceptance_rates) / len(
                acceptance_rates
            )

        return stats

    def export_for_training(self, output_path: str) -> str:
        """
        Export training data in a format suitable for mlx-lm fine-tuning.

        Args:
            output_path: Path to save the training data

        Returns:
            Path to the exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        training_data = self.get_all_training_data()

        # Format for instruction tuning
        formatted_data = []
        for ex in training_data:
            item = {
                "prompt": ex.prompt,
                "draft_tokens": ex.draft_output,
                "target_tokens": ex.target_output,
                "acceptance_rate": ex.acceptance_rate,
            }

            # Include token-level disagreement data if available
            if ex.has_detailed_data:
                # Get prompt_length from metadata if available
                prompt_length = (
                    ex.metadata.get("prompt_length", 0) if ex.metadata else 0
                )

                # Export disagreements with relative positions (positions in generated sequence)
                # Convert absolute positions to relative for training purposes
                disagreements_export = []
                for d in ex.disagreements:
                    d_dict = d.to_dict()
                    # Store both absolute and relative positions for flexibility
                    d_dict["absolute_position"] = d.position
                    d_dict["relative_position"] = d.position - prompt_length
                    disagreements_export.append(d_dict)

                item["disagreements"] = disagreements_export
                # Use relative positions for training (positions in generated sequence)
                item["disagreement_positions"] = [
                    d.position - prompt_length for d in ex.disagreements
                ]
                item["high_confidence_failure_positions"] = [
                    d.position - prompt_length for d in ex.high_confidence_failures
                ]

            formatted_data.append(item)

        with open(output_path, "w") as f:
            for item in formatted_data:
                f.write(json.dumps(item) + "\n")

        logger.info(f"Exported {len(formatted_data)} examples to {output_path}")

        # Log stats about detailed data
        detailed_count = sum(1 for ex in training_data if ex.has_detailed_data)
        if detailed_count > 0:
            total_disagreements = sum(
                len(ex.disagreements) for ex in training_data if ex.has_detailed_data
            )
            logger.info(
                f"  {detailed_count} examples with token-level data, "
                f"{total_disagreements} total disagreements"
            )

        return str(output_path)

    def add_detailed_result(
        self,
        prompt: str,
        generated_tokens: List[int],
        disagreements: List[TokenLevelDisagreement],
        acceptance_rate: float,
        metadata: Optional[Dict[str, Any]] = None,
        prompt_length: Optional[int] = None,
        prompt_tokens: Optional[List[int]] = None,
    ) -> bool:
        """
        Add a result from manual speculative decoding with token-level details.

        This method is specifically for results from ManualSpeculativeDecoder
        which provides detailed disagreement information.

        Args:
            prompt: The input prompt
            generated_tokens: All tokens generated (final output, without prompt)
            disagreements: List of token-level disagreements (positions are absolute, including prompt)
            acceptance_rate: Overall acceptance rate
            metadata: Optional additional metadata
            prompt_length: Length of the tokenized prompt (required if disagreements exist)

        Returns:
            True if training should be triggered
        """
        # Reconstruct draft output from disagreements
        # Start with final output (target tokens), then replace disagreement positions
        # with what the draft model actually predicted
        draft_output = generated_tokens.copy()

        # Convert absolute positions to relative positions (subtract prompt length)
        # Disagreement positions include prompt tokens, but generated_tokens doesn't
        if disagreements and prompt_length is None:
            logger.warning(
                "prompt_length is required when disagreements exist. "
                "Cannot reconstruct draft output accurately."
            )

        for disagreement in disagreements:
            # Convert absolute position to relative position in generated sequence
            relative_position = disagreement.position - (prompt_length or 0)

            if relative_position < 0 or relative_position >= len(draft_output):
                logger.warning(
                    f"Disagreement at absolute position {disagreement.position} "
                    f"(relative: {relative_position}) out of bounds "
                    f"for generated sequence length {len(generated_tokens)}. Skipping."
                )
                continue
            draft_output[relative_position] = disagreement.draft_token

        # Create training example with detailed data
        # Store prompt_length in metadata for later use in export
        # Create a copy to avoid mutating the caller's dictionary
        metadata_with_prompt_length = dict(metadata) if metadata else {}
        if prompt_length is not None:
            metadata_with_prompt_length["prompt_length"] = prompt_length

        example = TrainingExample(
            id=self._generate_id(),
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            draft_output=draft_output,  # Draft's original predictions
            target_output=generated_tokens,  # Final output (target's choices)
            acceptance_rate=acceptance_rate,
            timestamp=datetime.now().isoformat(),
            is_failure=acceptance_rate < 0.5,  # Consider low acceptance as failure
            metadata=metadata_with_prompt_length,
            disagreements=disagreements,
        )

        # Always treat examples with disagreements as valuable training data
        if disagreements:
            self.failure_cases.append(example)
            logger.info(
                f"Added detailed example with {len(disagreements)} disagreements "
                f"(acceptance: {acceptance_rate:.2%}). "
                f"Total: {len(self.failure_cases)}/{self.max_failure_cases}"
            )
        elif acceptance_rate >= 0.5:
            # High acceptance with no disagreements -> replay buffer
            self.replay_buffer.append(example)
            if len(self.replay_buffer) > self.replay_buffer_size:
                self.replay_buffer.pop(0)
        else:
            # Low acceptance but no detailed disagreements
            self.failure_cases.append(example)
            logger.info(
                f"Added failure case (acceptance: {acceptance_rate:.2%}). "
                f"Total: {len(self.failure_cases)}/{self.max_failure_cases}"
            )

        # Save to disk
        self._save_to_disk()

        # Check if training should be triggered
        return len(self.failure_cases) >= self.max_failure_cases


class AcceptanceRateTracker:
    """
    Tracks acceptance rates over time for monitoring and evaluation.

    Maintains a sliding window of recent acceptance rates and
    provides statistics for monitoring training progress.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize the tracker.

        Args:
            window_size: Size of sliding window for recent rates
        """
        self.window_size = window_size
        self.all_rates: List[float] = []
        self.timestamps: List[str] = []

        # Track by prompt type (optional categorization)
        self.rates_by_category: Dict[str, List[float]] = {}

    def add_rate(
        self,
        rate: float,
        category: Optional[str] = None,
    ) -> None:
        """
        Add an acceptance rate observation.

        Args:
            rate: Acceptance rate (0-1)
            category: Optional category for the prompt
        """
        self.all_rates.append(rate)
        self.timestamps.append(datetime.now().isoformat())

        if category:
            if category not in self.rates_by_category:
                self.rates_by_category[category] = []
            self.rates_by_category[category].append(rate)

    def get_recent_average(self) -> float:
        """Get average acceptance rate over recent window."""
        if not self.all_rates:
            return 0.0
        recent = self.all_rates[-self.window_size :]
        return sum(recent) / len(recent)

    def get_overall_average(self) -> float:
        """Get overall average acceptance rate."""
        if not self.all_rates:
            return 0.0
        return sum(self.all_rates) / len(self.all_rates)

    def get_trend(self, window: int = 50) -> float:
        """
        Calculate trend in acceptance rates.

        Returns:
            Positive value = improving, negative = declining
        """
        if len(self.all_rates) < window * 2:
            return 0.0

        older = self.all_rates[-window * 2 : -window]
        newer = self.all_rates[-window:]

        older_avg = sum(older) / len(older)
        newer_avg = sum(newer) / len(newer)

        return newer_avg - older_avg

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            "total_observations": len(self.all_rates),
            "recent_average": self.get_recent_average(),
            "overall_average": self.get_overall_average(),
            "trend": self.get_trend(),
        }

        if self.all_rates:
            stats["min_rate"] = min(self.all_rates)
            stats["max_rate"] = max(self.all_rates)

            # Percentiles
            sorted_rates = sorted(self.all_rates)
            n = len(sorted_rates)
            stats["p25"] = sorted_rates[n // 4]
            stats["p50"] = sorted_rates[n // 2]
            stats["p75"] = sorted_rates[3 * n // 4]

        # Category stats
        if self.rates_by_category:
            stats["by_category"] = {
                cat: sum(rates) / len(rates)
                for cat, rates in self.rates_by_category.items()
                if rates
            }

        return stats

    def save(self, filepath: str) -> None:
        """Save tracking data to file."""
        data = {
            "rates": self.all_rates,
            "timestamps": self.timestamps,
            "by_category": self.rates_by_category,
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    def load(self, filepath: str) -> None:
        """Load tracking data from file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        self.all_rates = data.get("rates", [])
        self.timestamps = data.get("timestamps", [])
        self.rates_by_category = data.get("by_category", {})
