"""Experiment directory management for organized result storage."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class ExperimentDirectoryManager:
    """Manages structured directory layout for benchmark experiments.

    Creates and manages the following directory structure:
        results/
        └── preprint/
            └── {experiment_name}/
                ├── config.json
                ├── by_task/
                │   ├── humaneval_instruct_local/
                │   │   ├── q0.00/
                │   │   │   ├── samples.jsonl
                │   │   │   ├── metrics.json
                │   │   │   └── rank_logs/
                │   │   ├── q0.50/
                │   │   └── ...
                │   ├── gsm8k_cot/
                │   └── ifeval/
                └── aggregated/
                    ├── all_metrics.csv
                    └── summary.json

    Example:
        >>> dirs = ExperimentDirectoryManager("/path/to/results")
        >>> dirs.setup()
        >>> task_dir = dirs.get_task_quantile_dir("gsm8k_cot", 0.99)
        >>> print(task_dir)
        /path/to/results/preprint/{timestamp}_benchmark/by_task/gsm8k_cot/q0.99
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        experiment_name: Optional[str] = None,
        submission_name: str = "preprint"
    ):
        """Initialize experiment directory manager.

        Args:
            base_dir: Base results directory
            experiment_name: Name for this experiment run (default: timestamp)
            submission_name: Submission identifier (default: preprint)
        """
        self.base_dir = Path(base_dir)
        self.submission_name = submission_name

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S_benchmark")
        self.experiment_name = experiment_name

        # Main paths
        self.root = self.base_dir / submission_name / experiment_name
        self.by_task = self.root / "by_task"
        self.aggregated = self.root / "aggregated"

    def setup(self) -> None:
        """Create all base directories."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.by_task.mkdir(exist_ok=True)
        self.aggregated.mkdir(exist_ok=True)

    def get_task_quantile_dir(self, task: str, quantile: float) -> Path:
        """Get directory for a specific task/quantile combination.

        Args:
            task: Task name (e.g., "gsm8k_cot")
            quantile: EOS quantile value (e.g., 0.99)

        Returns:
            Path to the task/quantile directory
        """
        q_str = f"q{quantile:.2f}"
        return self.by_task / task / q_str

    def setup_task_quantile(self, task: str, quantile: float) -> Path:
        """Create directories for a task/quantile run.

        Args:
            task: Task name
            quantile: EOS quantile value

        Returns:
            Path to the task/quantile directory
        """
        task_dir = self.get_task_quantile_dir(task, quantile)
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "rank_logs").mkdir(exist_ok=True)
        return task_dir

    def get_rank_log_dir(self, task: str, quantile: float) -> Path:
        """Get the rank_logs directory for a task/quantile.

        Args:
            task: Task name
            quantile: EOS quantile value

        Returns:
            Path to the rank_logs directory
        """
        return self.get_task_quantile_dir(task, quantile) / "rank_logs"

    def get_samples_path(self, task: str, quantile: float) -> Path:
        """Get path for merged samples JSONL file.

        Args:
            task: Task name
            quantile: EOS quantile value

        Returns:
            Path to samples.jsonl
        """
        return self.get_task_quantile_dir(task, quantile) / "samples.jsonl"

    def get_metrics_path(self, task: str, quantile: float) -> Path:
        """Get path for aggregated metrics JSON file.

        Args:
            task: Task name
            quantile: EOS quantile value

        Returns:
            Path to metrics.json
        """
        return self.get_task_quantile_dir(task, quantile) / "metrics.json"

    def save_config(self, config: dict) -> None:
        """Save experiment configuration to config.json.

        Args:
            config: Configuration dictionary to save
        """
        config_path = self.root / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def save_summary(self, summary: dict) -> None:
        """Save experiment summary to aggregated/summary.json.

        Args:
            summary: Summary dictionary to save
        """
        summary_path = self.aggregated / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def get_all_task_dirs(self) -> list[Path]:
        """Get all task directories that exist.

        Returns:
            List of paths to task directories
        """
        if not self.by_task.exists():
            return []
        return [d for d in self.by_task.iterdir() if d.is_dir()]

    def get_all_quantile_dirs(self, task: str) -> list[Path]:
        """Get all quantile directories for a task.

        Args:
            task: Task name

        Returns:
            List of paths to quantile directories
        """
        task_dir = self.by_task / task
        if not task_dir.exists():
            return []
        return sorted([d for d in task_dir.iterdir() if d.is_dir()])

    def __repr__(self) -> str:
        return f"ExperimentDirectoryManager(root={self.root})"
