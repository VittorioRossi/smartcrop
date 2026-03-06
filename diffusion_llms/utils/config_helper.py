"""
Clean configuration management for LLaDA training.
Handles YAML config loading and validation.
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Clean configuration class for LLaDA training."""

    # Metadata
    debug: bool

    # Model settings
    model_name: str
    hidden_size: int
    context_length: int
    model_type: str

    # Data settings
    cache_dir: str
    embedding_dir: str
    num_workers: int
    val_split: float
    test_split: float

    # Training settings
    batch_size: int
    learning_rate: float
    epochs: int
    max_steps: int
    accumulate_grad_batches: int
    val_check_interval: int
    seed: int

    # Checkpoint settings
    checkpoint_dir: str
    save_top_k: int
    monitor: str
    mode: str
    save_every_n_steps: int

    # Early stopping settings
    patience: int
    min_delta: float
    early_stopping_monitor: str
    early_stopping_mode: str

    # Logging settings
    project_name: str
    run_name: str
    log_dir: str
    use_wandb: bool
    log_every_n_steps: int

    # Output settings
    model_dir: str
    results_dir: str
    predictions_file: str

    # Optional
    pos_weight: Optional[float] = None

    @classmethod
    def from_yaml(self, path_to_yaml):
        config = load_config(path_to_yaml)
        return config_to_dataclass(config)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train LLaDA model")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/llada_classifier.yml",
        help="Path to configuration YAML file",
    )
    parser.add_argument("--run-name", type=str, help="Override run name")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r") as file:
        try:
            config = yaml.safe_load(file)
            print(f"✓ Configuration loaded from {config_path}")
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")


def create_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories from config."""
    directories = [
        config["checkpoint"]["dir"],
        config["logging"]["log_dir"],
        config["output"]["model_dir"],
        config["output"]["results_dir"],
        config["data"]["cache_dir"],
        config["data"]["embedding_dir"],
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration values."""
    required_sections = [
        "model",
        "data",
        "training",
        "checkpoint",
        "early_stopping",
        "logging",
        "output",
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate model type
    valid_types = ["classifier", "regressor"]
    if config["model"]["type"] not in valid_types:
        raise ValueError(
            f"Invalid model type: {config['model']['type']}. Must be one of {valid_types}"
        )

    # Validate splits
    total_split = config["data"]["val_split"] + config["data"]["test_split"]
    if total_split >= 1.0:
        raise ValueError(f"Val split + test split ({total_split}) must be < 1.0")

    print("✓ Configuration validation passed")


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply command-line overrides to configuration."""
    if args.run_name:
        config["logging"]["run_name"] = args.run_name
        print(f"✓ Override run_name: {args.run_name}")

    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
        print(f"✓ Override batch_size: {args.batch_size}")

    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
        print(f"✓ Override learning_rate: {args.learning_rate}")

    if args.epochs:
        config["training"]["epochs"] = args.epochs
        print(f"✓ Override epochs: {args.epochs}")

    return config


def config_to_dataclass(config: Dict[str, Any]) -> TrainingConfig:
    """Convert config dict to TrainingConfig dataclass. Returns None if required keys are missing."""
    try:
        return TrainingConfig(
            # Metadata
            debug=config.get("metadata", {}).get("debug"),
            # Model settings
            model_name=config.get("model", {}).get("name"),
            hidden_size=config.get("model", {}).get("hidden_size"),
            context_length=config.get("model", {}).get("context_length"),
            model_type=config.get("model", {}).get("type"),
            pos_weight=config.get("model", {}).get("pos_weight"),
            # Data settings
            cache_dir=config.get("data", {}).get("cache_dir"),
            embedding_dir=config.get("data", {}).get("embedding_dir"),
            num_workers=config.get("data", {}).get("num_workers"),
            val_split=config.get("data", {}).get("val_split"),
            test_split=config.get("data", {}).get("test_split"),
            # Training settings
            batch_size=config.get("training", {}).get("batch_size"),
            learning_rate=config.get("training", {}).get("learning_rate"),
            epochs=config.get("training", {}).get("epochs"),
            max_steps=config.get("training", {}).get("max_steps"),
            accumulate_grad_batches=config.get("training", {}).get(
                "accumulate_grad_batches"
            ),
            val_check_interval=config.get("training", {}).get("val_check_interval"),
            seed=config.get("training", {}).get("seed"),
            # Checkpoint settings
            checkpoint_dir=config.get("checkpoint", {}).get("dir"),
            save_top_k=config.get("checkpoint", {}).get("save_top_k"),
            monitor=config.get("checkpoint", {}).get("monitor"),
            mode=config.get("checkpoint", {}).get("mode"),
            save_every_n_steps=config.get("checkpoint", {}).get("save_every_n_steps"),
            # Early stopping settings
            patience=config.get("early_stopping", {}).get("patience"),
            min_delta=config.get("early_stopping", {}).get("min_delta"),
            early_stopping_monitor=config.get("early_stopping", {}).get("monitor"),
            early_stopping_mode=config.get("early_stopping", {}).get("mode"),
            # Logging settings
            project_name=config.get("logging", {}).get("project_name"),
            run_name=config.get("logging", {}).get("run_name"),
            log_dir=config.get("logging", {}).get("log_dir"),
            use_wandb=config.get("logging", {}).get("use_wandb"),
            log_every_n_steps=config.get("logging", {}).get("log_every_n_steps"),
            # Output settings
            model_dir=config.get("output", {}).get("model_dir"),
            results_dir=config.get("output", {}).get("results_dir"),
            predictions_file=config.get("output", {}).get("predictions_file"),
        )
    except Exception as e:
        print(f"Error converting config to dataclass: {e}")
        return None


def load_training_config() -> TrainingConfig:
    """Main function to load and validate training configuration."""
    print("Loading training configuration...")

    # Parse arguments
    args = parse_args()

    # Load config file
    config = load_config(args.config)

    # Validate configuration
    validate_config(config)

    # Apply command-line overrides
    config = apply_overrides(config, args)

    # Create necessary directories
    create_directories(config)

    # Convert to dataclass
    training_config = config_to_dataclass(config)

    print("Configuration loaded successfully!")
    print(f" Model: {training_config.model_type}")
    print(f" Run: {training_config.run_name}")
    print(f" Batch size: {training_config.batch_size}")
    print(f" Learning rate: {training_config.learning_rate}")

    return training_config


# Legacy function for backward compatibility
def get_config() -> Dict[str, Any]:
    """Legacy function - returns dict for backward compatibility."""
    config = load_training_config()

    # Convert back to flat dict for legacy code
    return {
        "model_name": config.model_name,
        "hidden_size": config.hidden_size,
        "context_length": config.context_length,
        "model_type": config.model_type,
        "pos_weight": config.pos_weight or 0.277,
        "cache_dir": config.cache_dir,
        "embedding_dir": config.embedding_dir,
        "num_workers": config.num_workers,
        "val_test_perc": config.val_split,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "n_epochs": config.epochs,
        "max_steps": config.max_steps,
        "accumulate_grad": config.accumulate_grad_batches,
        "val_check_interval": config.val_check_interval,
        "seed": config.seed,
        "checkpoint_dir": config.checkpoint_dir,
        "patience": config.patience,
        "min_delta": config.min_delta,
        "monitor": config.early_stopping_monitor,
        "mode": config.early_stopping_mode,
        "run_name": config.run_name,
        "project_name": config.project_name,
        "log_dir": config.log_dir,
        "output_dir": config.model_dir,
        "model_to_train": {"type": config.model_type, "output_dir": config.model_dir},
    }
