"""Configuration management for PixelRNN training.

This module provides configuration loading and management using OmegaConf.
"""

from typing import Dict, Any, Optional
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import yaml


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    return config


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        OmegaConf.save(config, f)


def get_default_config() -> DictConfig:
    """Get default configuration for PixelRNN training.
    
    Returns:
        Default configuration
    """
    config = OmegaConf.create({
        # Model configuration
        "model": {
            "in_channels": 3,
            "hidden_size": 128,
            "num_layers": 2,
            "num_mixtures": 5,
            "image_size": 32,
        },
        
        # Data configuration
        "data": {
            "dataset_name": "cifar10",
            "data_dir": "./data",
            "batch_size": 64,
            "num_workers": 4,
            "image_size": 32,
            "augment": True,
        },
        
        # Training configuration
        "training": {
            "num_epochs": 100,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "optimizer": "adam",
            "grad_clip": 1.0,
            "sample_temperature": 1.0,
            "sample_interval": 10,
            "checkpoint_interval": 10,
            "checkpoint_dir": "./checkpoints",
            "sample_dir": "./samples",
        },
        
        # Evaluation configuration
        "evaluation": {
            "num_samples": 1000,
            "compute_fid": True,
            "compute_is": True,
            "compute_lpips": True,
        },
        
        # Logging configuration
        "logging": {
            "use_wandb": False,
            "wandb_project": "pixelrnn",
            "experiment_name": "default",
            "log_interval": 100,
        },
        
        # System configuration
        "system": {
            "device": "auto",  # auto, cuda, mps, cpu
            "seed": 42,
            "deterministic": True,
        },
    })
    
    return config


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """Merge two configurations.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override with
        
    Returns:
        Merged configuration
    """
    return OmegaConf.merge(base_config, override_config)


def create_config_from_dict(config_dict: Dict[str, Any]) -> DictConfig:
    """Create configuration from dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Configuration object
    """
    return OmegaConf.create(config_dict)


def validate_config(config: DictConfig) -> None:
    """Validate configuration.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate model config
    if config.model.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    
    if config.model.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    
    if config.model.num_mixtures <= 0:
        raise ValueError("num_mixtures must be positive")
    
    if config.model.image_size <= 0:
        raise ValueError("image_size must be positive")
    
    # Validate data config
    if config.data.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    
    if config.data.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    
    # Validate training config
    if config.training.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    
    if config.training.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    
    if config.training.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    
    # Validate optimizer
    valid_optimizers = ["adam", "adamw", "sgd"]
    if config.training.optimizer.lower() not in valid_optimizers:
        raise ValueError(f"optimizer must be one of {valid_optimizers}")
    
    # Validate device
    valid_devices = ["auto", "cuda", "mps", "cpu"]
    if config.system.device.lower() not in valid_devices:
        raise ValueError(f"device must be one of {valid_devices}")


def get_experiment_config(
    experiment_name: str,
    dataset_name: str = "cifar10",
    image_size: int = 32,
    hidden_size: int = 128,
    num_layers: int = 2,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    use_wandb: bool = False,
) -> DictConfig:
    """Get configuration for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        dataset_name: Name of dataset to use
        image_size: Size of images
        hidden_size: Size of hidden state
        num_layers: Number of LSTM layers
        batch_size: Batch size for training
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        use_wandb: Whether to use wandb logging
        
    Returns:
        Experiment configuration
    """
    config = get_default_config()
    
    # Override with experiment-specific values
    config.model.hidden_size = hidden_size
    config.model.num_layers = num_layers
    config.model.image_size = image_size
    
    config.data.dataset_name = dataset_name
    config.data.batch_size = batch_size
    config.data.image_size = image_size
    
    config.training.num_epochs = num_epochs
    config.training.learning_rate = learning_rate
    
    config.logging.use_wandb = use_wandb
    config.logging.experiment_name = experiment_name
    
    return config
