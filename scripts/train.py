#!/usr/bin/env python3
"""Main training script for PixelRNN model.

This script provides a command-line interface for training PixelRNN models
with various configurations and datasets.
"""

import argparse
import sys
from pathlib import Path
import torch
import wandb

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.pixelrnn import PixelRNN
from data.dataset import DataModule, get_device, set_seed
from training.trainer import PixelRNNTrainer
from config.config import (
    load_config, 
    get_default_config, 
    validate_config,
    get_experiment_config,
)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PixelRNN model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--experiment", type=str, help="Experiment name")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--image_size", type=int, default=32, help="Image size")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb", action="store_true", help="Use wandb logging")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate model")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
        
        # Override with command line arguments
        if args.experiment:
            config.logging.experiment_name = args.experiment
        
        config.data.dataset_name = args.dataset
        config.data.image_size = args.image_size
        config.data.batch_size = args.batch_size
        
        config.model.hidden_size = args.hidden_size
        config.model.num_layers = args.num_layers
        config.model.image_size = args.image_size
        
        config.training.num_epochs = args.num_epochs
        config.training.learning_rate = args.learning_rate
        
        config.system.device = args.device
        config.system.seed = args.seed
        config.logging.use_wandb = args.wandb
    
    # Validate configuration
    validate_config(config)
    
    # Set random seed
    set_seed(config.system.seed)
    
    # Get device
    if config.system.device == "auto":
        device = get_device()
    else:
        device = torch.device(config.system.device)
    
    print(f"Using device: {device}")
    print(f"Configuration: {config}")
    
    # Setup data module
    data_module = DataModule(
        dataset_name=config.data.dataset_name,
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        image_size=config.data.image_size,
        augment=config.data.augment,
    )
    data_module.setup()
    
    # Print dataset info
    dataset_info = data_module.get_dataset_info()
    print(f"Dataset info: {dataset_info}")
    
    # Create model
    model = PixelRNN(
        in_channels=config.model.in_channels,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        num_mixtures=config.model.num_mixtures,
        image_size=config.model.image_size,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = PixelRNNTrainer(
        model=model,
        data_module=data_module,
        config=config,
        device=device,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            trainer.load_checkpoint(checkpoint_path)
            print(f"Resumed from checkpoint: {checkpoint_path}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
    
    # Evaluate only mode
    if args.eval_only:
        val_loader = data_module.val_dataloader()
        metrics = trainer.validate(val_loader)
        print(f"Evaluation metrics: {metrics}")
        
        # Generate samples
        trainer.save_samples(num_samples=64)
        print("Generated samples saved")
        
        return
    
    # Train model
    try:
        trainer.train(config.training.num_epochs)
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        trainer.save_checkpoint()
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        trainer.save_checkpoint()
        raise


if __name__ == "__main__":
    main()
