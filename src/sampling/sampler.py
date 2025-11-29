"""Sampling utilities for PixelRNN model.

This module provides utilities for generating samples from the trained PixelRNN model
with various sampling strategies and controls.
"""

from typing import Optional, Tuple, List, Union
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from PIL import Image
import io

from ..models.pixelrnn import PixelRNN
from ..config.config import load_config, get_default_config
from ..data.dataset import get_device, set_seed


class PixelRNNSampler:
    """Sampler class for PixelRNN model."""
    
    def __init__(
        self,
        model: PixelRNN,
        device: Optional[torch.device] = None,
    ):
        """Initialize sampler.
        
        Args:
            model: Trained PixelRNN model
            device: Device to generate on
        """
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        self.model.eval()
    
    def generate_samples(
        self,
        num_samples: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate samples from the model.
        
        Args:
            num_samples: Number of samples to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k most likely tokens
            top_p: Keep tokens with cumulative probability <= top_p
            seed: Random seed for reproducibility
            
        Returns:
            Generated samples of shape (num_samples, channels, height, width)
        """
        if seed is not None:
            set_seed(seed)
        
        with torch.no_grad():
            samples = self.model.generate(
                batch_size=num_samples,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=self.device,
            )
        
        return samples
    
    def generate_interpolation(
        self,
        num_steps: int = 10,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate interpolation between random samples.
        
        Args:
            num_steps: Number of interpolation steps
            temperature: Sampling temperature
            seed: Random seed for reproducibility
            
        Returns:
            Interpolated samples
        """
        if seed is not None:
            set_seed(seed)
        
        # Generate two random samples
        sample1 = self.generate_samples(1, temperature, seed=seed)
        sample2 = self.generate_samples(1, temperature, seed=seed + 1 if seed else None)
        
        # Create interpolation
        interpolations = []
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            interpolated = (1 - alpha) * sample1 + alpha * sample2
            interpolations.append(interpolated)
        
        return torch.cat(interpolations, dim=0)
    
    def generate_conditional_samples(
        self,
        condition: torch.Tensor,
        num_samples: int = 64,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate samples conditioned on partial image.
        
        Args:
            condition: Partial image to condition on
            num_samples: Number of samples to generate
            temperature: Sampling temperature
            seed: Random seed for reproducibility
            
        Returns:
            Generated samples
        """
        if seed is not None:
            set_seed(seed)
        
        # This is a simplified implementation
        # In practice, you'd need to modify the model to support conditioning
        with torch.no_grad():
            samples = self.generate_samples(
                num_samples=num_samples,
                temperature=temperature,
                seed=seed,
            )
        
        return samples
    
    def save_samples(
        self,
        samples: torch.Tensor,
        save_path: Union[str, Path],
        nrow: int = 8,
        title: Optional[str] = None,
    ) -> None:
        """Save samples as image grid.
        
        Args:
            samples: Samples to save
            save_path: Path to save to
            nrow: Number of images per row
            title: Title for the plot
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Denormalize samples
        samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
        samples = torch.clamp(samples, 0, 1)
        
        # Create grid
        from torchvision.utils import make_grid
        grid = make_grid(samples, nrow=nrow, normalize=False)
        
        # Convert to numpy
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        
        # Create plot
        plt.figure(figsize=(12, 12))
        plt.imshow(grid_np)
        plt.axis('off')
        
        if title:
            plt.title(title, fontsize=16)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_individual_samples(
        self,
        samples: torch.Tensor,
        save_dir: Union[str, Path],
        prefix: str = "sample",
    ) -> None:
        """Save individual samples as separate images.
        
        Args:
            samples: Samples to save
            save_dir: Directory to save to
            prefix: Prefix for filenames
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Denormalize samples
        samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
        samples = torch.clamp(samples, 0, 1)
        
        for i, sample in enumerate(samples):
            # Convert to PIL Image
            sample_np = sample.permute(1, 2, 0).cpu().numpy()
            sample_np = (sample_np * 255).astype(np.uint8)
            
            image = Image.fromarray(sample_np)
            image.save(save_dir / f"{prefix}_{i:04d}.png")
    
    def generate_and_save(
        self,
        num_samples: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        save_path: Optional[Union[str, Path]] = None,
        nrow: int = 8,
    ) -> torch.Tensor:
        """Generate samples and save them.
        
        Args:
            num_samples: Number of samples to generate
            temperature: Sampling temperature
            top_k: Keep only top k most likely tokens
            top_p: Keep tokens with cumulative probability <= top_p
            seed: Random seed for reproducibility
            save_path: Path to save samples
            nrow: Number of images per row
            
        Returns:
            Generated samples
        """
        samples = self.generate_samples(
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
        
        if save_path:
            self.save_samples(samples, save_path, nrow=nrow)
        
        return samples


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    config: Optional[dict] = None,
    device: Optional[torch.device] = None,
) -> Tuple[PixelRNN, dict]:
    """Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Model configuration (if None, loaded from checkpoint)
        device: Device to load model on
        
    Returns:
        Tuple of (model, config)
    """
    checkpoint_path = Path(checkpoint_path)
    device = device or get_device()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    if config is None:
        config = checkpoint.get("config", {})
    
    # Create model
    model_config = config.get("model", {})
    model = PixelRNN(**model_config)
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, config


def main():
    """Main function for sampling script."""
    parser = argparse.ArgumentParser(description="Generate samples from PixelRNN model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--num_samples", type=int, default=64, help="Number of samples to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, help="Top-p sampling")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--output", type=str, default="./samples.png", help="Output path")
    parser.add_argument("--nrow", type=int, default=8, help="Number of images per row")
    parser.add_argument("--individual", action="store_true", help="Save individual samples")
    parser.add_argument("--interpolation", action="store_true", help="Generate interpolation")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of interpolation steps")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # Load model
    model, model_config = load_model_from_checkpoint(args.checkpoint, config)
    
    # Create sampler
    sampler = PixelRNNSampler(model)
    
    # Generate samples
    if args.interpolation:
        samples = sampler.generate_interpolation(
            num_steps=args.num_steps,
            temperature=args.temperature,
            seed=args.seed,
        )
        output_path = Path(args.output).parent / "interpolation.png"
    else:
        samples = sampler.generate_samples(
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        )
        output_path = args.output
    
    # Save samples
    if args.individual:
        sampler.save_individual_samples(samples, Path(args.output).parent)
    
    sampler.save_samples(samples, output_path, nrow=args.nrow)
    
    print(f"Generated {len(samples)} samples")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
