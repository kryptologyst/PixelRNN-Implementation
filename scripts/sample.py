#!/usr/bin/env python3
"""Sampling script for PixelRNN model.

This script provides a command-line interface for generating samples from
trained PixelRNN models.
"""

import argparse
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.pixelrnn import PixelRNN
from sampling.sampler import PixelRNNSampler, load_model_from_checkpoint
from config.config import load_config, get_default_config
from data.dataset import get_device, set_seed


def main():
    """Main sampling function."""
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
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Get device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model, model_config = load_model_from_checkpoint(args.checkpoint, config, device)
    
    # Create sampler
    sampler = PixelRNNSampler(model, device)
    
    # Generate samples
    if args.interpolation:
        print(f"Generating interpolation with {args.num_steps} steps...")
        samples = sampler.generate_interpolation(
            num_steps=args.num_steps,
            temperature=args.temperature,
            seed=args.seed,
        )
        output_path = Path(args.output).parent / "interpolation.png"
    else:
        print(f"Generating {args.num_samples} samples...")
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
        individual_dir = Path(args.output).parent / "individual_samples"
        sampler.save_individual_samples(samples, individual_dir)
        print(f"Individual samples saved to: {individual_dir}")
    
    sampler.save_samples(samples, output_path, nrow=args.nrow)
    print(f"Generated {len(samples)} samples")
    print(f"Saved to: {output_path}")
    
    # Print sample statistics
    print(f"Sample statistics:")
    print(f"  Mean: {samples.mean().item():.4f}")
    print(f"  Std: {samples.std().item():.4f}")
    print(f"  Min: {samples.min().item():.4f}")
    print(f"  Max: {samples.max().item():.4f}")


if __name__ == "__main__":
    main()
