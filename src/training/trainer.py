"""Training module for PixelRNN model.

This module provides training utilities, loss functions, and evaluation metrics
for the PixelRNN model.
"""

from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from pathlib import Path
import wandb
from torchmetrics.image import FrechetInceptionDistance, InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import lpips
from clean_fid import fid

from ..models.pixelrnn import PixelRNN
from ..data.dataset import DataModule, get_device, set_seed


class PixelRNNTrainer:
    """Trainer class for PixelRNN model."""
    
    def __init__(
        self,
        model: PixelRNN,
        data_module: DataModule,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        """Initialize trainer.
        
        Args:
            model: PixelRNN model to train
            data_module: Data module for loading data
            config: Training configuration
            device: Device to train on
        """
        self.model = model
        self.data_module = data_module
        self.config = config
        self.device = device or get_device()
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup metrics
        self._setup_metrics()
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Create output directories
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "./checkpoints"))
        self.sample_dir = Path(config.get("sample_dir", "./samples"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        optimizer_name = self.config.get("optimizer", "adam").lower()
        lr = self.config.get("learning_rate", 1e-3)
        weight_decay = self.config.get("weight_decay", 1e-4)
        
        if optimizer_name == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
            )
        elif optimizer_name == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
            )
        elif optimizer_name == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _setup_metrics(self) -> None:
        """Setup evaluation metrics."""
        self.fid_metric = FrechetInceptionDistance(feature=2048, normalize=True)
        self.is_metric = InceptionScore(normalize=True)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex')
        
        # Move metrics to device
        self.fid_metric.to(self.device)
        self.is_metric.to(self.device)
        self.lpips_metric.to(self.device)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            loss = self.model.compute_loss(images)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config["grad_clip"]
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log to wandb
            if self.config.get("use_wandb", False):
                wandb.log({
                    "train/loss": loss.item(),
                    "train/epoch": self.current_epoch,
                    "train/batch": batch_idx,
                })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, _ in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                
                # Compute loss
                loss = self.model.compute_loss(images)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        metrics = {"val_loss": avg_loss}
        
        # Generate samples for evaluation
        if num_batches > 0:
            sample_metrics = self._evaluate_samples(val_loader)
            metrics.update(sample_metrics)
        
        return metrics
    
    def _evaluate_samples(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate generated samples.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of sample evaluation metrics
        """
        # Generate samples
        num_samples = min(1000, len(val_loader.dataset))
        batch_size = val_loader.batch_size
        
        generated_images = []
        real_images = []
        
        with torch.no_grad():
            # Generate samples
            for _ in range(num_samples // batch_size + 1):
                samples = self.model.generate(
                    batch_size=batch_size,
                    temperature=self.config.get("sample_temperature", 1.0),
                    device=self.device,
                )
                generated_images.append(samples.cpu())
            
            # Get real samples
            for images, _ in val_loader:
                real_images.append(images)
                if len(real_images) * batch_size >= num_samples:
                    break
        
        # Concatenate and limit
        generated_images = torch.cat(generated_images, dim=0)[:num_samples]
        real_images = torch.cat(real_images, dim=0)[:num_samples]
        
        # Move to device for metrics
        generated_images = generated_images.to(self.device)
        real_images = real_images.to(self.device)
        
        # Compute metrics
        metrics = {}
        
        try:
            # FID
            self.fid_metric.update(real_images, real=True)
            self.fid_metric.update(generated_images, real=False)
            metrics["fid"] = self.fid_metric.compute().item()
            self.fid_metric.reset()
            
            # IS
            self.is_metric.update(generated_images)
            is_mean, is_std = self.is_metric.compute()
            metrics["inception_score"] = is_mean.item()
            metrics["inception_score_std"] = is_std.item()
            self.is_metric.reset()
            
        except Exception as e:
            print(f"Error computing metrics: {e}")
            metrics["fid"] = float('inf')
            metrics["inception_score"] = 0.0
        
        return metrics
    
    def save_samples(self, num_samples: int = 64) -> None:
        """Save generated samples.
        
        Args:
            num_samples: Number of samples to generate
        """
        self.model.eval()
        
        with torch.no_grad():
            samples = self.model.generate(
                batch_size=num_samples,
                temperature=self.config.get("sample_temperature", 1.0),
                device=self.device,
            )
            
            # Save samples
            sample_path = self.sample_dir / f"samples_epoch_{self.current_epoch}.png"
            self._save_image_grid(samples, sample_path, nrow=8)
            
            # Log to wandb
            if self.config.get("use_wandb", False):
                wandb.log({
                    "samples": wandb.Image(str(sample_path)),
                    "epoch": self.current_epoch,
                })
    
    def _save_image_grid(
        self, 
        images: torch.Tensor, 
        path: Path, 
        nrow: int = 8
    ) -> None:
        """Save images as a grid.
        
        Args:
            images: Images to save
            path: Path to save to
            nrow: Number of images per row
        """
        from torchvision.utils import make_grid
        
        # Denormalize images
        images = (images + 1) / 2  # [-1, 1] -> [0, 1]
        images = torch.clamp(images, 0, 1)
        
        # Create grid
        grid = make_grid(images, nrow=nrow, normalize=False)
        
        # Convert to numpy and save
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        plt.figure(figsize=(12, 12))
        plt.imshow(grid_np)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            is_best: Whether this is the best checkpoint
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "config": self.config,
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs: int) -> None:
        """Train the model.
        
        Args:
            num_epochs: Number of epochs to train
        """
        # Setup data loaders
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()
        
        # Initialize wandb if enabled
        if self.config.get("use_wandb", False):
            wandb.init(
                project=self.config.get("wandb_project", "pixelrnn"),
                config=self.config,
                name=f"pixelrnn_{self.config.get('experiment_name', 'default')}",
            )
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Save samples
            if epoch % self.config.get("sample_interval", 10) == 0:
                self.save_samples()
            
            # Save checkpoint
            is_best = val_metrics["val_loss"] < self.best_loss
            if is_best:
                self.best_loss = val_metrics["val_loss"]
            
            if epoch % self.config.get("checkpoint_interval", 10) == 0:
                self.save_checkpoint(is_best)
            
            # Log metrics
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['val_loss']:.4f}")
            if "fid" in val_metrics:
                print(f"FID: {val_metrics['fid']:.2f}, IS: {val_metrics['inception_score']:.2f}")
            
            # Log to wandb
            if self.config.get("use_wandb", False):
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_metrics["val_loss"],
                    **{f"val/{k}": v for k, v in val_metrics.items() if k != "val_loss"},
                })
        
        # Final checkpoint
        self.save_checkpoint(is_best)
        
        if self.config.get("use_wandb", False):
            wandb.finish()
        
        print("Training completed!")
