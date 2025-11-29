"""Tests for PixelRNN model."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.pixelrnn import PixelRNN
from data.dataset import DataModule, set_seed
from config.config import get_default_config


class TestPixelRNN:
    """Test cases for PixelRNN model."""
    
    def test_model_creation(self):
        """Test model creation."""
        model = PixelRNN(
            in_channels=3,
            hidden_size=64,
            num_layers=2,
            num_mixtures=5,
            image_size=32,
        )
        
        assert model.in_channels == 3
        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert model.num_mixtures == 5
        assert model.image_size == 32
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = PixelRNN(
            in_channels=3,
            hidden_size=64,
            num_layers=2,
            num_mixtures=5,
            image_size=32,
        )
        
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)
        
        output = model(x)
        
        assert output.shape == (batch_size, 32 * 32, model.output_size)
    
    def test_generate_samples(self):
        """Test sample generation."""
        model = PixelRNN(
            in_channels=3,
            hidden_size=64,
            num_layers=2,
            num_mixtures=5,
            image_size=32,
        )
        
        batch_size = 4
        samples = model.generate(batch_size=batch_size)
        
        assert samples.shape == (batch_size, 3, 32, 32)
        assert samples.min() >= -1.0
        assert samples.max() <= 1.0
    
    def test_compute_loss(self):
        """Test loss computation."""
        model = PixelRNN(
            in_channels=3,
            hidden_size=64,
            num_layers=2,
            num_mixtures=5,
            image_size=32,
        )
        
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)
        
        loss = model.compute_loss(x)
        
        assert loss.item() >= 0
        assert torch.isfinite(loss)
    
    def test_different_temperatures(self):
        """Test generation with different temperatures."""
        model = PixelRNN(
            in_channels=3,
            hidden_size=64,
            num_layers=2,
            num_mixtures=5,
            image_size=32,
        )
        
        batch_size = 2
        
        # Generate with different temperatures
        samples_low = model.generate(batch_size=batch_size, temperature=0.5)
        samples_high = model.generate(batch_size=batch_size, temperature=2.0)
        
        assert samples_low.shape == (batch_size, 3, 32, 32)
        assert samples_high.shape == (batch_size, 3, 32, 32)
    
    def test_seed_reproducibility(self):
        """Test that generation is reproducible with same seed."""
        model = PixelRNN(
            in_channels=3,
            hidden_size=64,
            num_layers=2,
            num_mixtures=5,
            image_size=32,
        )
        
        batch_size = 2
        seed = 42
        
        # Generate samples with same seed
        set_seed(seed)
        samples1 = model.generate(batch_size=batch_size, seed=seed)
        
        set_seed(seed)
        samples2 = model.generate(batch_size=batch_size, seed=seed)
        
        # Should be identical
        assert torch.allclose(samples1, samples2, atol=1e-6)


class TestDataModule:
    """Test cases for DataModule."""
    
    def test_data_module_creation(self):
        """Test data module creation."""
        data_module = DataModule(
            dataset_name="cifar10",
            data_dir="./test_data",
            batch_size=32,
            num_workers=0,  # Use 0 for testing
            image_size=32,
            augment=False,  # Disable augmentation for testing
        )
        
        assert data_module.dataset_name == "cifar10"
        assert data_module.batch_size == 32
        assert data_module.image_size == 32
    
    def test_toy_dataset_creation(self):
        """Test toy dataset creation."""
        from data.dataset import create_toy_dataset
        
        num_samples = 100
        image_size = 32
        num_classes = 10
        
        images, labels = create_toy_dataset(
            num_samples=num_samples,
            image_size=image_size,
            num_classes=num_classes,
        )
        
        assert images.shape == (num_samples, 3, image_size, image_size)
        assert labels.shape == (num_samples,)
        assert labels.min() >= 0
        assert labels.max() < num_classes


class TestConfig:
    """Test cases for configuration management."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = get_default_config()
        
        assert config.model.hidden_size > 0
        assert config.model.num_layers > 0
        assert config.data.batch_size > 0
        assert config.training.num_epochs > 0
    
    def test_config_validation(self):
        """Test configuration validation."""
        from config.config import validate_config, create_config_from_dict
        
        # Valid config
        valid_config = create_config_from_dict({
            "model": {"hidden_size": 128, "num_layers": 2, "num_mixtures": 5, "image_size": 32},
            "data": {"batch_size": 64, "num_workers": 4, "image_size": 32},
            "training": {"num_epochs": 100, "learning_rate": 0.001, "weight_decay": 0.0001, "optimizer": "adam"},
            "system": {"device": "auto", "seed": 42},
        })
        
        # Should not raise exception
        validate_config(valid_config)
        
        # Invalid config
        invalid_config = create_config_from_dict({
            "model": {"hidden_size": -1},  # Invalid
            "data": {"batch_size": 64},
            "training": {"num_epochs": 100, "learning_rate": 0.001, "weight_decay": 0.0001, "optimizer": "adam"},
            "system": {"device": "auto", "seed": 42},
        })
        
        # Should raise exception
        with pytest.raises(ValueError):
            validate_config(invalid_config)


if __name__ == "__main__":
    pytest.main([__file__])
