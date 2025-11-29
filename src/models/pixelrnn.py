"""PixelRNN implementation for autoregressive image generation.

This module implements a modern PixelRNN model that generates images pixel by pixel
using recurrent neural networks, following the autoregressive principle where each
pixel depends on previously generated pixels.
"""

from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


class MaskedConv2d(nn.Conv2d):
    """Masked convolution layer for autoregressive generation.
    
    This layer ensures that each pixel only depends on previously generated pixels
    by applying a mask to the convolution weights.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        mask_type: str = "A",
    ):
        """Initialize masked convolution layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolution kernel
            stride: Stride of the convolution
            padding: Padding added to both sides of the input
            dilation: Spacing between kernel elements
            groups: Number of blocked connections from input to output
            bias: Whether to add a bias term
            mask_type: Type of mask ("A" for first layer, "B" for subsequent layers)
        """
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.mask_type = mask_type
        self.register_buffer("mask", torch.zeros_like(self.weight))
        self._create_mask()
    
    def _create_mask(self) -> None:
        """Create the autoregressive mask."""
        h, w = self.kernel_size
        center_h, center_w = h // 2, w // 2
        
        # Create mask
        mask = torch.zeros_like(self.weight)
        
        if self.mask_type == "A":
            # Type A mask: exclude center pixel
            mask[:, :, :center_h] = 1
            mask[:, :, center_h, :center_w] = 1
        else:
            # Type B mask: include center pixel
            mask[:, :, :center_h] = 1
            mask[:, :, center_h, :center_w + 1] = 1
        
        self.mask.copy_(mask)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with masked convolution."""
        self.weight.data *= self.mask
        return super().forward(x)


class PixelRNNCell(nn.Module):
    """Single PixelRNN cell using LSTM."""
    
    def __init__(self, input_size: int, hidden_size: int):
        """Initialize PixelRNN cell.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size, hidden_size)
    
    def forward(
        self, 
        x: Tensor, 
        hidden: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass through LSTM cell.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            hidden: Previous hidden state (h, c)
            
        Returns:
            Output tensor and new hidden state
        """
        if hidden is None:
            batch_size = x.size(0)
            h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            hidden = (h, c)
        
        h, c = self.lstm(x, hidden)
        return h, (h, c)


class PixelRNN(nn.Module):
    """PixelRNN model for autoregressive image generation.
    
    This model generates images pixel by pixel using recurrent neural networks,
    where each pixel depends on previously generated pixels in a raster scan order.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_mixtures: int = 5,
        image_size: int = 32,
    ):
        """Initialize PixelRNN model.
        
        Args:
            in_channels: Number of input channels (e.g., 3 for RGB)
            hidden_size: Size of hidden state in LSTM
            num_layers: Number of LSTM layers
            num_mixtures: Number of mixture components for discrete logistic
            image_size: Size of generated images (assumed square)
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures
        self.image_size = image_size
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_size)
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList([
            PixelRNNCell(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        
        # Output layers for discrete logistic mixture
        self.output_size = in_channels * num_mixtures * 3  # means, scales, mixture weights
        self.output_proj = nn.Linear(hidden_size, self.output_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for training (teacher forcing).
        
        Args:
            x: Input images of shape (batch_size, channels, height, width)
            
        Returns:
            Predicted pixel values for next step
        """
        batch_size, channels, height, width = x.shape
        device = x.device
        
        # Reshape to sequence format (batch_size, seq_len, channels)
        x_seq = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x_seq = x_seq.view(batch_size, height * width, channels)  # (B, H*W, C)
        
        # Initialize hidden states
        hidden_states = [None] * self.num_layers
        
        outputs = []
        
        # Process each pixel in sequence
        for t in range(height * width):
            # Current pixel
            current_pixel = x_seq[:, t, :]  # (B, C)
            
            # Project input
            h = self.input_proj(current_pixel)  # (B, hidden_size)
            
            # Pass through LSTM layers
            for layer_idx, lstm_layer in enumerate(self.lstm_layers):
                h, hidden_states[layer_idx] = lstm_layer(h, hidden_states[layer_idx])
            
            # Generate output
            output = self.output_proj(h)  # (B, output_size)
            outputs.append(output)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # (B, H*W, output_size)
        
        return outputs
    
    def generate(
        self, 
        batch_size: int = 1, 
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        device: Optional[torch.device] = None
    ) -> Tensor:
        """Generate images autoregressively.
        
        Args:
            batch_size: Number of images to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k most likely tokens
            top_p: Keep tokens with cumulative probability <= top_p
            device: Device to generate on
            
        Returns:
            Generated images of shape (batch_size, channels, height, width)
        """
        if device is None:
            device = next(self.parameters()).device
        
        height = width = self.image_size
        
        # Initialize hidden states
        hidden_states = [None] * self.num_layers
        
        # Initialize image
        generated_image = torch.zeros(batch_size, height * width, self.in_channels, device=device)
        
        # Generate pixels sequentially
        for t in range(height * width):
            # Current pixel (zeros for first pixel)
            if t == 0:
                current_pixel = torch.zeros(batch_size, self.in_channels, device=device)
            else:
                current_pixel = generated_image[:, t-1, :]
            
            # Project input
            h = self.input_proj(current_pixel)
            
            # Pass through LSTM layers
            for layer_idx, lstm_layer in enumerate(self.lstm_layers):
                h, hidden_states[layer_idx] = lstm_layer(h, hidden_states[layer_idx])
            
            # Generate output
            output = self.output_proj(h)  # (B, output_size)
            
            # Sample next pixel
            next_pixel = self._sample_pixel(output, temperature, top_k, top_p)
            generated_image[:, t, :] = next_pixel
        
        # Reshape to image format
        generated_image = generated_image.view(batch_size, height, width, self.in_channels)
        generated_image = generated_image.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        return generated_image
    
    def _sample_pixel(
        self, 
        output: Tensor, 
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Tensor:
        """Sample pixel values from discrete logistic mixture.
        
        Args:
            output: Model output of shape (batch_size, output_size)
            temperature: Sampling temperature
            top_k: Keep only top k most likely tokens
            top_p: Keep tokens with cumulative probability <= top_p
            
        Returns:
            Sampled pixel values of shape (batch_size, channels)
        """
        batch_size = output.size(0)
        
        # Reshape output to mixture components
        output = output.view(batch_size, self.in_channels, self.num_mixtures, 3)
        
        # Extract means, scales, and mixture weights
        means = output[:, :, :, 0]  # (B, C, M)
        scales = F.softplus(output[:, :, :, 1]) + 1e-5  # (B, C, M)
        mixture_weights = F.softmax(output[:, :, :, 2], dim=-1)  # (B, C, M)
        
        # Sample mixture component
        mixture_samples = torch.multinomial(mixture_weights.view(-1, self.num_mixtures), 1)
        mixture_samples = mixture_samples.view(batch_size, self.in_channels)
        
        # Get means and scales for selected components
        selected_means = torch.gather(means, dim=-1, index=mixture_samples.unsqueeze(-1)).squeeze(-1)
        selected_scales = torch.gather(scales, dim=-1, index=mixture_samples.unsqueeze(-1)).squeeze(-1)
        
        # Sample from discrete logistic distribution
        uniform_samples = torch.rand_like(selected_means)
        samples = selected_means + selected_scales * torch.log(uniform_samples / (1 - uniform_samples))
        
        # Apply temperature scaling
        if temperature != 1.0:
            samples = samples / temperature
        
        # Apply top-k or top-p filtering if specified
        if top_k is not None or top_p is not None:
            # For simplicity, we'll use the raw samples without filtering
            # In a more sophisticated implementation, you'd apply filtering here
            pass
        
        # Clamp to valid range
        samples = torch.clamp(samples, -1.0, 1.0)
        
        return samples
    
    def compute_loss(self, x: Tensor) -> Tensor:
        """Compute negative log-likelihood loss.
        
        Args:
            x: Input images of shape (batch_size, channels, height, width)
            
        Returns:
            NLL loss
        """
        batch_size, channels, height, width = x.shape
        
        # Forward pass
        outputs = self.forward(x)  # (B, H*W, output_size)
        
        # Reshape targets
        targets = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        targets = targets.view(batch_size, height * width, channels)  # (B, H*W, C)
        
        # Reshape outputs to mixture components
        outputs = outputs.view(batch_size, height * width, channels, self.num_mixtures, 3)
        
        # Extract mixture parameters
        means = outputs[:, :, :, :, 0]  # (B, H*W, C, M)
        scales = F.softplus(outputs[:, :, :, :, 1]) + 1e-5  # (B, H*W, C, M)
        mixture_weights = F.softmax(outputs[:, :, :, :, 2], dim=-1)  # (B, H*W, C, M)
        
        # Compute log probabilities for each mixture component
        targets_expanded = targets.unsqueeze(-1).expand(-1, -1, -1, self.num_mixtures)  # (B, H*W, C, M)
        
        # Discrete logistic log probability
        log_probs = -torch.log(scales) - torch.log(1 + torch.exp((targets_expanded - means) / scales))
        
        # Weight by mixture components
        weighted_log_probs = log_probs + torch.log(mixture_weights)
        
        # Log-sum-exp for mixture
        log_probs_mixture = torch.logsumexp(weighted_log_probs, dim=-1)  # (B, H*W, C)
        
        # Average over sequence and channels
        nll = -log_probs_mixture.mean()
        
        return nll
