"""Streamlit demo app for PixelRNN model.

This module provides an interactive web interface for generating samples
from the trained PixelRNN model.
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.pixelrnn import PixelRNN
from sampling.sampler import PixelRNNSampler, load_model_from_checkpoint
from config.config import load_config, get_default_config
from data.dataset import get_device, set_seed


@st.cache_resource
def load_model(checkpoint_path: str, config_path: str = None):
    """Load model from checkpoint with caching."""
    try:
        if config_path:
            config = load_config(config_path)
        else:
            config = get_default_config()
        
        model, model_config = load_model_from_checkpoint(checkpoint_path, config)
        sampler = PixelRNNSampler(model)
        return sampler, model_config
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="PixelRNN Demo",
        page_icon="🎨",
        layout="wide",
    )
    
    st.title("🎨 PixelRNN Image Generation Demo")
    st.markdown("Generate images pixel by pixel using autoregressive recurrent neural networks")
    
    # Sidebar for controls
    st.sidebar.header("Model Configuration")
    
    # Model selection
    checkpoint_path = st.sidebar.text_input(
        "Checkpoint Path",
        value="./checkpoints/best.pth",
        help="Path to the trained model checkpoint"
    )
    
    config_path = st.sidebar.text_input(
        "Config Path (optional)",
        value="",
        help="Path to the model configuration file"
    )
    
    # Load model
    if st.sidebar.button("Load Model"):
        with st.spinner("Loading model..."):
            sampler, config = load_model(checkpoint_path, config_path if config_path else None)
            if sampler is not None:
                st.sidebar.success("Model loaded successfully!")
                st.session_state.sampler = sampler
                st.session_state.config = config
            else:
                st.sidebar.error("Failed to load model")
    
    # Check if model is loaded
    if "sampler" not in st.session_state:
        st.warning("Please load a model first using the sidebar controls.")
        return
    
    sampler = st.session_state.sampler
    config = st.session_state.config
    
    # Generation parameters
    st.sidebar.header("Generation Parameters")
    
    num_samples = st.sidebar.slider(
        "Number of Samples",
        min_value=1,
        max_value=100,
        value=16,
        help="Number of images to generate"
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Sampling temperature (higher = more random)"
    )
    
    top_k = st.sidebar.slider(
        "Top-K",
        min_value=1,
        max_value=100,
        value=50,
        help="Keep only top K most likely tokens"
    )
    
    top_p = st.sidebar.slider(
        "Top-P",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.1,
        help="Keep tokens with cumulative probability <= top_p"
    )
    
    seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=1000000,
        value=42,
        help="Random seed for reproducibility"
    )
    
    # Generation buttons
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        generate_samples = st.button("🎨 Generate", type="primary")
    
    with col2:
        generate_interpolation = st.button("🔄 Interpolate")
    
    with col3:
        clear_samples = st.button("🗑️ Clear")
    
    # Main content area
    if generate_samples:
        with st.spinner("Generating samples..."):
            # Set seed
            set_seed(seed)
            
            # Generate samples
            samples = sampler.generate_samples(
                num_samples=num_samples,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed,
            )
            
            # Store in session state
            st.session_state.samples = samples
            st.session_state.generation_params = {
                "num_samples": num_samples,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "seed": seed,
            }
    
    if generate_interpolation:
        with st.spinner("Generating interpolation..."):
            # Set seed
            set_seed(seed)
            
            # Generate interpolation
            num_steps = st.sidebar.slider("Interpolation Steps", 5, 20, 10)
            samples = sampler.generate_interpolation(
                num_steps=num_steps,
                temperature=temperature,
                seed=seed,
            )
            
            # Store in session state
            st.session_state.samples = samples
            st.session_state.generation_params = {
                "type": "interpolation",
                "num_steps": num_steps,
                "temperature": temperature,
                "seed": seed,
            }
    
    if clear_samples:
        if "samples" in st.session_state:
            del st.session_state.samples
        if "generation_params" in st.session_state:
            del st.session_state.generation_params
        st.rerun()
    
    # Display samples
    if "samples" in st.session_state:
        samples = st.session_state.samples
        params = st.session_state.generation_params
        
        st.header("Generated Samples")
        
        # Display generation parameters
        st.subheader("Generation Parameters")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Samples", len(samples))
        
        with col2:
            st.metric("Temperature", params.get("temperature", "N/A"))
        
        with col3:
            st.metric("Top-K", params.get("top_k", "N/A"))
        
        with col4:
            st.metric("Seed", params.get("seed", "N/A"))
        
        # Display samples
        st.subheader("Samples")
        
        # Convert to PIL Images
        images = []
        for i, sample in enumerate(samples):
            # Denormalize
            sample = (sample + 1) / 2  # [-1, 1] -> [0, 1]
            sample = torch.clamp(sample, 0, 1)
            
            # Convert to numpy
            sample_np = sample.permute(1, 2, 0).cpu().numpy()
            sample_np = (sample_np * 255).astype(np.uint8)
            
            # Convert to PIL
            image = Image.fromarray(sample_np)
            images.append(image)
        
        # Display in grid
        nrow = 4
        ncol = len(images) // nrow + (1 if len(images) % nrow > 0 else 0)
        
        for row in range(ncol):
            cols = st.columns(nrow)
            for col_idx, col in enumerate(cols):
                img_idx = row * nrow + col_idx
                if img_idx < len(images):
                    col.image(images[img_idx], caption=f"Sample {img_idx + 1}")
        
        # Download button
        st.subheader("Download")
        
        # Create a zip file with all samples
        import zipfile
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w') as zip_file:
                for i, image in enumerate(images):
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format='PNG')
                    zip_file.writestr(f"sample_{i:04d}.png", img_bytes.getvalue())
            
            with open(tmp_file.name, 'rb') as f:
                st.download_button(
                    label="📥 Download All Samples",
                    data=f.read(),
                    file_name="pixelrnn_samples.zip",
                    mime="application/zip"
                )
    
    # Model information
    st.sidebar.header("Model Information")
    
    if "config" in st.session_state:
        config = st.session_state.config
        
        st.sidebar.text(f"Dataset: {config.get('data', {}).get('dataset_name', 'N/A')}")
        st.sidebar.text(f"Image Size: {config.get('model', {}).get('image_size', 'N/A')}")
        st.sidebar.text(f"Hidden Size: {config.get('model', {}).get('hidden_size', 'N/A')}")
        st.sidebar.text(f"LSTM Layers: {config.get('model', {}).get('num_layers', 'N/A')}")
        st.sidebar.text(f"Mixtures: {config.get('model', {}).get('num_mixtures', 'N/A')}")
    
    # Instructions
    st.sidebar.header("Instructions")
    st.sidebar.markdown("""
    1. **Load Model**: Enter the path to your trained model checkpoint
    2. **Configure**: Adjust generation parameters in the sidebar
    3. **Generate**: Click the generate button to create samples
    4. **Download**: Use the download button to save all samples
    
    **Tips**:
    - Lower temperature = more conservative samples
    - Higher temperature = more creative/diverse samples
    - Use the same seed for reproducible results
    """)


if __name__ == "__main__":
    main()
