"""
1D Convolutional Denoising Autoencoder (CDAE) for Raman spectroscopy.
Architecture based on user specifications with encoder-decoder structure.
"""

import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CDAE_Raman(nn.Module):
    """
    1D Convolutional Denoising Autoencoder for Raman spectra.
    
    Architecture:
    Encoder:
        - Conv1D: 32 filters, kernel=7, ReLU
        - MaxPooling1D: pool_size=2
        - Conv1D: 64 filters, kernel=5, ReLU
        - MaxPooling1D: pool_size=2
        - Flatten
        - Dense: 64 neurons (latent space)
    
    Decoder:
        - Dense: (calculated size for reshape)
        - Reshape to 3D
        - UpSampling1D: size=2
        - Conv1D: 32 filters, kernel=5, ReLU, padding=same
        - UpSampling1D: size=2
        - Conv1D: 1 filter, kernel=7, Linear, padding=same
    """
    
    def __init__(self, input_length: int = 2361, latent_dim: int = 64):
        """
        Initialize the CDAE model.
        
        Args:
            input_length: Length of input Raman spectrum (number of wavenumbers)
            latent_dim: Dimension of the latent space bottleneck
        """
        super(CDAE_Raman, self).__init__()
        
        self.input_length = input_length
        self.latent_dim = latent_dim
        
        # Calculate sizes after each pooling layer
        # After Conv1 (kernel=7, padding=3): input_length
        # After MaxPool1 (pool=2): input_length // 2
        # After Conv2 (kernel=5, padding=2): input_length // 2
        # After MaxPool2 (pool=2): input_length // 4
        
        self.encoder_conv1_output = input_length
        self.encoder_pool1_output = input_length // 2
        self.encoder_conv2_output = self.encoder_pool1_output
        self.encoder_pool2_output = self.encoder_pool1_output // 2
        
        # Flattened size before latent layer
        self.flatten_size = 64 * self.encoder_pool2_output
        
        # Decoder reshape dimensions
        self.decoder_channels = 64
        self.decoder_initial_length = self.encoder_pool2_output
        
        logger.info(f"Building CDAE with input_length={input_length}, latent_dim={latent_dim}")
        logger.info(f"Encoder progression: {input_length} -> {self.encoder_pool1_output} -> {self.encoder_pool2_output}")
        logger.info(f"Flatten size: {self.flatten_size}")
        
        # ============ ENCODER ============
        self.encoder = nn.Sequential(
            # Conv1D: 32 filters, kernel=7, ReLU
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=7,
                padding=3,  # 'same' padding
                stride=1
            ),
            nn.ReLU(),
            
            # MaxPooling1D: pool_size=2
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Conv1D: 64 filters, kernel=5, ReLU
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2,  # 'same' padding
                stride=1
            ),
            nn.ReLU(),
            
            # MaxPooling1D: pool_size=2
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        
        # Latent space (bottleneck)
        self.encoder_fc = nn.Linear(self.flatten_size, latent_dim)
        
        # ============ DECODER ============
        # Dense layer to expand from latent space
        self.decoder_fc = nn.Linear(latent_dim, self.flatten_size)
        
        self.decoder = nn.Sequential(
            # UpSampling1D: size=2
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # Conv1D: 32 filters, kernel=5, ReLU, padding=same
            nn.Conv1d(
                in_channels=64,
                out_channels=32,
                kernel_size=5,
                padding=2,  # 'same' padding
                stride=1
            ),
            nn.ReLU(),
            
            # UpSampling1D: size=2
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # Conv1D (Output): 1 filter, kernel=7, Linear, padding=same
            nn.Conv1d(
                in_channels=32,
                out_channels=1,
                kernel_size=7,
                padding=3,  # 'same' padding
                stride=1
            ),
            # No activation (Linear output)
        )
        
        # Handle potential size mismatch due to rounding in pooling/upsampling
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (batch_size, input_length)
        
        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        # Reshape to (batch_size, 1, input_length) for Conv1D
        x = x.unsqueeze(1)
        
        # Pass through encoder convolutions
        x = self.encoder(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Latent space
        latent = self.encoder_fc(x)
        
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstructed spectrum.
        
        Args:
            latent: Latent tensor of shape (batch_size, latent_dim)
        
        Returns:
            Reconstructed spectrum of shape (batch_size, input_length)
        """
        # Expand from latent space
        x = self.decoder_fc(latent)
        
        # Reshape for decoder
        x = x.view(x.size(0), self.decoder_channels, self.decoder_initial_length)
        
        # Pass through decoder
        x = self.decoder(x)
        
        # Remove channel dimension and adjust size if needed
        x = x.squeeze(1)
        
        # Handle size mismatch (crop or pad to match input_length)
        if x.size(1) > self.input_length:
            x = x[:, :self.input_length]
        elif x.size(1) < self.input_length:
            padding = self.input_length - x.size(1)
            x = nn.functional.pad(x, (0, padding))
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_length)
        
        Returns:
            Reconstructed spectrum of shape (batch_size, input_length)
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the latent representation of input data.
        
        Args:
            x: Input tensor of shape (batch_size, input_length)
        
        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        with torch.no_grad():
            return self.encode(x)


def get_model_summary(model: nn.Module, input_size: tuple) -> str:
    """
    Get a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch_size, input_length)
    
    Returns:
        Summary string
    """
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = []
    summary.append("=" * 80)
    summary.append("Model Architecture Summary")
    summary.append("=" * 80)
    summary.append(f"Total parameters: {count_parameters(model):,}")
    summary.append(f"Input size: {input_size}")
    summary.append(f"Latent dimension: {model.latent_dim}")
    summary.append("=" * 80)
    
    # Test forward pass to verify architecture
    device = next(model.parameters()).device
    test_input = torch.randn(input_size).to(device)
    
    try:
        with torch.no_grad():
            output = model(test_input)
            latent = model.encode(test_input)
        summary.append(f"Output size: {tuple(output.shape)}")
        summary.append(f"Latent size: {tuple(latent.shape)}")
        summary.append("✓ Architecture verified successfully")
    except Exception as e:
        summary.append(f"✗ Architecture verification failed: {str(e)}")
    
    summary.append("=" * 80)
    return "\n".join(summary)


if __name__ == "__main__":
    # Test the model
    input_length = 2361  # Number of wavenumbers in Raman spectrum
    batch_size = 4
    latent_dim = 64
    
    # Create model
    model = CDAE_Raman(input_length=input_length, latent_dim=latent_dim)
    
    # Print summary
    print(get_model_summary(model, (batch_size, input_length)))
    
    # Test forward pass
    test_input = torch.randn(batch_size, input_length)
    output = model(test_input)
    latent = model.get_latent_representation(test_input)
    
    print(f"\nTest forward pass:")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Input/Output match: {test_input.shape == output.shape}")
