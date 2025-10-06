import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    """Simple 1D convolution block with residual connection."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                              stride=1, padding=kernel_size//2)
        
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(0.1)
        
        # Projection for residual connection if channels differ
        self.use_residual = (in_channels == out_channels and stride == 1)
        if not self.use_residual:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, 1, stride=stride)
    
    def forward(self, x):
        residual = x
        
        # Main path
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        
        # Residual connection
        if self.use_residual:
            out = out + residual
        else:
            residual = self.residual_proj(residual)
            out = out + residual
            
        return out


class UNetBlock1D(nn.Module):
    """UNet-style block with conditioning support for diffusion models."""
    
    def __init__(self, in_channels, out_channels, cond_channels=0, time_cond_channels=0, 
                 kernel_size=3, stride=1, upsample=False):
        super().__init__()
        
        self.upsample = upsample
        self.cond_channels = cond_channels
        self.time_cond_channels = time_cond_channels
        
        # Adjust input channels for conditioning
        total_in_channels = in_channels + time_cond_channels
        
        # Main convolution block
        self.conv_block = ConvBlock1D(total_in_channels, out_channels, kernel_size, stride)
        
        # Global conditioning projection (for timbre embedding)
        if cond_channels > 0:
            self.cond_proj = nn.Linear(cond_channels, out_channels)
        
        # Upsampling layer
        if upsample:
            self.upsample_layer = nn.ConvTranspose1d(out_channels, out_channels, 
                                                   kernel_size=4, stride=2, padding=1)
    
    def forward(self, x, cond=None, time_cond=None):
        """
        Args:
            x: Input tensor [B, C, T]
            cond: Global conditioning (timbre) [B, cond_channels]
            time_cond: Time-varying conditioning (structure) [B, time_cond_channels, T]
        """
        # Concatenate time-varying conditioning
        if time_cond is not None:
            # Ensure same temporal dimension
            if time_cond.shape[-1] != x.shape[-1]:
                time_cond = nn.functional.interpolate(time_cond, size=x.shape[-1], mode='linear')
            x = torch.cat([x, time_cond], dim=1)
        
        # Apply main convolution
        x = self.conv_block(x)
        
        # Add global conditioning
        if cond is not None:
            cond_proj = self.cond_proj(cond)  # [B, out_channels]
            x = x + cond_proj.unsqueeze(-1)   # Broadcast over time dimension
        
        # Upsample if needed
        if self.upsample:
            x = self.upsample_layer(x)
            
        return x


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps):
        """
        Args:
            timesteps: [B] or [B, 1]
        Returns:
            embeddings: [B, dim]
        """
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings