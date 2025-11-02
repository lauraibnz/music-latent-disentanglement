import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    """Simple 1D convolution block with residual connection and FiLM conditioning."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, num_groups=8, cond_channels=0):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               stride=1, padding=kernel_size // 2)
        
        # GroupNorm replaces BatchNorm - ensure num_groups divides channels
        def get_valid_groups(channels, max_groups):
            for groups in range(min(max_groups, channels), 0, -1):
                if channels % groups == 0:
                    return groups
            return 1  # Fallback to 1 group
        
        self.norm1 = nn.GroupNorm(num_groups=get_valid_groups(in_channels, num_groups), num_channels=in_channels)
        self.norm2 = nn.GroupNorm(num_groups=get_valid_groups(out_channels, num_groups), num_channels=out_channels)
        
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(0.1)
        
        # FiLM conditioning for global embeddings
        self.cond_channels = cond_channels
        if cond_channels > 0:
            # Project conditioning to gamma (scale) and beta (shift) for each conv layer
            self.film1 = nn.Linear(cond_channels, out_channels * 2)  # *2 for gamma and beta
            self.film2 = nn.Linear(cond_channels, out_channels * 2)
            
            # Initialize FiLM layers to identity transformation (Stable Diffusion trick)
            nn.init.zeros_(self.film1.weight)
            nn.init.zeros_(self.film1.bias)
            nn.init.zeros_(self.film2.weight)
            nn.init.zeros_(self.film2.bias)
            # Set beta (shift) initial values to small positive for first layer
            self.film1.bias.data[:out_channels] = 1.0  # gamma starts at 1
            self.film2.bias.data[:out_channels] = 1.0  # gamma starts at 1
        
        # Projection for residual connection if channels differ
        self.use_residual = (in_channels == out_channels and stride == 1)
        if not self.use_residual:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, 1, stride=stride)
    
    def forward(self, x, cond=None):
        residual = x
        
        # Main path - First conv + norm + FiLM
        out = self.norm1(x)
        out = self.activation(out)
        out = self.conv1(out)
        
        # Apply FiLM conditioning after first conv if available
        if cond is not None and self.cond_channels > 0:
            film_params = self.film1(cond)  # [B, out_channels * 2]
            gamma, beta = film_params.chunk(2, dim=1)  # [B, out_channels] each
            out = gamma.unsqueeze(-1) * out + beta.unsqueeze(-1)  # Broadcast over time
        
        # Second conv + norm + FiLM  
        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        
        # Apply FiLM conditioning after second conv if available
        if cond is not None and self.cond_channels > 0:
            film_params = self.film2(cond)  # [B, out_channels * 2]
            gamma, beta = film_params.chunk(2, dim=1)  # [B, out_channels] each
            out = gamma.unsqueeze(-1) * out + beta.unsqueeze(-1)  # Broadcast over time
        
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
        
        # Main convolution block with FiLM conditioning
        self.conv_block = ConvBlock1D(total_in_channels, out_channels, kernel_size, stride, cond_channels=cond_channels)
        
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
        
        # Apply main convolution with FiLM conditioning
        x = self.conv_block(x, cond=cond)
        
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