import torch
import torch.nn as nn


def get_valid_groups(channels, max_groups):
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock1D(nn.Module):
    """Simple 1D convolution block with residual connection and FiLM conditioning."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, num_groups=8, cond_channels=0):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               stride=1, padding=kernel_size // 2)
        
        # GroupNorm replaces BatchNorm - ensure num_groups divides channels
        self.norm1 = nn.GroupNorm(num_groups=get_valid_groups(in_channels, num_groups), num_channels=in_channels)
        self.norm2 = nn.GroupNorm(num_groups=get_valid_groups(out_channels, num_groups), num_channels=out_channels)
        
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(0.1)
        
        # FiLM conditioning for global embeddings
        self.cond_channels = cond_channels
        if cond_channels > 0:
            # Project conditioning to scale and shift (uses 1+scale parameterization)
            # Standard practice: apply FiLM once per residual block
            self.film = nn.Linear(cond_channels, out_channels * 2)  # *2 for scale and shift
            
            # Initialize FiLM to identity transformation: out * (1 + 0) + 0
            # More numerically stable than gamma=1 approach
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        
        # Projection for residual connection if channels differ
        self.use_residual = (in_channels == out_channels and stride == 1)
        if not self.use_residual:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, 1, stride=stride)
    
    def forward(self, x, cond=None):
        residual = x
        
        # First block: norm → activation → conv
        out = self.norm1(x)
        out = self.activation(out)
        out = self.conv1(out)
        
        # Second block: norm → FiLM → activation → dropout → conv
        out = self.norm2(out)
        
        # Apply FiLM conditioning after norm, before activation (standard pattern)
        if cond is not None and self.cond_channels > 0:
            film_params = self.film(cond)  # [B, out_channels * 2]
            scale, shift = film_params.chunk(2, dim=1)  # [B, out_channels] each
            # Use 1+scale parameterization for numerical stability
            # Optional: constrain scale to prevent sign flips if instability occurs
            # scale = 0.1 * torch.tanh(scale)  # Uncomment if needed
            out = out * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)  # Broadcast over time
        
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
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings
