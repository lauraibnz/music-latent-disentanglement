import torch
import torch.nn as nn
import gin

from .blocks import UNetBlock1D, SinusoidalPositionalEmbedding


@gin.configurable
class UNet1D(nn.Module):
    """Simple 1D UNet for diffusion-based audio generation."""
    
    def __init__(self,
                 in_channels=64,  # Input latent dimensions
                 out_channels=64,  # Output latent dimensions
                 channels=[128, 256, 512],  # UNet channel progression
                 cond_channels=512,  # Timbre embedding dimension
                 time_cond_channels=512,  # Structure embedding dimension
                 time_embed_dim=128,  # Diffusion timestep embedding dimension
                 kernel_size=3):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embed_dim = time_embed_dim
        
        # Timestep embedding for diffusion
        self.time_embedding = SinusoidalPositionalEmbedding(time_embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Input projection
        self.input_proj = nn.Conv1d(in_channels, channels[0], 1)
        
        # Encoder (downsampling) path
        self.encoder_blocks = nn.ModuleList()
        prev_channels = channels[0]
        
        for i, ch in enumerate(channels):
            # Add timestep embedding to conditioning
            total_cond = cond_channels + time_embed_dim if cond_channels > 0 else time_embed_dim
            
            self.encoder_blocks.append(
                UNetBlock1D(
                    in_channels=prev_channels,
                    out_channels=ch,
                    cond_channels=total_cond,
                    time_cond_channels=time_cond_channels,
                    kernel_size=kernel_size,
                    stride=1  # No downsampling - preserves temporal resolution
                )
            )
            prev_channels = ch
        
        # Middle block (bottleneck)
        self.middle_block = UNetBlock1D(
            in_channels=channels[-1],
            out_channels=channels[-1],
            cond_channels=total_cond,
            time_cond_channels=time_cond_channels,
            kernel_size=kernel_size,
            stride=1
        )
        
        # Decoder (upsampling) path
        self.decoder_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        
        for i, ch in enumerate(reversed_channels):
            # Input is concat of skip connection + current
            skip_channels = ch  # Skip connection from encoder has 'ch' channels
            
            # Current input channels: bottleneck for first, previous decoder output for rest
            if i == 0:
                current_channels = channels[-1]  # From bottleneck (512)
            else:
                current_channels = reversed_channels[i-1]  # From previous decoder output
            
            total_block_input = current_channels + skip_channels
            
            self.decoder_blocks.append(
                UNetBlock1D(
                    in_channels=total_block_input,  # Concatenated input
                    out_channels=ch,
                    cond_channels=total_cond,
                    time_cond_channels=time_cond_channels,
                    kernel_size=kernel_size,
                    upsample=False  # No upsampling needed since no downsampling
                )
            )
        
        # Output projection
        self.output_proj = nn.Conv1d(channels[0], out_channels, 1)
    
    def forward(self, x, timesteps, cond=None, time_cond=None):
        """
        Args:
            x: Noisy input [B, in_channels, T]
            timesteps: Diffusion timesteps [B] 
            cond: Global conditioning (timbre embedding) [B, cond_channels]
            time_cond: Time-varying conditioning (structure) [B, time_cond_channels, T']
        Returns:
            Denoised output [B, out_channels, T]
        """
        # Store original input size for final matching
        original_size = x.shape[-1]
        
        # Embed timesteps
        t_emb = self.time_embedding(timesteps)  # [B, time_embed_dim]
        t_emb = self.time_proj(t_emb)
        
        # Combine global conditioning with timestep embedding
        if cond is not None:
            global_cond = torch.cat([cond, t_emb], dim=1)  # [B, cond_channels + time_embed_dim]
        else:
            global_cond = t_emb
        
        # Input projection
        x = self.input_proj(x)
        
        # Encoder path (store skip connections)
        skip_connections = []
        for i, block in enumerate(self.encoder_blocks):
            x = block(x, cond=global_cond, time_cond=time_cond)
            skip_connections.append(x)
        
        # Middle block
        x = self.middle_block(x, cond=global_cond, time_cond=time_cond)
        
        # Decoder path (use skip connections)
        for i, block in enumerate(self.decoder_blocks):
            # Get corresponding skip connection (reverse order)
            skip = skip_connections[-(i+1)]
            
            # Skip connections should match exactly since no downsampling
            assert skip.shape[-1] == x.shape[-1], f"Skip shape {skip.shape} doesn't match x shape {x.shape}"
            
            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            
            x = block(x, cond=global_cond, time_cond=time_cond)
        
        # Output projection
        x = self.output_proj(x)
        
        # Ensure output matches original input size
        if x.shape[-1] != original_size:
            x = nn.functional.interpolate(x, size=original_size, mode='linear')
        
        return x