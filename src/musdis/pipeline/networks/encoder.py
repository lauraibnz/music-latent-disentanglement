import torch
import torch.nn as nn
import gin

from .blocks import ConvBlock1D


@gin.configurable
class Encoder1D(nn.Module):
    """Simple 1D encoder for latent sequence processing."""
    
    def __init__(self, 
                 in_channels=1,
                 channels=[64, 128, 256, 512],
                 kernel_size=5,
                 strides=[2, 2, 2, 2],
                 average_out=False,
                 use_tanh=True):
        super().__init__()
        
        # Sanity check
        assert len(channels) == len(strides)

        self.average_out = average_out
        self.use_tanh = use_tanh
        
        # Build encoder blocks
        blocks = []
        prev_channels = in_channels
        
        for i, (out_channels, stride) in enumerate(zip(channels, strides)):
            blocks.append(ConvBlock1D(prev_channels, out_channels, kernel_size, stride))
            prev_channels = out_channels
        
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = channels[-1]
        
        # Calculate total downsampling ratio
        self.total_ratio = 1
        for stride in strides:
            self.total_ratio *= stride
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, T]
        Returns:
            Encoded tensor [B, C', T'] or [B, C'] if average_out=True
        """
        # Pass through encoder blocks
        x = self.blocks(x)
        
        # Optional temporal averaging
        if self.average_out:
            x = torch.mean(x, dim=-1)  # [B, C', T'] -> [B, C']
        
        # Optional tanh activation
        if self.use_tanh:
            x = torch.tanh(x)
            
        return x
    
    def get_output_length(self, input_length):
        """Calculate output sequence length given input length."""
        return input_length // self.total_ratio



