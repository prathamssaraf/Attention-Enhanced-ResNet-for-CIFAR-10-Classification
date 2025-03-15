"""Attention mechanisms for neural networks."""

import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """Spatial attention module that helps the model focus on specific regions of the input."""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Generate spatial attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention module that helps the model focus on important channels."""
    
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # Ensure minimum number of channels in the bottleneck
        reduction = max(in_channels // reduction_ratio, 4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Average pooling branch
        avg_out = self.shared_mlp(self.avg_pool(x))
        
        # Max pooling branch
        max_out = self.shared_mlp(self.max_pool(x))
        
        # Combine and apply sigmoid activation
        out = avg_out + max_out
        out = self.sigmoid(out)
        
        return x * out
