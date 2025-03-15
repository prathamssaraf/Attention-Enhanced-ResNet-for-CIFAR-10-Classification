"""Building blocks for neural network architectures."""

import torch.nn as nn
from src.models.attention import SpatialAttention, ChannelAttention


class AttentionResidualBlock(nn.Module):
    """Residual block with both channel and spatial attention mechanisms."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(AttentionResidualBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduction = max(out_channels // 16, 4)  # Ensure at least 4 channels
        self.channel_attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = SpatialAttention(kernel_size=7)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Channel attention
        avg_out = self.channel_attention(self.avg_pool(out))
        max_out = self.channel_attention(self.max_pool(out))
        out = out * (avg_out + max_out)
        
        # Spatial attention
        out = self.spatial_attention(out)
        
        # Residual connection
        identity = self.shortcut(identity)
        out += identity
        out = self.relu(out)
        
        return out


class ChannelAttentionBlock(nn.Module):
    """Residual block with channel attention only (for compatibility with original model)."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ChannelAttentionBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduction = max(out_channels // 16, 4)  # Ensure at least 4 channels
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Channel attention
        avg_out = self.attention(self.avg_pool(out))
        max_out = self.attention(self.max_pool(out))
        out = out * (avg_out + max_out)
        
        # Residual connection
        identity = self.shortcut(identity)
        out += identity
        out = self.relu(out)
        
        return out
