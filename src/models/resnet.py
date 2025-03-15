"""ResNet model architectures with attention mechanisms."""

import torch.nn as nn
from src.models.blocks import AttentionResidualBlock, ChannelAttentionBlock


class EnhancedEfficientResNet(nn.Module):
    """Enhanced Efficient ResNet with both spatial and channel attention."""
    
    def __init__(self, num_classes=10, base_width=31):
        super(EnhancedEfficientResNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        
        # Layer configurations (slightly modified)
        self.layer1 = self._make_layer(base_width, base_width*2, 2, stride=1)
        self.layer2 = self._make_layer(base_width*2, base_width*4, 2, stride=2)
        self.layer3 = self._make_layer(base_width*4, base_width*8, 2, stride=2)
        self.layer4 = self._make_layer(base_width*8, base_width*8, 2, stride=1)
        
        # Additional small layer to fine-tune parameter count
        self.extra_conv = nn.Conv2d(base_width*8, base_width*8, kernel_size=1, bias=False)
        self.extra_bn = nn.BatchNorm2d(base_width*8)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(base_width*8, num_classes)
        
        # Weight initialization
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [
            AttentionResidualBlock(in_channels, out_channels, stride)
        ]
        
        for _ in range(1, blocks):
            layers.append(
                AttentionResidualBlock(out_channels, out_channels)
            )
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Apply the extra convolution
        x = self.extra_conv(x)
        x = self.extra_bn(x)
        x = self.relu(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
    
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EfficientResNet(nn.Module):
    """Original Efficient ResNet with channel attention only (for compatibility)."""
    
    def __init__(self, num_classes=10, base_width=32):
        super(EfficientResNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        
        # Layer configurations
        self.layer1 = self._make_layer(base_width, base_width*2, 2, stride=1)
        self.layer2 = self._make_layer(base_width*2, base_width*4, 2, stride=2)
        self.layer3 = self._make_layer(base_width*4, base_width*8, 2, stride=2)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(base_width*8, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [
            ChannelAttentionBlock(in_channels, out_channels, stride)
        ]
        
        for _ in range(1, blocks):
            layers.append(
                ChannelAttentionBlock(out_channels, out_channels)
            )
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
