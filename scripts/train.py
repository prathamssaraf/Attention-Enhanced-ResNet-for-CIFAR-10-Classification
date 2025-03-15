#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training script for CIFAR-10 classification model."""

import os
import sys
import argparse
import yaml

# Add the src directory to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.models.resnet import EnhancedEfficientResNet
from src.training.trainer import train_model
from src.data.datasets import load_cifar10_data
from src.utils.common import set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CIFAR-10 classification model")
    
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for training (cuda, mps, cpu)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def update_config_with_args(config, args):
    """Update configuration with command line arguments."""
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.device is not None:
        config['device'] = args.device
    
    return config


def main():
    """Main entry point for training."""
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Load and update configuration
    config = load_config(args.config)
    config = update_config_with_args(config, args)
    
    # Set device
    if config.get('device') is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 
                              ('cuda' if torch.cuda.is_available() else 'cpu'))
    else:
        device = torch.device(config['device'])
    
    print(f"Using device: {device}")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load data
    train_loader, val_loader, test_loader = load_cifar10_data(
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    
    # Initialize model
    model = EnhancedEfficientResNet(
        num_classes=10,
        base_width=config['model']['base_width']
    )
    
    # Print model parameter count
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")
    
    # Check parameter count against limit (if any)
    if config['model'].get('param_limit') and param_count > config['model']['param_limit']:
        print(f"WARNING: Model exceeds {config['model']['param_limit']} parameter limit "
              f"by {param_count - config['model']['param_limit']:,}")
    
    # Move model to device
    model = model.to(device)
    
    # Train model
    model, best_val_acc, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        num_epochs=config['training']['num_epochs'],
        checkpoint_dir=args.checkpoint_dir,
        device=device
    )
    
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
