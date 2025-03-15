#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Prediction script for CIFAR-10 classification model."""

import os
import sys
import argparse

# Add the src directory to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.models.resnet import EnhancedEfficientResNet
from src.data.datasets import EnhancedCIFARTestDataset
from src.utils.common import adaptive_prediction, enhanced_tta_prediction, class_specialized_prediction, ensemble_prediction_files


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate predictions for CIFAR-10 model")
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the test data pickle file')
    parser.add_argument('--output_dir', type=str, default='submissions',
                        help='Directory to save prediction files')
    parser.add_argument('--method', type=str, default='adaptive',
                        choices=['tta', 'class_specialized', 'adaptive', 'ensemble'],
                        help='Prediction method to use')
    parser.add_argument('--ensemble_files', type=str, nargs='+', 
                        help='Prediction files to ensemble (if method=ensemble)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for prediction (cuda, mps, cpu)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for prediction')
    
    return parser.parse_args()


def main():
    """Main entry point for prediction."""
    args = parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 
                            ('cuda' if torch.cuda.is_available() else 'cpu'))
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle ensemble method first
    if args.method == 'ensemble':
        if not args.ensemble_files:
            print("Error: ensemble_files must be provided when using method='ensemble'")
            return
        
        output_file = os.path.join(args.output_dir, 'ensemble_submission.csv')
        ensemble_prediction_files(args.ensemble_files, output_file)
        return
    
    # For other methods, we need to load the model
    print(f"Loading model from {args.model_path}")
    
    # Try to load the model
    try:
        model = EnhancedEfficientResNet(num_classes=10)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Successfully loaded model with enhanced architecture")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting alternative loading...")
        
        # Try loading with a more flexible approach
        state_dict = torch.load(args.model_path, map_location=device)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'])
        else:
            model.load_state_dict(state_dict, strict=False)
            print("Warning: Model loaded with strict=False, some weights may not be loaded")
    
    model = model.to(device)
    model.eval()
    
    # Generate predictions based on the method
    if args.method == 'tta':
        output_file = os.path.join(args.output_dir, 'tta_submission.csv')
        enhanced_tta_prediction(model, args.data_path, output_file)
    
    elif args.method == 'class_specialized':
        output_file = os.path.join(args.output_dir, 'specialized_submission.csv')
        class_specialized_prediction(model, args.data_path, output_file)
    
    elif args.method == 'adaptive':
        output_file = os.path.join(args.output_dir, 'adaptive_submission.csv')
        adaptive_prediction(args.model_path, args.data_path, output_file)


if __name__ == "__main__":
    main()
