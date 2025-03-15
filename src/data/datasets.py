"""Dataset classes and data loading utilities."""

import pickle
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
from src.data.augmentation import transform_train, transform_test


def load_cifar10_data(batch_size=128, num_workers=4):
    """Load CIFAR-10 dataset with train/val/test splits."""
    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )

    # Split training data into train and validation sets
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class EnhancedCIFARTestDataset(Dataset):
    """Enhanced CIFAR test dataset with robust error handling and preprocessing."""
    
    def __init__(self, pkl_file_path, transform=None):
        """
        Args:
            pkl_file_path (string): Path to the .pkl file containing test data
            transform (callable, optional): Transform to be applied on a sample
        """
        self.transform = transform
        
        # Load and process test data with error handling
        try:
            with open(pkl_file_path, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
            
            # Extract images and IDs
            self.images = data[b'data']
            self.ids = data[b'ids'] if b'ids' in data else np.arange(len(self.images))
            
            # Handle different data formats
            if len(self.images.shape) == 2:
                # If images are stored as flat arrays (N, 3072), reshape
                print(f"Reshaping flat images of shape {self.images.shape}")
                self.images = self.images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                print(f"Reshaped to {self.images.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Convert numpy array to PIL Image with proper error handling
        try:
            image = self.images[idx]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image.astype('uint8'))
            
            if self.transform:
                image = self.transform(image)
            
            return image, self.ids[idx]
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            # Return a dummy image as fallback
            if self.transform:
                return torch.zeros(3, 32, 32), self.ids[idx]
            else:
                return np.zeros((32, 32, 3), dtype=np.uint8), self.ids[idx]
