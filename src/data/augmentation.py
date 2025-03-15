"""Data augmentation techniques for CIFAR-10."""

import torchvision.transforms as transforms

# Enhanced Data Augmentation for training
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Basic transformations for testing
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Define test-time normalization for CIFAR-10
test_normalization = transforms.Normalize(
    mean=(0.4914, 0.4822, 0.4465), 
    std=(0.2023, 0.1994, 0.2010)
)

# Advanced test-time augmentation transforms
advanced_transforms = [
    # 1. Original transform (base)
    transforms.Compose([
        transforms.ToTensor(),
        test_normalization,
    ]),
    # 2. Horizontal flip
    transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        test_normalization,
    ]),
    # 3. Small crop 1
    transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.ToTensor(),
        test_normalization,
    ]),
    # 4. Small crop 2 (different padding)
    transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='edge'),
        transforms.ToTensor(),
        test_normalization,
    ]),
    # 5. Slight rotate 1
    transforms.Compose([
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        test_normalization,
    ]),
    # 6. Slight rotate 2
    transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        test_normalization,
    ]),
    # 7. Color jitter
    transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        test_normalization,
    ]),
    # 8. Color jitter 2
    transforms.Compose([
        transforms.ColorJitter(brightness=0.05, contrast=0.15, saturation=0.05, hue=0),
        transforms.ToTensor(),
        test_normalization,
    ]),
]
