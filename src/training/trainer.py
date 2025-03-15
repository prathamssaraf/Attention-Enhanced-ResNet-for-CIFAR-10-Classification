"""Training utilities and main training loop."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

from src.training.ema import EMA
from src.training.losses import mixup_data, mixup_criterion, cutmix_data
from src.utils.common import plot_history


def train_model(model, train_loader, val_loader, learning_rate=0.1, weight_decay=5e-4, 
                num_epochs=500, checkpoint_dir='checkpoints', device='cuda'):
    """Main training function.
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        learning_rate: Initial learning rate
        weight_decay: Weight decay (L2 penalty)
        num_epochs: Number of epochs to train for
        checkpoint_dir: Directory to save model checkpoints
        device: Device to train on
    
    Returns:
        model: Trained model
        best_val_acc: Best validation accuracy
        history: Training history dictionary
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer with weight decay - SGD with momentum and nesterov
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, 
                          weight_decay=weight_decay, nesterov=True)
    
    # Standard cosine annealing without restarts
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Initialize EMA model
    ema = EMA(model, decay=0.999)
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Track best model
    best_val_acc = 0.0
    best_model_state = None
    best_ema_state = None
    
    # Training history
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Mixup and Cutmix probabilities
    mixup_prob = 0.3
    cutmix_prob = 0.3
    mixup_alpha = 1.0
    cutmix_alpha = 1.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply mixup or cutmix with probability
            r = np.random.rand()
            if r < mixup_prob:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha, device)
                mixed = True
            elif r < mixup_prob + cutmix_prob:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, cutmix_alpha, device)
                mixed = True
            else:
                mixed = False
            
            # Mixed precision training
            with autocast():
                optimizer.zero_grad()
                outputs = model(inputs)
                
                if mixed:
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, targets)
            
            # Scale gradients and optimize
            scaler.scale(loss).backward()
            
            # Gradient clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA model
            ema.update()
            
            # Memory management for MPS (Apple Silicon)
            if device == 'mps' and (progress_bar.n + 1) % 10 == 0:
                torch.mps.empty_cache()
            
            train_loss += loss.item()
            
            # Calculate accuracy (with original targets if using mixup/cutmix)
            if not mixed:
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            else:
                # Approximate accuracy for mixed samples
                _, predicted = outputs.max(1)
                total += targets.size(0)
                # For mixup/cutmix, use original targets for progress display
                correct += (lam * predicted.eq(targets_a).sum().item() + 
                           (1 - lam) * predicted.eq(targets_b).sum().item())
            
            progress_bar.set_postfix({
                'loss': train_loss / (progress_bar.n + 1),
                'acc': 100. * correct / total if total > 0 else 0.0,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total if total > 0 else 0.0
        
        # Validation phase - use EMA model
        ema.apply_shadow()
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Restore original model
        ema.restore()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            # Also save EMA state
            ema.apply_shadow()
            best_ema_state = {k: v.clone() for k, v in model.state_dict().items()}
            ema.restore()
            
            # Save both models
            torch.save(best_model_state, os.path.join(checkpoint_dir, 'best_model.pth'))
            torch.save(best_ema_state, os.path.join(checkpoint_dir, 'best_ema_model.pth'))
            print(f"Best model saved with validation accuracy: {best_val_acc:.2f}%")
        
        scheduler.step()
    
    # Plot training history
    plot_history(history, save_path="logs/training_history.png")
    
    # Load best model weights for evaluation
    model.load_state_dict(best_ema_state)  # Use EMA model for final evaluation
    return model, best_val_acc, history
