"""Common utility functions for the project."""

import os
import random
import numpy as np
import torch
import pandas as pd
import gc
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_submission(model, test_loader, filename="submission.csv", device='cuda'):
    """Generate a submission file with predictions."""
    model.eval()
    all_predictions = []
    all_ids = []
    
    with torch.no_grad():
        for inputs, ids in tqdm(test_loader, desc="Generating predictions"):
            inputs = inputs.to(device)
            
            # Test time augmentation - average predictions from different transforms
            outputs = model(inputs)
            
            # Add horizontal flip augmentation
            flipped_inputs = torch.flip(inputs, dims=[3])
            flipped_outputs = model(flipped_inputs)
            
            # Average predictions
            avg_outputs = (outputs + flipped_outputs) / 2.0
            
            _, predictions = avg_outputs.max(1)
            all_predictions.extend(predictions.cpu().numpy())
            all_ids.extend(ids.numpy())
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'ID': all_ids,
        'Labels': all_predictions
    })
    
    submission_df.to_csv(filename, index=False)
    print(f"Submission file created: {filename}")
    return submission_df


def enhanced_tta_prediction(model, pkl_file_path, output_filename="tta_submission.csv", num_transforms=8, device='cuda'):
    """Advanced test-time augmentation with weighted averaging of predictions."""
    from src.data.datasets import EnhancedCIFARTestDataset
    from src.data.augmentation import advanced_transforms
    
    print("Starting enhanced TTA prediction...")
    model.eval()
    
    # Select requested number of transforms
    transforms_to_use = advanced_transforms[:num_transforms]
    
    # Weights for different transforms (giving higher weight to original image)
    weights = [2.0]  # Increase from 1.5 to 2.0
    weights.extend([1.0] * (len(transforms_to_use) - 1))
    
    # Normalize weights
    weights = [w / sum(weights) for w in weights]
    
    # Storage for all predictions
    all_probs = []
    image_ids = None
    
    # Process each transform
    for i, transform in enumerate(tqdm(transforms_to_use, desc="Processing augmentations")):
        # Create dataset with this transform
        dataset = EnhancedCIFARTestDataset(pkl_file_path, transform=transform)
        dataloader = DataLoader(
            dataset, 
            batch_size=32,  # Smaller batch size to avoid OOM
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )
        
        # Collect batch predictions
        batch_probs = []
        batch_ids = []
        
        with torch.no_grad():
            for images, ids in dataloader:
                images = images.to(device)
                outputs = model(images)
                
                # Apply temperature scaling for better calibration
                outputs = outputs / 0.9  # Temperature parameter
                
                # Get softmax probabilities
                probs = F.softmax(outputs, dim=1)
                
                batch_probs.append(probs.cpu().numpy())
                batch_ids.append(ids.numpy())
                
                # Free up memory
                del images, outputs, probs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Concatenate all batches
        augmentation_probs = np.concatenate(batch_probs)
        
        # Apply weight to this augmentation's predictions
        all_probs.append(augmentation_probs * weights[i])
        
        # Store IDs (same for all augmentations)
        if image_ids is None:
            image_ids = np.concatenate(batch_ids)
        
        # Free memory
        del batch_probs, augmentation_probs
        gc.collect()
    
    # Combine predictions by averaging softmax probabilities
    avg_probs = np.sum(all_probs, axis=0)
    final_preds = np.argmax(avg_probs, axis=1)
    
    # Create and save submission
    submission_df = pd.DataFrame({
        'ID': image_ids, 
        'Labels': final_preds
    })
    submission_df = submission_df.sort_values('ID')
    submission_df.to_csv(output_filename, index=False)
    print(f"Enhanced TTA submission file created: {output_filename}")
    return submission_df


def class_specialized_prediction(model, pkl_file_path, output_filename="specialized_submission.csv", device='cuda'):
    """Creates predictions with specialized handling for different classes based on confidence thresholds."""
    from src.data.datasets import EnhancedCIFARTestDataset
    from src.data.augmentation import advanced_transforms
    
    print("Starting class-specialized prediction...")
    model.eval()
    
    # Initial pass with base transform
    base_transform = advanced_transforms[0]
    dataset = EnhancedCIFARTestDataset(pkl_file_path, transform=base_transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    
    # First pass - get initial predictions and confidence
    initial_preds = []
    confidence_scores = []
    image_ids = []
    
    with torch.no_grad():
        for images, ids in tqdm(dataloader, desc="Initial prediction pass"):
            images = images.to(device)
            outputs = model(images)
            
            # Get softmax probabilities
            probs = F.softmax(outputs, dim=1)
            
            # Get predictions and confidence
            values, preds = torch.max(probs, dim=1)
            
            initial_preds.extend(preds.cpu().numpy())
            confidence_scores.extend(values.cpu().numpy())
            image_ids.extend(ids.numpy())
            
            # Free memory
            del images, outputs, probs, values, preds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Confidence thresholds for different classes - based on common confusion patterns
    class_confidence_thresholds = {
        0: 0.85,  # airplane
        1: 0.90,  # automobile
        2: 0.75,  # bird - still relatively difficult
        3: 0.70,  # cat - difficult class
        4: 0.75,  # deer
        5: 0.70,  # dog - difficult class
        6: 0.85,  # frog
        7: 0.85,  # horse
        8: 0.90,  # ship
        9: 0.90,  # truck
    }
    
    # Identify low confidence predictions
    low_conf_indices = []
    for i, (pred, conf) in enumerate(zip(initial_preds, confidence_scores)):
        if conf < class_confidence_thresholds.get(pred, 0.75):
            low_conf_indices.append(i)
    
    print(f"Found {len(low_conf_indices)} low confidence predictions ({len(low_conf_indices)/len(initial_preds)*100:.2f}%)")
    
    # For low confidence predictions, use enhanced TTA
    final_preds = list(initial_preds)
    
    if low_conf_indices:
        # Process all transforms for low confidence cases only
        low_conf_probs = []
        
        for transform in tqdm(advanced_transforms, desc="Processing difficult cases"):
            dataset = EnhancedCIFARTestDataset(pkl_file_path, transform=transform)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
            
            all_outputs = []
            
            with torch.no_grad():
                for batch_idx, (images, _) in enumerate(dataloader):
                    # Only process batches that contain low confidence samples
                    batch_start = batch_idx * 64
                    batch_end = min(batch_start + 64, len(dataset))
                    batch_indices = list(range(batch_start, batch_end))
                    
                    # Check if any low confidence indices are in this batch
                    process_batch = any(idx in low_conf_indices for idx in batch_indices)
                    
                    if process_batch:
                        images = images.to(device)
                        outputs = model(images)
                        all_outputs.append(outputs.cpu())
                    else:
                        # Skip this batch by adding dummy outputs
                        all_outputs.append(torch.zeros(len(batch_indices), 10))
            
            # Concatenate all outputs
            all_outputs = torch.cat(all_outputs)
            
            # Extract only the low confidence predictions
            selected_outputs = all_outputs[low_conf_indices]
            selected_probs = F.softmax(selected_outputs / 1.2, dim=1).numpy()  # Apply temperature
            
            low_conf_probs.append(selected_probs)
            
            # Free memory
            del all_outputs, selected_outputs, selected_probs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Average probabilities for low confidence predictions
        avg_probs = np.mean(np.stack(low_conf_probs), axis=0)
        improved_preds = np.argmax(avg_probs, axis=1)
        
        # Update predictions for low confidence cases
        for i, idx in enumerate(low_conf_indices):
            final_preds[idx] = improved_preds[i]
    
    # Create and save submission
    submission_df = pd.DataFrame({
        'ID': image_ids, 
        'Labels': final_preds
    })
    submission_df = submission_df.sort_values('ID')
    submission_df.to_csv(output_filename, index=False)
    print(f"Class-specialized submission file created: {output_filename}")
    return submission_df


def adaptive_prediction(model_path, pkl_file_path, output_filename="adaptive_submission.csv", device='cuda'):
    """Creates predictions using an adaptive approach that combines multiple techniques."""
    import torch
    import os
    from src.data.datasets import EnhancedCIFARTestDataset
    from src.data.augmentation import advanced_transforms
    from src.models.resnet import EnhancedEfficientResNet, EfficientResNet
    
    print("Starting adaptive prediction process...")
    
    # Load model
    try:
        # First try the enhanced model architecture
        model = EnhancedEfficientResNet(num_classes=10)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Successfully loaded model with enhanced architecture")
    except:
        try:
            # Fall back to original architecture
            model = EfficientResNet(num_classes=10)
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Successfully loaded model with original architecture")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting alternative loading method...")
            
            # Try loading just the state dict
            model = EnhancedEfficientResNet(num_classes=10)
            state_dict = torch.load(model_path, map_location=device)
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                model.load_state_dict(state_dict['state_dict'])
            else:
                model.load_state_dict(state_dict, strict=False)
                print("Warning: Model loaded with strict=False, some weights may not be loaded")
    
    model = model.to(device)
    model.eval()
    
    # Step 1: Run enhanced TTA
    print("\nStep 1: Running enhanced TTA prediction...")
    enhanced_tta_prediction(model, pkl_file_path, "tmp_tta.csv", num_transforms=8, device=device)
    
    # Step 2: Run class-specialized prediction
    print("\nStep 2: Running class-specialized prediction...")
    class_specialized_prediction(model, pkl_file_path, "tmp_spec.csv", device=device)
    
    # Step 3: Combine predictions based on confidence
    print("\nStep 3: Combining predictions adaptively...")
    
    # Load both prediction files
    tta_df = pd.read_csv("tmp_tta.csv")
    spec_df = pd.read_csv("tmp_spec.csv")
    
    # Ensure they have the same order
    tta_df = tta_df.sort_values('ID')
    spec_df = spec_df.sort_values('ID')
    
    # Get confidence scores for all predictions
    base_transform = advanced_transforms[0]
    dataset = EnhancedCIFARTestDataset(pkl_file_path, transform=base_transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    
    all_probs = []
    all_ids = []
    
    with torch.no_grad():
        for images, ids in tqdm(dataloader, desc="Computing confidence scores"):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_ids.extend(ids.numpy())
    
    all_probs = np.concatenate(all_probs)
    id_to_index = {id_val: i for i, id_val in enumerate(all_ids)}
    
    # Combine predictions
    final_labels = []
    
    # Thresholds for decision making
    HIGH_CONF = 0.95
    MED_CONF = 0.80
    
    # Additional confidence boosting for specific classes
    class_boost = {
        2: 0.05,  # bird
        3: 0.05,  # cat
        5: 0.05,  # dog
    }
    
    for i in range(len(tta_df)):
        img_id = tta_df.iloc[i]['ID']
        tta_pred = tta_df.iloc[i]['Labels']
        spec_pred = spec_df.loc[spec_df['ID'] == img_id, 'Labels'].values[0]
        
        # Get confidence for both predictions
        idx = id_to_index[img_id]
        prob_vector = all_probs[idx]
        
        # Apply confidence boosting for certain classes
        tta_conf = prob_vector[tta_pred]
        if tta_pred in class_boost:
            tta_conf += class_boost[tta_pred]
            
        spec_conf = prob_vector[spec_pred]
        if spec_pred in class_boost:
            spec_conf += class_boost[spec_pred]
        
        # Decision logic
        if tta_pred == spec_pred:
            # Both methods agree
            final_labels.append(tta_pred)
        elif tta_conf > HIGH_CONF:
            # TTA prediction has very high confidence
            final_labels.append(tta_pred)
        elif spec_conf > HIGH_CONF:
            # Class specialized prediction has very high confidence
            final_labels.append(spec_pred)
        elif tta_conf > MED_CONF and tta_conf > spec_conf:
            # TTA has decent confidence and higher than class specialized
            final_labels.append(tta_pred)
        elif spec_conf > MED_CONF and spec_conf > tta_conf:
            # Class specialized has decent confidence and higher than TTA
            final_labels.append(spec_pred)
        elif tta_pred in [2, 3, 5]:
            # For difficult classes, prefer specialized prediction
            final_labels.append(spec_pred)
        else:
            # Default to enhanced TTA
            final_labels.append(tta_pred)
    
    # Create final submission
    submission_df = pd.DataFrame({
        'ID': tta_df['ID'],
        'Labels': final_labels
    })
    submission_df.to_csv(output_filename, index=False)
    
    # Clean up temporary files
    if os.path.exists("tmp_tta.csv"):
        os.remove("tmp_tta.csv")
    if os.path.exists("tmp_spec.csv"):
        os.remove("tmp_spec.csv")
    
    print(f"Adaptive submission file created: {output_filename}")
    return submission_df


def ensemble_prediction_files(file_paths, output_filename="ensemble_submission.csv"):
    """Ensembles multiple prediction CSV files."""
    print(f"Creating ensemble from {len(file_paths)} prediction files...")
    
    # Load all prediction files
    dataframes = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df = df.sort_values('ID')
        dataframes.append(df)
    
    # Ensure all dataframes have the same IDs
    for i in range(1, len(dataframes)):
        assert np.array_equal(dataframes[0]['ID'].values, dataframes[i]['ID'].values), "ID mismatch between files"
    
    # Get predictions from each file
    all_preds = np.array([df['Labels'].values for df in dataframes])
    
    # Get majority vote for each sample
    final_preds = []
    for i in range(len(dataframes[0])):
        sample_preds = all_preds[:, i]
        values, counts = np.unique(sample_preds, return_counts=True)
        max_count_idx = np.argmax(counts)
        final_preds.append(values[max_count_idx])
    
    # Create final submission
    ensemble_df = pd.DataFrame({
        'ID': dataframes[0]['ID'],
        'Labels': final_preds
    })
    ensemble_df.to_csv(output_filename, index=False)
    print(f"Ensemble submission file created: {output_filename}")
    return ensemble_df


def plot_history(history, save_path="logs/training_history.png"):
    """Plot training history curves."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()
    
    print(f"Training history plots saved to {save_path}")
