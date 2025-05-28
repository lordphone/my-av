#!/usr/bin/env python3
"""
Data Analysis Script to identify steering prediction issues.

This script analyzes the training data to understand:
1. Data ranges for steering vs speed
2. Scale disparities causing loss magnitude differences
3. Root cause of steering prediction performance issues
"""

import torch
import numpy as np
from src.data.comma2k19dataset import Comma2k19Dataset
from src.data.processed_dataset import ProcessedDataset

def analyze_data_ranges():
    """Analyze the data ranges in the training dataset."""
    print("=== Data Range Analysis ===")
    
    # Load a small subset of data for analysis
    dataset_path = "/home/lordphone/my-av/tests/data"
    base_dataset = Comma2k19Dataset(dataset_path)
    processed_dataset = ProcessedDataset(
        base_dataset,
        window_size=20,
        target_length=1200,
        stride=20
    )
    
    print(f"Total samples in processed dataset: {len(processed_dataset)}")
    
    # Collect statistics from first 100 samples
    steering_values = []
    speed_values = []
    future_steering_values = []
    future_speed_values = []
    
    max_samples = min(500, len(processed_dataset))
    print(f"Analyzing first {max_samples} samples...")
    
    for i in range(max_samples):
        try:
            sample = processed_dataset[i]
            
            # Current steering and speed values
            steering_values.extend(sample['steering'].numpy().tolist())
            speed_values.extend(sample['speed'].numpy().tolist())
            
            # Future prediction targets
            future_steering_values.extend(sample['future_steering'].numpy().tolist())
            future_speed_values.extend(sample['future_speed'].numpy().tolist())
            
            if i % 20 == 0:
                print(f"Processed {i+1}/{max_samples} samples...")
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Convert to numpy arrays
    steering_values = np.array(steering_values)
    speed_values = np.array(speed_values)
    future_steering_values = np.array(future_steering_values)
    future_speed_values = np.array(future_speed_values)
    
    print("\n=== Current Values Statistics ===")
    print(f"Steering (degrees):")
    print(f"  Range: [{steering_values.min():.3f}, {steering_values.max():.3f}]")
    print(f"  Mean: {steering_values.mean():.3f}, Std: {steering_values.std():.3f}")
    print(f"  Typical MSE range: {steering_values.std()**2:.1f}")
    
    print(f"\nSpeed (m/s):")
    print(f"  Range: [{speed_values.min():.3f}, {speed_values.max():.3f}]")
    print(f"  Mean: {speed_values.mean():.3f}, Std: {speed_values.std():.3f}")
    print(f"  Typical MSE range: {speed_values.std()**2:.3f}")
    
    print("\n=== Future Prediction Targets Statistics ===")
    print(f"Future Steering (degrees):")
    print(f"  Range: [{future_steering_values.min():.3f}, {future_steering_values.max():.3f}]")
    print(f"  Mean: {future_steering_values.mean():.3f}, Std: {future_steering_values.std():.3f}")
    print(f"  Typical MSE range: {future_steering_values.std()**2:.1f}")
    
    print(f"\nFuture Speed (m/s):")
    print(f"  Range: [{future_speed_values.min():.3f}, {future_speed_values.max():.3f}]")
    print(f"  Mean: {future_speed_values.mean():.3f}, Std: {future_speed_values.std():.3f}")
    print(f"  Typical MSE range: {future_speed_values.std()**2:.3f}")
    
    # Calculate scale difference
    steering_scale = future_steering_values.std()**2
    speed_scale = future_speed_values.std()**2
    scale_ratio = steering_scale / speed_scale
    
    print(f"\n=== Scale Analysis ===")
    print(f"Steering MSE scale: ~{steering_scale:.1f}")
    print(f"Speed MSE scale: ~{speed_scale:.3f}")
    print(f"Scale ratio (steering/speed): {scale_ratio:.1f}:1")
    
    # Loss weighting analysis
    print(f"\n=== Current Loss Weighting Analysis ===")
    print(f"Current weights: steering_weight=1.0, speed_weight=0.5")
    
    # Observed loss values from logs
    observed_steering_loss = 225.4  # From epoch 1
    observed_speed_loss = 0.44      # From epoch 1
    
    print(f"Observed losses: steering={observed_steering_loss:.1f}, speed={observed_speed_loss:.2f}")
    print(f"Loss ratio: {observed_steering_loss/observed_speed_loss:.1f}:1")
    
    # Recommended weighting to balance losses
    recommended_steering_weight = 1.0
    recommended_speed_weight = observed_steering_loss / observed_speed_loss
    
    print(f"\n=== Recommendations ===")
    print(f"1. Data Scale Issue Identified:")
    print(f"   - Steering values have much larger magnitude than speed")
    print(f"   - MSE naturally produces larger values for steering")
    
    print(f"\n2. Recommended Loss Weighting:")
    print(f"   - steering_weight = {recommended_steering_weight}")
    print(f"   - speed_weight = {recommended_speed_weight:.1f}")
    print(f"   - This will roughly balance the contribution of both losses")
    
    print(f"\n3. Alternative Solutions:")
    print(f"   - Normalize steering angles to [-1, 1] range")
    print(f"   - Use different loss functions (e.g., Huber loss)")
    print(f"   - Apply separate normalization for steering predictions")

if __name__ == "__main__":
    analyze_data_ranges()
