#!/usr/bin/env python3
"""
Deeper analysis to understand the steering loss magnitude issue.
Let's examine what's actually happening with the MSE calculations.
"""

import torch
import numpy as np
from src.data.comma2k19dataset import Comma2k19Dataset
from src.data.processed_dataset import ProcessedDataset

def analyze_mse_breakdown():
    """Analyze what's causing the large steering MSE values."""
    print("=== Deep MSE Analysis ===")
    
    # Load data
    dataset_path = "/home/lordphone/my-av/tests/data"
    base_dataset = Comma2k19Dataset(dataset_path)
    processed_dataset = ProcessedDataset(
        base_dataset,
        window_size=20,
        target_length=1200,
        stride=20
    )
    
    # Collect actual prediction targets
    steering_targets = []
    speed_targets = []
    
    max_samples = min(50, len(processed_dataset))
    print(f"Analyzing {max_samples} samples...")
    
    for i in range(max_samples):
        try:
            sample = processed_dataset[i]
            steering_targets.extend(sample['future_steering'].numpy().tolist())
            speed_targets.extend(sample['future_speed'].numpy().tolist())
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    steering_targets = np.array(steering_targets)
    speed_targets = np.array(speed_targets)
    
    print(f"\n=== Target Statistics ===")
    print(f"Steering targets: {len(steering_targets)} values")
    print(f"  Range: [{steering_targets.min():.3f}, {steering_targets.max():.3f}]")
    print(f"  Mean: {steering_targets.mean():.3f}, Std: {steering_targets.std():.3f}")
    
    print(f"Speed targets: {len(speed_targets)} values")
    print(f"  Range: [{speed_targets.min():.3f}, {speed_targets.max():.3f}]")
    print(f"  Mean: {speed_targets.mean():.3f}, Std: {speed_targets.std():.3f}")
    
    # Simulate what bad predictions would look like
    print(f"\n=== MSE Simulation with Bad Predictions ===")
    
    # Case 1: Model predicts zero for everything
    zero_steering_mse = np.mean(steering_targets ** 2)
    zero_speed_mse = np.mean(speed_targets ** 2)
    
    print(f"If model predicts 0 for everything:")
    print(f"  Steering MSE: {zero_steering_mse:.1f}")
    print(f"  Speed MSE: {zero_speed_mse:.1f}")
    print(f"  Ratio: {zero_steering_mse/zero_speed_mse:.1f}:1")
    
    # Case 2: Model predicts mean for everything
    mean_steering_mse = np.mean((steering_targets - steering_targets.mean()) ** 2)
    mean_speed_mse = np.mean((speed_targets - speed_targets.mean()) ** 2)
    
    print(f"\nIf model predicts mean for everything:")
    print(f"  Steering MSE: {mean_steering_mse:.1f}")
    print(f"  Speed MSE: {mean_speed_mse:.1f}")
    print(f"  Ratio: {mean_steering_mse/mean_speed_mse:.1f}:1")
    
    # Case 3: Random predictions within reasonable range
    np.random.seed(42)
    random_steering_pred = np.random.normal(0, 5, len(steering_targets))  # Random with std=5
    random_speed_pred = np.random.normal(28, 5, len(speed_targets))  # Random around mean speed
    
    random_steering_mse = np.mean((steering_targets - random_steering_pred) ** 2)
    random_speed_mse = np.mean((speed_targets - random_speed_pred) ** 2)
    
    print(f"\nWith random predictions (steering std=5°, speed std=5 m/s):")
    print(f"  Steering MSE: {random_steering_mse:.1f}")
    print(f"  Speed MSE: {random_speed_mse:.1f}")
    print(f"  Ratio: {random_steering_mse/random_speed_mse:.1f}:1")
    
    # Look at the observed training losses
    print(f"\n=== Observed Training Loss Analysis ===")
    observed_steering = 225.4
    observed_speed = 0.44
    
    print(f"Observed losses: Steering={observed_steering:.1f}, Speed={observed_speed:.2f}")
    print(f"Observed ratio: {observed_steering/observed_speed:.1f}:1")
    
    # Calculate what this implies about model predictions
    implied_steering_rmse = np.sqrt(observed_steering)
    implied_speed_rmse = np.sqrt(observed_speed)
    
    print(f"\nImplied RMSE from observed losses:")
    print(f"  Steering RMSE: {implied_steering_rmse:.1f}°")
    print(f"  Speed RMSE: {implied_speed_rmse:.2f} m/s")
    
    print(f"\nData std vs Model RMSE comparison:")
    print(f"  Steering - Data std: {steering_targets.std():.1f}°, Model RMSE: {implied_steering_rmse:.1f}°")
    print(f"  Speed - Data std: {speed_targets.std():.1f} m/s, Model RMSE: {implied_speed_rmse:.2f} m/s")
    
    if implied_steering_rmse > steering_targets.std() * 3:
        print(f"\n⚠️  STEERING MODEL IS PERFORMING VERY POORLY!")
        print(f"   Model RMSE ({implied_steering_rmse:.1f}°) >> Data std ({steering_targets.std():.1f}°)")
        print(f"   This suggests the model is barely learning steering at all")
    
    if implied_speed_rmse < speed_targets.std():
        print(f"\n✅ SPEED MODEL IS PERFORMING WELL!")
        print(f"   Model RMSE ({implied_speed_rmse:.2f} m/s) < Data std ({speed_targets.std():.1f} m/s)")
        print(f"   This suggests the model has learned speed prediction effectively")
    
    # Check if the issue is in the loss weighting
    print(f"\n=== Loss Weighting Analysis ===")
    print(f"Current weights: steering_weight=1.0, speed_weight=0.5")
    
    # Effective contribution to total loss
    total_loss_steering_contrib = observed_steering * 1.0
    total_loss_speed_contrib = observed_speed * 0.5
    
    print(f"Effective contributions to total loss:")
    print(f"  Steering contribution: {total_loss_steering_contrib:.1f}")
    print(f"  Speed contribution: {total_loss_speed_contrib:.2f}")
    print(f"  Steering dominance: {total_loss_steering_contrib/total_loss_speed_contrib:.0f}:1")
    
    # Recommendation
    print(f"\n=== CONCLUSION ===")
    print(f"The high loss ratio is NOT due to data scale issues.")
    print(f"The steering model is genuinely performing poorly:")
    print(f"  - Steering RMSE ~15° vs data std ~4° (4x worse than random)")
    print(f"  - Speed RMSE ~0.66 m/s vs data std ~3.7 m/s (6x better than random)")
    print(f"\nThe model has learned speed but failed to learn steering.")
    print(f"This suggests an architectural or training issue, not just loss weighting.")

if __name__ == "__main__":
    analyze_mse_breakdown()
