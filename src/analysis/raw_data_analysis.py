import os
import numpy as np
from tqdm import tqdm
from src.data.comma2k19dataset import Comma2k19Dataset

def load_sensor_data(file_path):
    """Load sensor data from a numpy file (with or without .npy extension)."""
    try:
        # First try with the given path
        if os.path.exists(file_path):
            return np.load(file_path).astype(np.float32)
        # Try with .npy extension if it's not there
        elif os.path.exists(file_path + '.npy'):
            return np.load(file_path + '.npy').astype(np.float32)
        else:
            return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def analyze_dataset(dataset_path):
    """Analyze steering and speed data from the Comma2k19 dataset.
    
    Args:
        dataset_path (str): Path to the root directory of the Comma2k19 dataset
    """
    print(f"Loading dataset from: {dataset_path}")
    dataset = Comma2k19Dataset(dataset_path)
    print(f"Found {len(dataset)} segments in the dataset")
    
    # Initialize lists to store all values
    all_steering = []
    all_speed = []
    
    # Process each segment
    for i, segment in enumerate(tqdm(dataset, desc="Processing segments")):
        # Get the base log path
        log_path = segment['log_path']
        if log_path is None:
            if i < 5:  # Only print first few missing log paths to avoid flooding
                print(f"Segment {i}: No log_path found in segment")
            continue
            
        try:
            # Get file paths (without .npy extension)
            steering_path = os.path.join(log_path, 'CAN', 'steering_angle', 'value')
            speed_path = os.path.join(log_path, 'CAN', 'speed', 'value')
            
            if i < 5:  # Print first few paths for debugging
                print(f"\nProcessing segment {i}:")
                print(f"  Steering path: {steering_path}")
                print(f"  Speed path: {speed_path}")
            
            # Load data
            steering_data = load_sensor_data(steering_path)
            speed_data = load_sensor_data(speed_path)
            
            if i < 5:  # Print debug info for first few segments
                print(f"  Steering data: {'Found' if steering_data is not None else 'Not found'}")
                print(f"  Speed data: {'Found' if speed_data is not None else 'Not found'}")
            
            if steering_data is not None and len(steering_data) > 0:
                all_steering.extend(steering_data)
            if speed_data is not None and len(speed_data) > 0:
                all_speed.extend(speed_data)
                
        except Exception as e:
            if i < 5:  # Only print first few errors to avoid flooding
                print(f"Error processing segment {i}: {e}")
            continue
    
    # Print debug info
    print(f"\nFound {len(all_steering)} steering samples and {len(all_speed)} speed samples")
    
    if len(all_steering) == 0 or len(all_speed) == 0:
        print("Error: No data found. Please check if the dataset path is correct and contains the expected files.")
        print("Expected structure: <dataset_path>/Chunk_*/<route_id>/<segment_id>/processed_log/CAN/{steering_angle,speed}/value.npy")
        return
        
    # Convert to numpy arrays for calculations
    all_steering = np.array(all_steering)
    all_speed = np.array(all_speed)
    
    # Calculate statistics for steering (in degrees)
    steering_stats = {
        'count': len(all_steering),
        'mean': np.mean(all_steering),
        'std': np.std(all_steering),
        'min': np.min(all_steering),
        '1%': np.percentile(all_steering, 1) if len(all_steering) > 0 else 0,
        '5%': np.percentile(all_steering, 5) if len(all_steering) > 0 else 0,
        '25%': np.percentile(all_steering, 25) if len(all_steering) > 0 else 0,
        '50%': np.percentile(all_steering, 50) if len(all_steering) > 0 else 0,
        '75%': np.percentile(all_steering, 75) if len(all_steering) > 0 else 0,
        '95%': np.percentile(all_steering, 95) if len(all_steering) > 0 else 0,
        '99%': np.percentile(all_steering, 99) if len(all_steering) > 0 else 0,
        'max': np.max(all_steering) if len(all_steering) > 0 else 0,
    }
    
    # Calculate statistics for speed (in m/s)
    speed_stats = {
        'count': len(all_speed),
        'mean': np.mean(all_speed) if len(all_speed) > 0 else 0,
        'std': np.std(all_speed) if len(all_speed) > 0 else 0,
        'min': np.min(all_speed) if len(all_speed) > 0 else 0,
        '1%': np.percentile(all_speed, 1) if len(all_speed) > 0 else 0,
        '5%': np.percentile(all_speed, 5) if len(all_speed) > 0 else 0,
        '25%': np.percentile(all_speed, 25) if len(all_speed) > 0 else 0,
        '50%': np.percentile(all_speed, 50) if len(all_speed) > 0 else 0,
        '75%': np.percentile(all_speed, 75) if len(all_speed) > 0 else 0,
        '95%': np.percentile(all_speed, 95) if len(all_speed) > 0 else 0,
        '99%': np.percentile(all_speed, 99) if len(all_speed) > 0 else 0,
        'max': np.max(all_speed) if len(all_speed) > 0 else 0,
    }
    
    # Print results
    print("\n" + "="*50)
    print("STEERING ANGLE STATISTICS (degrees)")
    print("="*50)
    for stat, value in steering_stats.items():
        print(f"{stat}: {value:.4f}")
    
    print("\n" + "="*50)
    print("SPEED STATISTICS (m/s)")
    print("="*50)
    for stat, value in speed_stats.items():
        print(f"{stat}: {value:.4f}")
    
if __name__ == "__main__":
    analyze_dataset("/home/lordphone/my-av/data/raw/comma2k19/")