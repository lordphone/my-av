# processed_dataset.py
# Combines dataset with preprocessing.

import torch
from torch.utils.data import Dataset
from src.data.data_preprocessor import DataPreprocessor

class ProcessedDataset(Dataset):
    def __init__(self, base_dataset, window_size=19, target_length=600):
        self.base_dataset = base_dataset
        self.preprocessor = DataPreprocessor()
        self.window_size = window_size  # Using window_size=19 for 1s of driving data
        self.target_length = target_length

        # calculate window per segment based on target_length and window_size
        # Account for overlapping windows if needed
        self.stride = 1  # Can be adjusted for overlapping or skip windows
        self.windows_per_segment = (target_length - window_size) // self.stride + 1
        
        # Simplified cache - just single variables instead of dict and list
        self.cached_segment_idx = None
        self.cached_segment_data = None

        self._valid_indices = self._determine_valid_indices()
    
    def _determine_valid_indices(self):
        """Determine valid indices for the dataset based on the target length."""
        valid_indices = []
        total_possible_windows = len(self.base_dataset) * self.windows_per_segment
        for i in range(total_possible_windows):
            segment_idx = i // self.windows_per_segment
            
            # try to preload or at least check if the segment is valid
            try:
                segment_data = self.base_dataset[segment_idx]
                valid_indices.append(i)
            except IndexError:
                continue
        return valid_indices

    def __len__(self):
        # Calculate the total number of windows across all segments, each segment is 600 frames
        return len(self.base_dataset) * self.windows_per_segment

    def __getitem__(self, idx):
        # Map the index to the segment and window
        original_idx = self._valid_indices[idx]

        # Calculate the segment index and window index
        segment_idx = original_idx // self.windows_per_segment
        window_idx = original_idx % self.windows_per_segment

        # Check if the segment is already cached
        if self.cached_segment_idx != segment_idx:
            # Load and prepare the segment
            segment_data = self.base_dataset[segment_idx]
            self.cached_segment_data = self.preprocessor.preprocess_segment(segment_data)
            self.cached_segment_idx = segment_idx

        # Retrieve the cached segment data
        frames_tensor, steering_tensor, speed_tensor = self.cached_segment_data

        # Extract the window of data with stride
        start_idx = window_idx * self.stride
        end_idx = start_idx + self.window_size

        # Slice the tensors to get the window
        # frames_tensor now contains frame pairs where each frame is a stack of [current, T-100ms]
        frames_window = frames_tensor[start_idx:end_idx]  # [window_size, 6, H, W]
        steering_window = steering_tensor[start_idx:end_idx]  # [window_size]
        speed_window = speed_tensor[start_idx:end_idx]  # [window_size]

        # Get current values (at the end of the sequence)
        current_steering = steering_window[-1]
        current_speed = speed_window[-1]
        
        # Try to get future ground truth values if they exist in the segment
        # This is for the 5 future predictions (T+100ms to T+500ms)
        future_steering = []
        future_speed = []
        
        # Calculate how many future frames we need (5) and the step size (2 = 100ms assuming 20 FPS)
        num_future = 5
        future_step = 2
        
        # Get future values from the segment data if they exist
        end_idx_in_segment = start_idx + self.window_size
        for i in range(1, num_future + 1):
            future_idx = end_idx_in_segment + (i * future_step) - 1
            
            # Check if the future index is within bounds
            if future_idx < len(steering_tensor):
                future_steering.append(steering_tensor[future_idx].item())
                future_speed.append(speed_tensor[future_idx].item())
            else:
                # If out of bounds, use the last available value (or we could use extrapolation)
                future_steering.append(steering_tensor[-1].item())
                future_speed.append(speed_tensor[-1].item())
        
        # Package the data for the model
        return {
            'frames': frames_window,  # [window_size, 6, H, W] - each frame contains current and T-100ms
            'steering': steering_window,  # [window_size]
            'speed': speed_window,  # [window_size]
            'current_steering': current_steering,  # Single value
            'current_speed': current_speed,  # Single value
            'future_steering': torch.tensor(future_steering, dtype=torch.float32),  # [5] - T+100ms to T+500ms
            'future_speed': torch.tensor(future_speed, dtype=torch.float32)  # [5] - T+100ms to T+500ms
        }
