# processed_dataset.py
# Combines dataset with preprocessing.

import torch
import gc
from torch.utils.data import Dataset
from src.data.data_preprocessor import DataPreprocessor

class ProcessedDataset(Dataset):
    def __init__(self, base_dataset, window_size=20, target_length=1200, stride=20, 
                 img_size=(240, 320), frame_delay=2, future_steps=5, future_step_size=2,
                 fps=20):
        """Initialize the processed dataset.
        
        Args:
            base_dataset: The base dataset to process
            window_size: Number of frames in each window (20 frames = 1s at 20fps)
            target_length: Target length for each segment in frames
            stride: Stride between consecutive windows (20 for no overlapping)
            img_size: Size of the images (height, width)
            frame_delay: Number of frames for T-100ms lookback (2 for 100ms at 20fps)
            future_steps: Number of future steps to predict (default 5)
            future_step_size: Number of frames between future predictions (2 = 100ms at 20fps)
            fps: Original video frames per second
        """
        self.base_dataset = base_dataset
        self.window_size = window_size
        self.target_length = target_length
        self.stride = stride
        self.img_size = img_size
        self.frame_delay = frame_delay
        self.future_steps = future_steps
        self.future_step_size = future_step_size
        self.fps = fps
        
        # Create preprocessor with the same parameters
        self.preprocessor = DataPreprocessor(
            img_size=img_size,
            frame_delay=frame_delay,
            fps=fps
        )

        # Calculate windows per segment ensuring we have enough future frames for predictions
        future_frames_needed = (self.future_steps * self.future_step_size)  # 5 * 2 = 10 frames for T+500ms
        max_start = self.target_length - self.window_size - future_frames_needed
        self.windows_per_segment = (max_start // self.stride) + 1
        
        # Simple cache - only cache one video at a time
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
        # Calculate the total number of windows across all segments
        return len(self._valid_indices)

    def __getitem__(self, idx):
        # Map the index to the segment and window
        original_idx = self._valid_indices[idx]

        # Calculate the segment index and window index
        segment_idx = original_idx // self.windows_per_segment
        window_idx = original_idx % self.windows_per_segment

        # Check if the segment is already cached
        if self.cached_segment_idx != segment_idx:
            # Explicitly clear previous cached data to free memory
            if self.cached_segment_data is not None:
                # Delete references to help garbage collection
                del self.cached_segment_data
                self.cached_segment_data = None
                # Force garbage collection for large tensor cleanup
                gc.collect()
                
            # Load and prepare the segment
            segment_data = self.base_dataset[segment_idx]
            self.cached_segment_data = self.preprocessor.preprocess_segment(
                segment_data,
                target_length=self.target_length
            )
            self.cached_segment_idx = segment_idx

        # Retrieve the cached segment data
        frames_tensor, steering_tensor, speed_tensor = self.cached_segment_data

        # Extract the window of data with stride
        start_idx = window_idx * self.stride
        end_idx = start_idx + self.window_size

        # Get the 20-frame window
        frames_window = frames_tensor[start_idx:end_idx]  # [20, 3, H, W]
        steering_window = steering_tensor[start_idx:end_idx]  # [20]
        speed_window = speed_tensor[start_idx:end_idx]  # [20]

        # Create frame pairs: each frame paired with its T-100ms frame
        # This creates (window_size - frame_delay) = 18 pairs from 20 frames
        frame_pairs = []
        for i in range(self.frame_delay, self.window_size):  # frames 2-19 (18 pairs)
            current_frame = frames_window[i]  # [3, H, W]
            past_frame = frames_window[i - self.frame_delay]  # [3, H, W] 
            
            # Stack current and past frames along channel dimension
            frame_pair = torch.cat([current_frame, past_frame], dim=0)  # [6, H, W]
            frame_pairs.append(frame_pair)
        
        # Stack all frame pairs [18, 6, H, W] (sequence_length = window_size - frame_delay)
        frames_paired = torch.stack(frame_pairs)
        
        # Get corresponding steering and speed for the paired frames (frames 2-19)
        steering_sequence = steering_window[self.frame_delay:]  # [18]
        speed_sequence = speed_window[self.frame_delay:]  # [18]
        
        # Try to get future ground truth values if they exist in the segment
        # This is for the future predictions (T+100ms to T+500ms)
        future_steering = []
        future_speed = []
        
        # Use class variables instead of hardcoded values
        num_future = self.future_steps  # Default 5
        future_step = self.future_step_size  # Default 2 (100ms at 20fps)
        
        # Get future values from the segment data if they exist
        # Use the last frame in the window as reference point (frame 19 in the 20-frame window)
        current_frame_idx = start_idx + self.window_size - 1
        
        # Additional safety check to ensure we don't exceed bounds
        max_future_idx = current_frame_idx + (num_future * future_step)
        if max_future_idx >= len(steering_tensor):
            # Log a warning if this happens frequently
            pass  # Could add logging here if needed
        
        for i in range(1, num_future + 1):
            future_idx = current_frame_idx + (i * future_step)
            
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
            'frames': frames_paired,  # [18, 6, H, W] - 18 frame pairs, each with current and T-100ms
            'steering': steering_sequence,  # [18] - steering for frames 2-19
            'speed': speed_sequence,  # [18] - speed for frames 2-19
            'future_steering': torch.tensor(future_steering, dtype=torch.float32),  # [5] - T+100ms to T+500ms
            'future_speed': torch.tensor(future_speed, dtype=torch.float32)  # [5] - T+100ms to T+500ms
        }
    
    def clear_cache(self):
        """Explicitly clear the cached segment data to free memory."""
        if self.cached_segment_data is not None:
            del self.cached_segment_data
            self.cached_segment_data = None
            self.cached_segment_idx = None
            gc.collect()  # Force garbage collection
    
    def __del__(self):
        """Cleanup method called when the object is destroyed."""
        self.clear_cache()
