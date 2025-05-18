# processed_dataset.py
# Combines dataset with preprocessing.

from torch.utils.data import Dataset
from src.data.data_preprocessor import DataPreprocessor

class ProcessedDataset(Dataset):
    def __init__(self, base_dataset, window_size=15, target_length=600):
        self.base_dataset = base_dataset
        self.preprocessor = DataPreprocessor()
        self.window_size = window_size
        self.target_length = target_length

        # calculate window per segment based on target_length and window_size
        self.windows_per_segment = target_length // window_size
        
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

        # Extract the window of data
        start_idx = window_idx * self.window_size
        end_idx = start_idx + self.window_size

        # slice the tensors to get the window
        frames_window = frames_tensor[start_idx:end_idx]
        steering_window = steering_tensor[start_idx:end_idx]
        speed_window = speed_tensor[start_idx:end_idx]

        # Return the window of data
        return {
            'frames': frames_window,
            'steering': steering_window,
            'speed': speed_window
        }
