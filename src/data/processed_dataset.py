# processed_dataset.py
# Combines dataset with preprocessing.

from torch.utils.data import Dataset
from src.data.data_preprocessor import DataPreprocessor

class ProcessedDataset(Dataset):
    def __init__(self, base_dataset, window_size=16, target_length=1200):
        self.max_cache_size = 2
        self.base_dataset = base_dataset
        self.preprocessor = DataPreprocessor()
        self.window_size = window_size
        self.target_length = target_length

        # calculate window per segment based on target_length and window_size
        self.windows_per_segment = target_length // window_size
        self.segment_cache = {}
        self.cache_usage_order = []  # Track the order of cached segments

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
        # Calculate the total number of windows across all segments, each segment is 1200 frames
        return len(self.base_dataset) * self.windows_per_segment

    def __getitem__(self, idx):
        # Map the index to the segment and window
        original_idx = self._valid_indices[idx]

        # Calculate the segment index and window index
        segment_idx = original_idx // self.windows_per_segment
        window_idx = original_idx % self.windows_per_segment

        # Check if the segment is already cached
        if segment_idx not in self.segment_cache:
            # evict the oldest segment if cache is full
            if len(self.segment_cache) >= self.max_cache_size:
                oldest_segment = self.cache_usage_order.pop(0)
                del self.segment_cache[oldest_segment]

            # Add the new segment to the cache usage order
            self.cache_usage_order.append(segment_idx)

            # Load and prepare the segment
            segment_data = self.base_dataset[segment_idx]
            prepared_data = self.preprocessor.preprocess_segment(segment_data)
            self.segment_cache[segment_idx] = prepared_data
        else:
            # Update the usage order to mark this segment as recently used
            self.cache_usage_order.remove(segment_idx)
            self.cache_usage_order.append(segment_idx)

        # Retrieve the cached segment data
        frames_tensor, steering_tensor, speed_tensor = self.segment_cache[segment_idx]

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
