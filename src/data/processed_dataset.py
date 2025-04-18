# processed_dataset.py
# Combines dataset with preprocessing.

from torch.utils.data import Dataset
from src.data.data_preprocessor import DataPreprocessor

class ProcessedDataset(Dataset):
    def __init__(self, base_dataset, window_size=12):
        self.base_dataset = base_dataset
        self.preprocessor = DataPreprocessor()
        self.window_size = window_size

    def __len__(self):
        # Calculate the total number of windows across all segments, each segment is 1200 frames
        return len(self.base_dataset) * (1200 // self.window_size)

    def __getitem__(self, idx):
        # Find the correct segment for the given index
        segment_idx = idx // (1200 // self.window_size)
        window_idx = idx % (1200 // self.window_size)

        # Get the segment data
        segment_data = self.base_dataset[segment_idx]

        # Preprocess the segment to get all windows
        windowed_data = self.preprocessor.preprocess_segment(segment_data, window_size=self.window_size, stride=self.window_size)

        # Return the specific window
        return windowed_data[window_idx]