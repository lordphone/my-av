# processed_dataset.py
# Combines dataset with preprocessing.

from torch.utils.data import Dataset
from src.data.data_preprocessor import DataPreprocessor

class ProcessedDataset(Dataset):
    def __init__(self, base_dataset, cache_data=True):
        self.base_dataset = base_dataset
        self.preprocessor = DataPreprocessor()
        # self.cache_data = cache_data
        # self.cache = {}
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        segment_data = self.base_dataset[idx]
        windowed_data = self.preprocessor.preprocess_segment(segment_data)

        # Flatten the list of windowed data to access individual windows
        return windowed_data[idx % len(windowed_data)]