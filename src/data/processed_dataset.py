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
        # if idx in self.cache:
        #     return self.cache[idx]
            
        segment_data = self.base_dataset[idx]
        processed_data = self.preprocessor.preprocess_segment(segment_data)
        
        # if self.cache_data and processed_data is not None:
        #     self.cache[idx] = processed_data
            
        return processed_data