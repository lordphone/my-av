import src.utils.data_utils as data_utils
import random

from torch.utils.data import Sampler, Subset

class ChunkShufflingSampler(Sampler):
    def __init__(self, dataset, video_indices, shuffle=True):
        """
        Custom sampler for chunk shuffling, where each chunk corresponds to a single video.
        
        Args:
            dataset: The dataset object (ProcessedDataset).
            shuffle: Whether to shuffle the chunks (videos).
        """
        self.dataset = dataset
        self.shuffle = shuffle
        self.video_indices = video_indices

        # If the dataset is a Subset, retrieve the underlying dataset
        if isinstance(self.dataset, Subset):
            self.dataset = self.dataset.dataset
            
        self.grouped_indices = self._group_by_video()

    def _group_by_video(self):
        grouped_indices = data_utils.group_by_video(self.dataset, self.video_indices)
        return grouped_indices

    def __iter__(self):
        """
        Yield indices for the DataLoader.
        """
        if self.shuffle:
            random.shuffle(self.video_indices)  # Shuffle the video indices
            print(f"Unshuffled grouped indices: {self.grouped_indices}")
            self.grouped_indices = self._group_by_video()  # Regenerate the grouped indices after shuffling
            print(f"Shuffled grouped indices: {self.grouped_indices}")

        for group in self.grouped_indices.values():
            yield from group  # Yield all indices from the group sequentially

    def __len__(self):
        """
        Total number of samples in the dataset.
        """
        return len(self.dataset)