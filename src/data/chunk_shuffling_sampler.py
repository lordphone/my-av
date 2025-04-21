from torch.utils.data import Sampler
import random

class ChunkShufflingSampler(Sampler):
    def __init__(self, dataset, shuffle=True):
        """
        Custom sampler for chunk shuffling, where each chunk corresponds to a single video.
        
        Args:
            dataset: The dataset object (ProcessedDataset).
            shuffle: Whether to shuffle the chunks (videos).
        """
        self.dataset = dataset
        self.shuffle = shuffle
        self.video_indices = self._group_by_video()

    def _group_by_video(self):
        """
        Group dataset indices by video (chunk).
        """
        video_indices = {}
        for idx in range(len(self.dataset)):
            segment_idx = idx // self.dataset.windows_per_segment
            if segment_idx not in video_indices:
                video_indices[segment_idx] = []
            video_indices[segment_idx].append(idx)
        return list(video_indices.values())

    def __iter__(self):
        """
        Yield indices for the DataLoader.
        """
        if self.shuffle:
            random.shuffle(self.video_indices)  # Shuffle the order of videos

        for video in self.video_indices:
            yield from video  # Yield all indices from the video sequentially

    def __len__(self):
        """
        Total number of samples in the dataset.
        """
        return len(self.dataset)