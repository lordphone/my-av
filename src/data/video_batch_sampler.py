import random
import torch
from torch.utils.data import BatchSampler

class VideoBatchSampler(BatchSampler):
    def __init__(self, dataset, video_indices=None, batch_size=16, shuffle=True, drop_last=False):
        """
        Custom batch sampler that:
        1. Assigns different videos to different workers
        2. Respects the batch_size limit for windows
        """
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Calculate windows per segment
        self.windows_per_segment = self.dataset.windows_per_segment
        
        # Get number of segments (videos)
        self.num_segments = len(self.dataset.base_dataset)
        
        # Use specific video indices if provided, otherwise use all
        self.video_indices = video_indices if video_indices is not None else list(range(self.num_segments))
        
        # Map segment indices to their corresponding window indices
        self.segment_to_windows = self._group_windows_by_segment()
        
    def _group_windows_by_segment(self):
        """Group window indices by their segment (video)."""
        segment_to_windows = {}
        
        # Convert _valid_indices to a set for faster lookup
        valid_indices_set = set(self.dataset._valid_indices)
        
        for i in self.video_indices:
            # For each segment, get all its window indices
            start_idx = i * self.windows_per_segment
            end_idx = start_idx + self.windows_per_segment
            
            # Only include indices that are in _valid_indices
            valid_window_indices = []
            for idx in range(start_idx, end_idx):
                if idx in valid_indices_set:
                    valid_window_indices.append(idx)
            
            if valid_window_indices:
                segment_to_windows[i] = valid_window_indices
                
        return segment_to_windows
        
    def __iter__(self):
        """
        Yield batches of indices, where:
        1. Each batch contains windows from the same video
        2. Batch size is limited to self.batch_size
        """
        # Get list of segments (videos)
        segments = list(self.segment_to_windows.keys())
        
        # Shuffle the segments if required
        if self.shuffle:
            random.shuffle(segments)
        
        # For each segment, yield batches of windows up to batch_size
        for segment_idx in segments:
            window_indices = self.segment_to_windows[segment_idx]
            
            # Optionally shuffle windows within this video
            if self.shuffle:
                random.shuffle(window_indices)
                
            # Yield batches of max size self.batch_size
            for i in range(0, len(window_indices), self.batch_size):
                if i + self.batch_size > len(window_indices) and self.drop_last:
                    continue
                yield window_indices[i:i + self.batch_size]
    
    def __len__(self):
        """Return the number of batches."""
        if self.drop_last:
            return sum(len(windows) // self.batch_size for windows in self.segment_to_windows.values())
        else:
            return sum((len(windows) + self.batch_size - 1) // self.batch_size 
                      for windows in self.segment_to_windows.values())