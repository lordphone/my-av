# Iterable dataset class used to allow multiple workers to load separate videos in parallel
import torch
import random
import math
import src.utils.data_utils as data_utils

class VideoWindowIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, processed_dataset, video_indices=None, shuffle=True):
        """
        Iterable dataset that allows each worker to process different videos.
        
        Args:
            processed_dataset: The ProcessedDataset object
            video_indices: List of video indices to use (if None, uses all)
            shuffle: Whether to shuffle the windows within each video
        """
        self.processed_dataset = processed_dataset
        self.shuffle = shuffle
        
        # Use data_utils to get video indices if none are provided
        if video_indices is None:
            self.video_indices = data_utils.get_video_indices(processed_dataset)
        else:
            self.video_indices = video_indices
            
        # Track current epoch for shuffling
        self.epoch = 0
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            
        # Get a copy of video indices to process
        video_indices = list(self.video_indices)
            
        # Shuffle all videos first
        if self.shuffle:
            random.shuffle(video_indices)
        
        # Divide videos among workers
        per_worker = math.ceil(len(video_indices) / num_workers)
        start_idx = worker_id * per_worker
        end_idx = min(start_idx + per_worker, len(video_indices))
        
        # Get this worker's videos
        worker_videos = video_indices[start_idx:end_idx]
        
        # Use data_utils to group indices by video
        all_grouped_indices = data_utils.group_by_video(self.processed_dataset, worker_videos)
            
        # For each video, yield all its windows
        for video_idx in worker_videos:
            window_indices = all_grouped_indices[video_idx]
            
            # Shuffle window order if requested
            if self.shuffle:
                random.shuffle(window_indices)
                
            for idx in window_indices:
                yield self.processed_dataset[idx]

    def __len__(self):
        """
        Return the total number of windows in all assigned videos.
        This is called in the main process before worker instantiation.
        """
        # Group all windows by video
        all_grouped_indices = data_utils.group_by_video(self.processed_dataset, self.video_indices)
        
        # Sum the number of windows across all videos
        total_windows = sum(len(windows) for windows in all_grouped_indices.values())
        
        return total_windows

    def set_epoch(self, epoch):
        """Set current epoch for video/window shuffling."""
        self.epoch = epoch
