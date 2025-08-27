import src.utils.data_utils as data_utils
import random
import torch
import numpy as np
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
            # print(f"Unshuffled grouped indices: {self.grouped_indices}")
            self.grouped_indices = self._group_by_video()  # Regenerate the grouped indices after shuffling
            # print(f"Shuffled grouped indices: {self.grouped_indices}")

        for group in self.grouped_indices.values():
            yield from group  # Yield all indices from the group sequentially

    def __len__(self):
        """
        Total number of samples in the dataset.
        """
        return len(self.dataset)


class WorkerAwareChunkSampler(Sampler):
    """
    Custom sampler that ensures each worker processes different videos.
    Each worker is assigned a subset of videos to process exclusively.
    """
    def __init__(self, dataset, video_indices, shuffle=True, num_workers=4, debug=False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.video_indices = list(video_indices)  # Ensure it's a list for indexing
        self.num_workers = num_workers
        self.debug = debug
        
        # If the dataset is a Subset, retrieve the underlying dataset
        if isinstance(self.dataset, Subset):
            self.dataset = self.dataset.dataset
    
    def __iter__(self):
        # Get worker info within the worker processes
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:  # Single-process data loading
            worker_id = 0
            num_workers = 1
            if self.debug:
                print(f"WARNING: Worker info is None! Falling back to single worker.")
        else:  # In a worker process
            worker_id = worker_info.id
            # Always use the num_workers from worker_info, not from __init__
            num_workers = worker_info.num_workers
            # Update our stored num_workers to match the actual worker count
            self.num_workers = num_workers
            
            if self.debug and worker_id == 0:
                print(f"Worker 0 detected total workers: {num_workers}, requested: {self.num_workers}")
                
        # Make a copy of video indices that we can shuffle
        video_indices_copy = self.video_indices.copy()
        if self.shuffle:
            # Use numpy's RandomState with a seed that's the same for all workers
            # but different for each epoch
            g = torch.Generator()
            g.manual_seed(torch.initial_seed())
            indices = torch.randperm(len(video_indices_copy), generator=g).tolist()
            video_indices_copy = [video_indices_copy[i] for i in indices]
        
        # Partition videos among workers
        videos_per_worker = len(video_indices_copy) // num_workers
        remainder = len(video_indices_copy) % num_workers
        
        # Calculate start and end indices for this worker's videos
        start_idx = worker_id * videos_per_worker + min(worker_id, remainder)
        end_idx = start_idx + videos_per_worker + (1 if worker_id < remainder else 0)
        
        # Get this worker's videos
        worker_videos = video_indices_copy[start_idx:end_idx]
        
        if self.debug:
            import os
            debug_dir = "worker_debug"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            with open(f"{debug_dir}/worker_{worker_id}_info.txt", "w") as f:
                f.write(f"Worker {worker_id}/{num_workers} assigned videos: {worker_videos}\n")
                f.write(f"Worker settings: shuffle={self.shuffle}, num_videos={len(worker_videos)}\n")
                f.write(f"Videos start_idx={start_idx}, end_idx={end_idx}\n")
        
        # Get all dataset indices for this worker's videos
        indices_to_yield = []
        grouped_indices = data_utils.group_by_video(self.dataset, worker_videos)
        
        for group in grouped_indices.values():
            indices_to_yield.extend(group)
        
        if self.debug:
            with open(f"{debug_dir}/worker_{worker_id}_info.txt", "a") as f:
                f.write(f"Worker {worker_id} will process {len(indices_to_yield)} dataset indices\n")
            
        # Yield all indices
        for idx in indices_to_yield:
            yield idx
            
    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.dataset)
