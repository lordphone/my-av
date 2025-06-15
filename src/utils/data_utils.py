# data_utils.py
# data grouping and shuffling, related functions

import os
import random

def group_by_video(dataset, video_indices):
    """
    Group dataset indices by video (chunk) in the same order as video_indices.
    input:
        dataset: The dataset object (ProcessedDataset).
        video_indices: A list of video indices.
    output:
        grouped_indices: A dictionary where keys are video indices and values are lists of indices for that video.
    """
    grouped_indices = {video_idx: [] for video_idx in video_indices}
    for idx in range(len(dataset)):
        segment_idx = idx // dataset.windows_per_segment
        if segment_idx in grouped_indices:
            grouped_indices[segment_idx].append(idx)
    return grouped_indices

def get_video_indices(dataset):
    """
    Get all video indices from the dataset.
    input:
        dataset: The dataset object (ProcessedDataset).
    output:
        video_indices: A list of all video indices.
    """
    video_indices = []
    for idx in range(len(dataset)):
        segment_idx = idx // dataset.windows_per_segment
        if segment_idx not in video_indices:
            video_indices.append(segment_idx)
    return video_indices
