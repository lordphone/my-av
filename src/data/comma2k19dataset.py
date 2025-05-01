import os
import torch
from PIL import Image
import numpy as np    

class Comma2k19Dataset(torch.utils.data.Dataset):
    """
    This class represents the Comma2k19 dataset, which is a collection of highway driving
    videos used for training and evaluating machine learning models in the context of autonomous driving.

    This dataset returns paths to vital data files such as processed logs, global poses, and video files.
    """
    
    def __init__(self, base_path, transform=None):
        self.base_path = base_path  # Base path to the dataset directory
        self.transform = transform  # Optional transformation to apply to the images
        
        # Initialize the sample list by scanning the dataset directory
        self.samples = []
        self._scan_dataset()
        
    def _scan_dataset(self):
        # Scan the dataset directory structure to build the samples list
        for chunk_id in range(1, 11):  #
            chunk_path = os.path.join(self.base_path, f"Chunk_{chunk_id}")
            if not os.path.exists(chunk_path):
                continue
                
            # Determine dongle_id based on chunk number (RAV4 for 1-2, Civic for 3-10)
            dongle_id = 'b0c9d2329ad1606b' if chunk_id <= 2 else '99c94dc769b5d96e'
            
            # Process each route in the chunk
            for route_id in os.listdir(chunk_path):
                route_path = os.path.join(chunk_path, route_id)
                if not os.path.isdir(route_path):
                    continue
                    
                # Process each segment in the route
                for segment_id in os.listdir(route_path):
                    segment_path = os.path.join(route_path, segment_id)
                    if not os.path.isdir(segment_path):
                        continue
                        
                    # Add to samples if it has the required data
                    self.samples.append({
                        'chunk_id': chunk_id,
                        'route_id': route_id,
                        'segment_id': segment_id,
                        'dongle_id': dongle_id
                    })

    def __len__(self):
        """
        Returns the total number of segments in the dataset.
        """
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns paths to data for the specified index.

        Args:
            idx: The index of the sample to retrieve
            
        Returns:
            A dictionary containing the paths to data for the specified sample
        """
        sample = self.samples[idx]
        
        # Metadata
        chunk_id = sample['chunk_id']
        route_id = sample['route_id']
        segment_id = sample['segment_id']
        
        # Build path to sample directory
        segment_path = os.path.join(self.base_path, f"Chunk_{chunk_id}", 
                                   route_id, str(segment_id))
        

        # Load processed log path
        log_path = None
        if os.path.exists(os.path.join(segment_path, "processed_log")):
            log_path = os.path.join(segment_path, "processed_log")         

        # Load global pose path
        pose_path = None
        if os.path.exists(os.path.join(segment_path, "global_pose")):
            pose_path = os.path.join(segment_path, "global_pose")

        # Load video path
        video_path = None
        if os.path.exists(os.path.join(segment_path, "video.hevc")):
            video_path = os.path.join(segment_path, "video.hevc")
            
        # Return data as a dictionary
        return {
            'log_path': log_path,
            'pose_path': pose_path,
            'video_path': video_path,
            'metadata': {
                'chunk_id': chunk_id,
                'route_id': route_id,
                'segment_id': segment_id,
                'dongle_id': sample['dongle_id']
            }
        }