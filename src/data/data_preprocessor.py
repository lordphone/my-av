# data_preprocessor.py
# Extracts and aligns data from video and logs

import os
import numpy as np
from src.utils.frame_reader import FrameReader
import torch
from torchvision import transforms

class DataPreprocessor:
    def __init__(self, 
                 img_size=(240, 320),
                 frame_delay=2,  # For T-100ms frames at 20fps
                 fps=20,         # Original video FPS 
                 normalization_mean=[0.485, 0.456, 0.406],  # ImageNet means
                 normalization_std=[0.229, 0.224, 0.225]    # ImageNet stds
                ):
        self.img_size = img_size
        self.frame_delay = frame_delay
        self.fps = fps
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=normalization_mean,
                std=normalization_std
            )
        ])

    def preprocess_segment(self, segment_data, target_length=1200):
        """Preprocess a single segment of data with a sliding window.
        
        Args:
            segment_data: Dictionary containing video_path, log_path, and pose_path
            target_length: Target length for processed segments, in frames
            
        Returns:
            tuple of (frame_pairs_tensor, steering_tensor, speed_tensor)
        """

        # Get paths from the segment data
        video_path = segment_data['video_path']
        log_path = segment_data['log_path']
        pose_path = segment_data['pose_path']

        # Load frame timestamps
        frame_times = np.load(os.path.join(pose_path, 'frame_times'))

        # Load steering data
        steering_times = np.load(os.path.join(log_path, 'CAN/steering_angle/t'))
        steering_values = np.load(os.path.join(log_path, 'CAN/steering_angle/value'))

        # Load speed data
        speed_times = np.load(os.path.join(log_path, 'CAN/speed/t'))
        speed_values = np.load(os.path.join(log_path, 'CAN/speed/value'))

        # Squeeze speed values since for some reason they are 3D
        speed_values = np.squeeze(speed_values)

        # Initialize frame reader
        frame_reader = FrameReader(video_path)

        # Read frames and apply transformations
        frames = []
        for i, frame in enumerate(frame_reader):
            transformed_frame = self.transform(frame)
            frames.append(transformed_frame)

        # Interpolate steering and speed data to match frames
        steering_for_frames = np.interp(frame_times, steering_times, steering_values)
        speed_for_frames = np.interp(frame_times, speed_times, speed_values)

        # Keep all frames since videos are already at desired FPS
        # If FPS adjustment is needed, would be handled here based on self.fps

        # Pad frames, steering, and speed data to ensure all videos meet target length
        current_len = len(frames)
        if current_len < target_length:
            padding_frames = target_length - current_len
            # Use the last available frame for padding, or a zero tensor if no frames exist
            pad_frame = frames[-1] if frames else torch.zeros_like(self.transform(np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)))
            frames.extend([pad_frame] * padding_frames)
        elif current_len > target_length: # Truncate if longer
            frames = frames[:target_length]
            steering_for_frames = steering_for_frames[:target_length]
            speed_for_frames = speed_for_frames[:target_length]

        # Pad steering and speed data
        if len(steering_for_frames) < target_length:
            padding_steering = target_length - len(steering_for_frames)
            steering_for_frames = np.pad(steering_for_frames, (0, padding_steering), 'edge')
        if len(speed_for_frames) < target_length:
            padding_speed = target_length - len(speed_for_frames)
            speed_for_frames = np.pad(speed_for_frames, (0, padding_speed), 'edge')

        # Stack all individual frames into a single tensor [N, 3, H, W]
        # Each frame is now just a single normalized frame, not a pair
        frames_tensor = torch.stack(frames)

        # Convert steering and speed data to tensors
        steering_tensor = torch.tensor(steering_for_frames, dtype=torch.float32)
        speed_tensor = torch.tensor(speed_for_frames, dtype=torch.float32)

        return frames_tensor, steering_tensor, speed_tensor