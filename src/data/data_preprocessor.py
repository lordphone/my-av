# data_preprocessor.py
# Extracts and aligns data from video and logs

import os
import numpy as np
from src.utils.frame_reader import FrameReader
import torch
from torchvision import transforms

class DataPreprocessor:
    def __init__(self, img_size=(160, 320)):
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]     # ImageNet stds
            )
        ])

    def preprocess_segment(self, segment_data, window_size=12, stride=12):
        """Preprocess a single segment of data with a sliding window."""

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
        for frame in frame_reader:
            transformed_frame = self.transform(frame)
            frames.append(transformed_frame)

        # Interpolate steering and speed data to match frames
        steering_for_frames = np.interp(frame_times, steering_times, steering_values)
        speed_for_frames = np.interp(frame_times, speed_times, speed_values)

        # Stack frames into a single tensor
        frames_tensor = torch.stack(frames)

        # Convert steering and speed data to tensors
        steering_tensor = torch.tensor(steering_for_frames, dtype=torch.float32)
        speed_tensor = torch.tensor(speed_for_frames, dtype=torch.float32)

        # Create sliding windows
        num_windows = (len(frames_tensor) - window_size) // stride + 1
        windowed_data = []

        for i in range(num_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size

            window_frames = frames_tensor[start_idx:end_idx]
            window_steering = steering_tensor[start_idx:end_idx]
            window_speed = speed_tensor[start_idx:end_idx]

            windowed_data.append({
                'frames': window_frames,
                'steering': window_steering,
                'speed': window_speed,
            })

        return windowed_data