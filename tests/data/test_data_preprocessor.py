import unittest
import torch
import os
import numpy as np
import time
from src.data.data_preprocessor import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Set up test fixtures - run preprocessing once and store the results for all tests"""
        self.data_preprocessor = DataPreprocessor(img_size=(240, 320))
        self.segment_data = {
            'video_path': '/home/lordphone/my-av/tests/data/Chunk_1/b0c9d2329ad1606b|2018-07-27--06-03-57/3/video.hevc',
            'log_path': '/home/lordphone/my-av/tests/data/Chunk_1/b0c9d2329ad1606b|2018-07-27--06-03-57/3/processed_log',
            'pose_path': '/home/lordphone/my-av/tests/data/Chunk_1/b0c9d2329ad1606b|2018-07-27--06-03-57/3/global_pose'
        }
        
        # Process data once for all tests
        self.frames_tensor, self.steering_tensor, self.speed_tensor = self.data_preprocessor.preprocess_segment(self.segment_data)

    def test_tensor_shapes(self):
        """Test that the preprocessor creates tensors with the correct shape"""
        # Check dimensions of frames tensor - now contains frame pairs with 6 channels
        self.assertEqual(len(self.frames_tensor.shape), 4, "Frames tensor should be 4-dimensional [frames, channels, height, width]")
        self.assertEqual(self.frames_tensor.shape[1], 6, "Each frame should have 6 channels (3 for current + 3 for T-100ms)")
        self.assertEqual(self.frames_tensor.shape[2:], (240, 320), "Frame dimensions should match the specified img_size")

        # Check dimensions of steering and speed tensors
        self.assertEqual(len(self.steering_tensor.shape), 1, "Steering tensor should be 1-dimensional")
        self.assertEqual(len(self.speed_tensor.shape), 1, "Speed tensor should be 1-dimensional")
        self.assertEqual(len(self.steering_tensor), len(self.speed_tensor), "Steering and speed tensors should have the same length")
        self.assertEqual(self.frames_tensor.shape[0], len(self.steering_tensor), "Number of frames should match length of steering data")
        
    def test_frame_pair_structure(self):
        """Test that each frame is a proper pair of (current, T-100ms)"""
        # Check that the first 3 channels and second 3 channels are not identical
        # Skip the first few frames where we use duplicates for T-100ms
        frame_delay = 2  # Same as in preprocessor
        for i in range(frame_delay + 1, min(10, self.frames_tensor.shape[0])):
            current_frame = self.frames_tensor[i, :3, :, :]
            past_frame = self.frames_tensor[i, 3:, :, :]
            
            # Get the expected past frame directly from the sequence
            expected_past_frame = self.frames_tensor[i - frame_delay, :3, :, :]
            
            # Check if the past frame matches the frame from 2 steps ago
            # This is a loose check, as the frames might not match exactly due to preprocessing
            similarity = torch.nn.functional.cosine_similarity(past_frame.flatten(), expected_past_frame.flatten(), dim=0)
            self.assertGreater(similarity, 0.9, f"The past frame at index {i} should be similar to the frame at index {i-frame_delay}")
            
            # Make sure current and past frames are not identical (would suggest an error)
            self.assertFalse(torch.all(torch.eq(current_frame, past_frame)), 
                           f"Current and past frames should not be identical at index {i}")

    def test_normalization(self):
        """Test that the frames are normalized correctly"""
        # Check normalization - need to handle 6 channels (2 RGB images)
        # ImageNet means and stds are applied to each RGB image separately
        mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406]).view(1, 6, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225]).view(1, 6, 1, 1)
        
        # Denormalize the frames to check if values are within the expected range
        denormalized_frames = self.frames_tensor * std + mean
        
        # Check that pixel values are within [0, 1] range
        self.assertTrue((denormalized_frames >= 0).all(), 
                "Denormalized frames should have values >= 0")
        self.assertTrue((denormalized_frames <= 1).all(), 
                "Denormalized frames should have values <= 1")

    def test_data_dtype(self):
        """Test that the tensors have the correct data types"""
        # Check data types
        self.assertEqual(self.frames_tensor.dtype, torch.float32, 
                         "Frames tensor should be of type float32")
        self.assertEqual(self.steering_tensor.dtype, torch.float32, 
                         "Steering tensor should be of type float32")  
        self.assertEqual(self.speed_tensor.dtype, torch.float32, 
                         "Speed tensor should be of type float32")

    def test_data_validity(self):
        """Test that the tensors don't contain invalid values"""
        # Check for NaN values
        self.assertFalse(torch.isnan(self.frames_tensor).any(), 
                         "Frames tensor should not contain NaN values")
        self.assertFalse(torch.isnan(self.steering_tensor).any(), 
                         "Steering tensor should not contain NaN values")
        self.assertFalse(torch.isnan(self.speed_tensor).any(), 
                         "Speed tensor should not contain NaN values")
        
        # Check for Inf values
        self.assertFalse(torch.isinf(self.frames_tensor).any(), 
                         "Frames tensor should not contain Inf values")
        self.assertFalse(torch.isinf(self.steering_tensor).any(), 
                         "Steering tensor should not contain Inf values")
        self.assertFalse(torch.isinf(self.speed_tensor).any(), 
                         "Speed tensor should not contain Inf values")

    def test_padding(self):
        """Test that the data is padded correctly to the target length"""
        target_length = 600

        # Check padding
        self.assertEqual(self.frames_tensor.shape[0], target_length, "Frames should be padded to the target length")
        self.assertEqual(len(self.steering_tensor), target_length, "Steering should be padded to the target length")
        self.assertEqual(len(self.speed_tensor), target_length, "Speed should be padded to the target length")
    
    def test_speed(self):
        """Test how long it takes to preprocess a segment"""
        start_time = time.time()
        self.data_preprocessor.preprocess_segment(self.segment_data)
        end_time = time.time()
        print(f"Time taken to preprocess a segment: {end_time - start_time} seconds")

if __name__ == "__main__":
    unittest.main()