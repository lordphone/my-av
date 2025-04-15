import unittest
import torch
import os
import numpy as np
from src.data.data_preprocessor import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Set up test fixtures - run preprocessing once and store the results for all tests"""
        self.data_preprocessor = DataPreprocessor(img_size=(160, 320))
        self.segment_data = {
            'video_path': '/home/lordphone/my-av/tests/data/Example_1/b0c9d2329ad1606b|2018-08-02--08-34-47/40/video.mp4',
            'log_path': '/home/lordphone/my-av/tests/data/Example_1/b0c9d2329ad1606b|2018-08-02--08-34-47/40/processed_log',
            'pose_path': '/home/lordphone/my-av/tests/data/Example_1/b0c9d2329ad1606b|2018-08-02--08-34-47/40/global_pose'
        }
        
        # Process data once for all tests
        self.result = self.data_preprocessor.preprocess_segment(self.segment_data)
        self.frames = self.result['frames']
        self.steering = self.result['steering']
        self.speed = self.result['speed']

    def test_tensor_creation_and_shape(self):
        """Test that the preprocessor creates tensors with the correct shape"""
        # Check that frames, steering and speed are returned
        self.assertIn('frames', self.result, "Result should contain 'frames' key")
        self.assertIn('steering', self.result, "Result should contain 'steering' key")
        self.assertIn('speed', self.result, "Result should contain 'speed' key")
        
        # Check types
        self.assertIsInstance(self.frames, torch.Tensor, "Frames should be a PyTorch tensor")
        self.assertIsInstance(self.steering, torch.Tensor, "Steering should be a PyTorch tensor")
        self.assertIsInstance(self.speed, torch.Tensor, "Speed should be a PyTorch tensor")
        
        # Check dimensions
        self.assertEqual(len(self.frames.shape), 4, "Frames tensor should be 4-dimensional [batch, channels, height, width]")
        self.assertEqual(self.frames.shape[1], 3, "Frames should have 3 channels (RGB)")
        self.assertEqual(self.frames.shape[2:], (160, 320), "Frame dimensions should match the specified img_size")
        
        # Check that steering and speed are 1D and match the number of frames
        self.assertEqual(len(self.steering.shape), 1, "Steering tensor should be 1-dimensional")
        self.assertEqual(len(self.speed.shape), 1, "Speed tensor should be 1-dimensional")
        self.assertEqual(len(self.steering), len(self.speed), "Steering and speed should have the same length")
        self.assertEqual(self.frames.shape[0], len(self.steering), "Number of frames should match length of steering data")

    def test_normalization(self):
        """Test that the frames are normalized correctly"""
        # Check normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        # Denormalize the frames to check if values are within the expected range
        denormalized_frames = self.frames * std + mean
        
        # Check that pixel values are within [0, 1] range
        self.assertTrue((denormalized_frames >= 0).all(), 
                "Denormalized frames should have values >= 0")
        self.assertTrue((denormalized_frames <= 1).all(), 
                "Denormalized frames should have values <= 1")

    def test_time_alignment(self):
        """Test that steering and speed data are properly time-aligned with frames"""
        # All three tensors should have matching first dimensions
        self.assertEqual(self.frames.shape[0], len(self.steering), 
                         "Number of frames should match length of steering data")
        self.assertEqual(self.frames.shape[0], len(self.speed), 
                         "Number of frames should match length of speed data")
        
        # The steering and speed should be aligned (same length)
        self.assertEqual(len(self.steering), len(self.speed), 
                         "Steering and speed tensors should have the same length")

    def test_data_dtype(self):
        """Test that the tensors have the correct data types"""
        # Check data types
        self.assertEqual(self.frames.dtype, torch.float32, 
                         "Frames tensor should be of type float32")
        self.assertEqual(self.steering.dtype, torch.float32, 
                         "Steering tensor should be of type float32")  
        self.assertEqual(self.speed.dtype, torch.float32, 
                         "Speed tensor should be of type float32")

    def test_data_validity(self):
        """Test that the tensors don't contain invalid values"""
        # Check for NaN values
        self.assertFalse(torch.isnan(self.frames).any(), 
                         "Frames tensor should not contain NaN values")
        self.assertFalse(torch.isnan(self.steering).any(), 
                         "Steering tensor should not contain NaN values")
        self.assertFalse(torch.isnan(self.speed).any(), 
                         "Speed tensor should not contain NaN values")
        
        # Check for Inf values
        self.assertFalse(torch.isinf(self.frames).any(), 
                         "Frames tensor should not contain Inf values")
        self.assertFalse(torch.isinf(self.steering).any(), 
                         "Steering tensor should not contain Inf values")
        self.assertFalse(torch.isinf(self.speed).any(), 
                         "Speed tensor should not contain Inf values")
        
    def test_batch_size(self):
        """Test that the batch size is correct"""
        # Check that the batch size is 1200
        self.assertEqual(self.frames.shape[0], 1200, 
                         "Batch size should be 1 for single segment processing")

if __name__ == "__main__":
    unittest.main()