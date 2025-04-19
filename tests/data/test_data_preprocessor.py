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
        self.result = self.data_preprocessor.preprocess_segment(self.segment_data, window_size=12, stride=12)

    def test_tensor_creation_and_shape(self):
        """Test that the preprocessor creates tensors with the correct shape"""
        # Check that frames, steering and speed are returned
        self.assertIn('frames', self.result[0], "Result should contain 'frames' key")
        self.assertIn('steering', self.result[0], "Result should contain 'steering' key")
        self.assertIn('speed', self.result[0], "Result should contain 'speed' key")
        
        # Check types
        self.assertIsInstance(self.result[0]['frames'], torch.Tensor, "Frames should be a PyTorch tensor")
        self.assertIsInstance(self.result[0]['steering'], torch.Tensor, "Steering should be a PyTorch tensor")
        self.assertIsInstance(self.result[0]['speed'], torch.Tensor, "Speed should be a PyTorch tensor")
        
        # Check dimensions
        self.assertEqual(len(self.result[0]['frames'].shape), 4, "Frames tensor should be 4-dimensional [batch, channels, height, width]")
        self.assertEqual(self.result[0]['frames'].shape[1], 3, "Frames should have 3 channels (RGB)")
        self.assertEqual(self.result[0]['frames'].shape[2:], (160, 320), "Frame dimensions should match the specified img_size")
        
        # Check that steering and speed are 1D and match the number of frames
        self.assertEqual(len(self.result[0]['steering'].shape), 1, "Steering tensor should be 1-dimensional")
        self.assertEqual(len(self.result[0]['speed'].shape), 1, "Speed tensor should be 1-dimensional")
        self.assertEqual(len(self.result[0]['steering']), len(self.result[0]['speed']), "Steering and speed should have the same length")
        self.assertEqual(self.result[0]['frames'].shape[0], len(self.result[0]['steering']), "Number of frames should match length of steering data")

    def test_normalization(self):
        """Test that the frames are normalized correctly"""
        # Check normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        # Denormalize the frames to check if values are within the expected range
        denormalized_frames = self.result[0]['frames'] * std + mean
        
        # Check that pixel values are within [0, 1] range
        self.assertTrue((denormalized_frames >= 0).all(), 
                "Denormalized frames should have values >= 0")
        self.assertTrue((denormalized_frames <= 1).all(), 
                "Denormalized frames should have values <= 1")

    def test_time_alignment(self):
        """Test that steering and speed data are properly time-aligned with frames"""
        # All three tensors should have matching first dimensions
        self.assertEqual(self.result[0]['frames'].shape[0], len(self.result[0]['steering']), 
                         "Number of frames should match length of steering data")
        self.assertEqual(self.result[0]['frames'].shape[0], len(self.result[0]['speed']), 
                         "Number of frames should match length of speed data")
        
        # The steering and speed should be aligned (same length)
        self.assertEqual(len(self.result[0]['steering']), len(self.result[0]['speed']), 
                         "Steering and speed tensors should have the same length")

    def test_data_dtype(self):
        """Test that the tensors have the correct data types"""
        # Check data types
        self.assertEqual(self.result[0]['frames'].dtype, torch.float32, 
                         "Frames tensor should be of type float32")
        self.assertEqual(self.result[0]['steering'].dtype, torch.float32, 
                         "Steering tensor should be of type float32")  
        self.assertEqual(self.result[0]['speed'].dtype, torch.float32, 
                         "Speed tensor should be of type float32")

    def test_data_validity(self):
        """Test that the tensors don't contain invalid values"""
        # Check for NaN values
        self.assertFalse(torch.isnan(self.result[0]['frames']).any(), 
                         "Frames tensor should not contain NaN values")
        self.assertFalse(torch.isnan(self.result[0]['steering']).any(), 
                         "Steering tensor should not contain NaN values")
        self.assertFalse(torch.isnan(self.result[0]['speed']).any(), 
                         "Speed tensor should not contain NaN values")
        
        # Check for Inf values
        self.assertFalse(torch.isinf(self.result[0]['frames']).any(), 
                         "Frames tensor should not contain Inf values")
        self.assertFalse(torch.isinf(self.result[0]['steering']).any(), 
                         "Steering tensor should not contain Inf values")
        self.assertFalse(torch.isinf(self.result[0]['speed']).any(), 
                         "Speed tensor should not contain Inf values")

    def test_windowed_data_structure(self):
        """Test that the preprocessor returns data in the correct windowed structure"""
        # Check that result is a list of dictionaries
        self.assertIsInstance(self.result, list, "Result should be a list of windows")
        self.assertGreater(len(self.result), 0, "Result should contain at least one window")

        # Check structure of each window
        for window in self.result:
            self.assertIn('frames', window, "Each window should contain 'frames' key")
            self.assertIn('steering', window, "Each window should contain 'steering' key")
            self.assertIn('speed', window, "Each window should contain 'speed' key")

            self.assertIsInstance(window['frames'], torch.Tensor, "Frames should be a PyTorch tensor")
            self.assertIsInstance(window['steering'], torch.Tensor, "Steering should be a PyTorch tensor")
            self.assertIsInstance(window['speed'], torch.Tensor, "Speed should be a PyTorch tensor")

            # Check dimensions of frames
            self.assertEqual(len(window['frames'].shape), 4, "Frames tensor should be 4-dimensional [batch, channels, height, width]")
            self.assertEqual(window['frames'].shape[1:], (3, 160, 320), "Frame dimensions should match the specified img_size")

            # Check dimensions of steering and speed
            self.assertEqual(len(window['steering'].shape), 1, "Steering tensor should be 1-dimensional")
            self.assertEqual(len(window['speed'].shape), 1, "Speed tensor should be 1-dimensional")
            self.assertEqual(len(window['steering']), len(window['speed']), "Steering and speed should have the same length")
            self.assertEqual(len(window['steering']), window['frames'].shape[0], "Number of frames should match length of steering data")

    def test_padding(self):
        """Test that the data is padded correctly to the target length"""
        target_length = 1200

        # Check padding in the first window
        total_frames = sum(len(self.result[i]['frames']) for i in range(len(self.result)))
        total_steering = sum(len(self.result[i]['steering']) for i in range(len(self.result)))
        total_speed = sum(len(self.result[i]['speed']) for i in range(len(self.result)))
        self.assertEqual(total_frames, target_length, "Frames should be padded to the target length")
        self.assertEqual(total_steering, target_length, "Steering should be padded to the target length")
        self.assertEqual(total_speed, target_length, "Speed should be padded to the target length")

if __name__ == "__main__":
    unittest.main()