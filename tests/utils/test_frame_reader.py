import sys
import os
import unittest
from src.utils.frame_reader import FrameReader


class TestFrameReader(unittest.TestCase):
    def setUp(self):
        self.video_path = "/home/lordphone/my-av/tests/data/Example_1/b0c9d2329ad1606b|2018-08-02--08-34-47/40/video.mp4"
        self.reader = FrameReader(self.video_path)
    
    def test_basic_functionality(self):
        """Test the basic functionality of the FrameReader."""
        try:
            # Verify video metadata is accessible
            self.assertGreater(self.reader.width, 0)
            self.assertGreater(self.reader.height, 0)
            self.assertGreater(self.reader.fps, 0)
            
            # Read a few frames and check their shapes
            for frame_idx in [0, 50, 100]:
                if frame_idx < self.reader.num_frames:
                    frame = self.reader.get(frame_idx)
                    self.assertEqual(len(frame.shape), 3)
                    self.assertEqual(frame.shape[0], self.reader.height)
                    self.assertEqual(frame.shape[1], self.reader.width)
                    self.assertEqual(frame.shape[2], 3)  # RGB channels
                    
                    # Check that frame data is not empty or all zeros
                    self.assertGreater(frame.sum(), 0)
        except Exception as e:
            self.fail(f"Basic FrameReader test failed: {e}")
    
    def test_iterator_functionality(self):
        """Test that the FrameReader can be iterated over."""
        frame_count = 0
        for frame in self.reader:
            # Check that each frame is a numpy array with the expected shape
            self.assertEqual(len(frame.shape), 3)
            self.assertEqual(frame.shape[0], self.reader.height)
            self.assertEqual(frame.shape[1], self.reader.width)
            self.assertEqual(frame.shape[2], 3)  # RGB channels
            
            frame_count += 1
            # Limit to first 5 frames to avoid long test times
            if frame_count >= 5:
                break
                
        # Verify we were able to get at least some frames
        self.assertGreater(frame_count, 0)
    
    def test_len_method(self):
        """Test the __len__ method returns the correct number of frames."""
        num_frames = len(self.reader)
        self.assertGreater(num_frames, 0)
        self.assertEqual(num_frames, self.reader.num_frames)
    
    def test_reset_method(self):
        """Test the reset method resets the iterator."""
        # Get first frame
        first_frame_initial = next(iter(self.reader))
        
        # Advance iterator
        next(iter(self.reader))
        next(iter(self.reader))
        
        # Reset iterator
        self.reader.reset()
        
        # Get first frame again
        first_frame_after_reset = next(iter(self.reader))
        
        # Compare pixel values of first 100 pixels to verify it's the same frame
        self.assertTrue(
            (first_frame_initial.flatten()[:100] == first_frame_after_reset.flatten()[:100]).all()
        )
    
    def test_iteration_completeness(self):
        """Test that iteration covers all frames in the video."""
        frame_count = sum(1 for _ in self.reader)
        self.assertEqual(frame_count, self.reader.num_frames)

    def test_get_frames_method(self):
        """Test the get_frames method for retrieving multiple consecutive frames."""
        try:
            # Test retrieving a small number of frames
            start_idx = 10
            num_frames = 25
            frames = self.reader.get_frames(start_idx, num_frames)
            
            # Verify the shape of the returned frames
            self.assertEqual(frames.shape, (num_frames, self.reader.height, self.reader.width, 3))
            
            # Verify that the frames are not empty or all zeros
            self.assertGreater(frames.sum(), 0)
            
            # Verify that the frames are consecutive
            for i in range(num_frames - 1):
                self.assertFalse((frames[i] == frames[i + 1]).all())
            
            # Test retrieving frames near the end of the video
            start_idx = max(0, self.reader.num_frames - 10)
            num_frames = 10
            frames = self.reader.get_frames(start_idx, num_frames)
            
            # Verify the shape of the returned frames
            expected_num_frames = min(num_frames, self.reader.num_frames - start_idx)
            self.assertEqual(frames.shape, (expected_num_frames, self.reader.height, self.reader.width, 3))
            
            # Verify that the frames are not empty or all zeros
            self.assertGreater(frames.sum(), 0)
        except Exception as e:
            self.fail(f"get_frames method test failed: {e}")


if __name__ == "__main__":
    unittest.main()