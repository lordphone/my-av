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
            for frame_idx in [0, 10, 20]:
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
        if self.reader.num_frames > 100:
            self.skipTest("Skipping full iteration test for long videos")
            
        frame_count = sum(1 for _ in self.reader)
        self.assertEqual(frame_count, self.reader.num_frames)

    def test_speed_and_frame_values(self):
        """Test the speed of the FrameReader and print numpy values for each frame."""
        import time

        start_time = time.time()
        frame_count = 0

        for frame in self.reader:
            # Print numpy values for the current frame
            print(f"Frame {frame_count}: {frame.flatten()[:10]}...")  # Print first 10 pixel values
            frame_count += 1

        elapsed_time = time.time() - start_time
        print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds.")

        # Ensure all frames were processed
        self.assertEqual(frame_count, self.reader.num_frames)


if __name__ == "__main__":
    unittest.main()