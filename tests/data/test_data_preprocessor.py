import unittest
from src.data.data_preprocessor import DataPreprocessor
from src.data.data_loader import DataLoader

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        self.data_preprocessor = DataPreprocessor(img_size=(160, 320))
        self.segment_data = {
            'video_path': '/home/lordphone/my-av/tests/data/Example_1/b0c9d2329ad1606b|2018-08-02--08-34-47/40/video.mp4',
            'log_path': '/home/lordphone/my-av/tests/data/Example_1/b0c9d2329ad1606b|2018-08-02--08-34-47/40/processed_log',
            'pose_path': '/home/lordphone/my-av/tests/data/Example_1/b0c9d2329ad1606b|2018-08-02--08-34-47/40/global_pose'
        }

    def test_preprocess_segment(self):
        try:
            # Call the preprocess_segment method
            frames_tensor, steering_tensor, speed_tensor = self.data_preprocessor.preprocess_segment(self.segment_data)
            
            # Check the shapes of the returned tensors

            # Check that the tensors are not empty
            
        except Exception as e:
            self.fail(f"Preprocessing segment failed: {e}")

if __name__ == "__main__":
    unittest.main()