import unittest
from src.data.processed_dataset import ProcessedDataset

class TestProcessedDataset(unittest.TestCase):
    def setUp(self):
        self. base_dataset_path = '/home/lordphone/my-av/tests/data'

    def test_get(self):
        windowed_data = ProcessedDataset(self.base_dataset_path, window_size=12)

        
if __name__ == '__main__':
    unittest.main()