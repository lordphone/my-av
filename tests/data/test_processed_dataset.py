import unittest
import time

from src.data.comma2k19dataset import Comma2k19Dataset
from torch.utils.data import DataLoader
from src.data.processed_dataset import ProcessedDataset

class TestProcessedDataset(unittest.TestCase):
    def setUp(self):
        self.base_dataset_path = '/home/lordphone/my-av/tests/data'
        self.base_dataset = Comma2k19Dataset(self.base_dataset_path)
        self.batch_size = 8
        self.window_size = 15
        self.target_length = 600
        print(f"Base dataset size: {len(self.base_dataset)}")

    def test_torch(self):
        processed_dataset = ProcessedDataset(self.base_dataset, window_size=self.window_size, target_length=self.target_length)
        self.assertGreater(len(processed_dataset), 0, "ProcessedDataset should not be empty")

        loader = DataLoader(processed_dataset, batch_size=self.batch_size, shuffle=True)
        print(f"Processed dataset size: {len(processed_dataset)}")

        start_time = time.time()
        for i, batch in enumerate(loader):
            self.assertIsNotNone(batch, "Batch should not be None")
            self.assertIn('frames', batch, "Batch should contain 'frames'")
            self.assertIn('steering', batch, "Batch should contain 'steering'")
            self.assertIn('speed', batch, "Batch should contain 'speed'")
            self.assertEqual(batch['frames'].shape[0], self.batch_size, "Batch size should match the DataLoader batch size")
            self.assertEqual(batch['frames'].shape[1], self.window_size, "Each batch should have the correct window size")
            self.assertEqual(batch['steering'].shape[1], self.window_size, "Steering data should match the window size")
            self.assertEqual(batch['speed'].shape[1], self.window_size, "Speed data should match the window size")
            break
        end_time = time.time()
        print(f"Data loading time: {end_time - start_time:.4f} seconds")
if __name__ == '__main__':
    unittest.main()