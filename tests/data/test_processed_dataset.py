import unittest
import time

from src.data.comma2k19dataset import Comma2k19Dataset
from torch.utils.data import DataLoader
from src.data.processed_dataset import ProcessedDataset

class TestProcessedDataset(unittest.TestCase):
    def setUp(self):
        self.base_dataset_path = '/home/lordphone/my-av/tests/data'
        self.base_dataset = Comma2k19Dataset(self.base_dataset_path)
        self.batch_size = 4
        print(f"Base dataset size: {len(self.base_dataset)}")

    def test_torch(self):
        processed_dataset = ProcessedDataset(self.base_dataset, window_size=12)
        self.assertGreater(len(processed_dataset), 0, "ProcessedDataset should not be empty")
        
        loader = DataLoader(processed_dataset, batch_size=self.batch_size, shuffle=True)
        print(f"Processed dataset size: {len(processed_dataset)}")

        start_time = time.time()
        for i, batch in enumerate(loader):
            # print(f"Batch {i}:")
            # batch_size = batch['frames'].shape[0]
            # print(f"Batch size: {batch_size}")
            # print(f"Batch frames: {batch['frames'].shape}")
            # print(f"Batch steering: {batch['steering'].shape}")
            # print(f"Batch speed: {batch['speed'].shape}")
            
            self.assertIsNotNone(batch, "Batch should not be None")
            self.assertIn('frames', batch, "Batch should contain 'frames'")
            self.assertIn('steering', batch, "Batch should contain 'steering'")
            self.assertIn('speed', batch, "Batch should contain 'speed'")
            self.assertEqual(batch['frames'].shape[0], self.batch_size, "Batch size should match the DataLoader batch size")
            break
        end_time = time.time()
        print(f"Data loading time: {end_time - start_time:.4f} seconds")
if __name__ == '__main__':
    unittest.main()