import unittest
import time
import torch

from src.data.comma2k19dataset import Comma2k19Dataset
from torch.utils.data import DataLoader
from src.data.processed_dataset import ProcessedDataset

class TestProcessedDataset(unittest.TestCase):
    def setUp(self):
        self.base_dataset_path = '/home/lordphone/my-av/tests/data'
        self.base_dataset = Comma2k19Dataset(self.base_dataset_path)
        self.batch_size = 8
        self.window_size = 20  # Updated to match the 1s of driving data requirement
        self.target_length = 1200
        print(f"Base dataset size: {len(self.base_dataset)}")

    def test_torch(self):
        processed_dataset = ProcessedDataset(self.base_dataset, window_size=self.window_size, target_length=self.target_length)
        self.assertGreater(len(processed_dataset), 0, "ProcessedDataset should not be empty")

        loader = DataLoader(processed_dataset, batch_size=self.batch_size, shuffle=True)
        print(f"Processed dataset size: {len(processed_dataset)}")

        start_time = time.time()
        for i, batch in enumerate(loader):
            self.assertIsNotNone(batch, "Batch should not be None")
            
            # Verify all required keys exist in the batch
            required_keys = ['frames', 'steering', 'speed', 'current_steering', 'current_speed',
                           'future_steering', 'future_speed']
            for key in required_keys:
                self.assertIn(key, batch, f"Batch should contain '{key}'")
            
            # Check batch shapes
            self.assertEqual(batch['frames'].shape[0], self.batch_size, "Batch size should match the DataLoader batch size")
            self.assertEqual(batch['frames'].shape[1], self.window_size, "Each batch should have the correct window size")
            
            # Check that frames have 6 channels (stacked current + T-100ms)
            self.assertEqual(batch['frames'].shape[2], 6, "Each frame should have 6 channels (stacked current + T-100ms)")
            
            # Check sequence dimensions
            self.assertEqual(batch['steering'].shape[1], self.window_size, "Steering data should match the window size")
            self.assertEqual(batch['speed'].shape[1], self.window_size, "Speed data should match the window size")
            
            # Check current values are single values
            self.assertEqual(batch['current_steering'].dim(), 1, "Current steering should be a 1D tensor")
            self.assertEqual(batch['current_steering'].shape[0], self.batch_size, "Current steering should have batch_size elements")
            self.assertEqual(batch['current_speed'].dim(), 1, "Current speed should be a 1D tensor")
            self.assertEqual(batch['current_speed'].shape[0], self.batch_size, "Current speed should have batch_size elements")
            
            # Check future predictions are the right shape (5 timesteps)
            self.assertEqual(batch['future_steering'].shape, (self.batch_size, 5), "Future steering should have 5 timesteps")
            self.assertEqual(batch['future_speed'].shape, (self.batch_size, 5), "Future speed should have 5 timesteps")
            
            # Verify that the current value (at T) is different from the first future value (at T+100ms)
            self.assertFalse(torch.all(torch.eq(batch['current_steering'].unsqueeze(1), batch['future_steering'][:, 0].unsqueeze(1))), 
                           "Current and future steering values should not be identical")
            
            print(f"Batch frames shape: {batch['frames'].shape}")
            print(f"Future steering shape: {batch['future_steering'].shape}")
            print(f"Future speed shape: {batch['future_speed'].shape}")
            break
            
        end_time = time.time()
        print(f"Data loading time: {end_time - start_time:.4f} seconds")
    
    def test_speed(self):
        """Test how long it takes to load a single segment or video"""
        processed_dataset = ProcessedDataset(self.base_dataset, window_size=self.window_size, target_length=self.target_length)
        start_time = time.time()
        processed_dataset[0]
        # for i in range(len(processed_dataset)):
        #     processed_dataset[i]
        end_time = time.time()
        print(f"Time taken to load a single segment: {end_time - start_time:.4f} seconds")

if __name__ == '__main__':
    unittest.main()