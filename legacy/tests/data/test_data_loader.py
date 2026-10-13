import unittest
from pathlib import Path
from src.data.data_loader import create_data_loader
from src.data.comma2k19dataset import Comma2k19Dataset
from torch.utils.data import DataLoader

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Set up a dataset path relative to this file
        self.data_path = Path(__file__).resolve().parent / "sample_dataset"
        self.batch_size = 4
        self.shuffle = False
        self.num_workers = 2

    def test_create_data_loader(self):
        # Test if the DataLoader and dataset are created correctly
        data_loader, dataset = create_data_loader(
            self.data_path, self.batch_size, self.shuffle, self.num_workers
        )

        self.assertIsInstance(data_loader, DataLoader)
        self.assertIsInstance(dataset, Comma2k19Dataset)
        self.assertEqual(len(data_loader), len(dataset) // self.batch_size)

    def test_data_loader_batch_size(self):
        # Test if the DataLoader respects the batch size
        data_loader, _ = create_data_loader(
            self.data_path, self.batch_size, self.shuffle, self.num_workers
        )

        for batch in data_loader:
            self.assertEqual(len(batch), self.batch_size)

if __name__ == "__main__":
    unittest.main()
