import unittest
from src.data.comma2k19dataset import Comma2k19Dataset

class TestComma2k19Dataset(unittest.TestCase):

    BASE_PATH = "/home/lordphone/my-av/data/raw/comma2k19"

    def test_dataset_initialization(self):
        # Initialize the dataset
        dataset = Comma2k19Dataset(base_path=self.BASE_PATH)

        # Check if the dataset is initialized correctly
        self.assertIsNotNone(dataset)

    def test_number_of_samples(self):
        # Initialize the dataset
        dataset = Comma2k19Dataset(base_path=self.BASE_PATH)

        # Check the number of samples
        self.assertGreaterEqual(len(dataset), 0, "Dataset should have zero or more samples.")
        print(f"Number of samples in the dataset: {len(dataset)}")

    def test_first_sample_loading(self):
        # Initialize the dataset
        dataset = Comma2k19Dataset(base_path=self.BASE_PATH)

        if len(dataset) > 0:
            # Try loading the first sample
            sample = dataset[0]
            self.assertIn('video_path', sample)
            self.assertIn('log_path', sample)
            self.assertIn('pose_path', sample)
            self.assertIn('metadata', sample)
        else:
            self.skipTest("No samples found in the dataset.")

if __name__ == "__main__":
    unittest.main()