import unittest
import time
import os
import torch

from src.training.train import train_model
from src.data.comma2k19dataset import Comma2k19Dataset

class TestTrainModel(unittest.TestCase):

    def setUp(self):
        # Set up temporary dataset path and mock data
        self.dataset_path = "/home/lordphone/my-av/tests/data"
        os.makedirs(self.dataset_path, exist_ok=True)

    def test_train(self):
        """Test if train_model runs without errors, and check if the model is saved."""
        try:
            dataset = Comma2k19Dataset(self.dataset_path)
            dataset_size = len(dataset)
            print(f"Training on {dataset_size} videos.")
            start_time = time.time()
            model = train_model(self.dataset_path, window_size=15, batch_size=8, num_epochs=2, lr=0.001)
            self.assertIsNotNone(model, "Model should not be None after training.")
            saved_model_path = os.path.join(self.dataset_path, 'best_model.pth')
            self.assertTrue(os.path.exists(saved_model_path), "Best model file should be saved.")
            end_time = time.time()
            print(f"Training completed in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            self.fail(f"train_model raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()

