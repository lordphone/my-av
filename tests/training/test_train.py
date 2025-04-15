import unittest
import os
from src.training.train import train_model
import torch

class TestTrainModel(unittest.TestCase):

    def setUp(self):
        # Set up temporary dataset path and mock data
        self.dataset_path = "/home/lordphone/my-av/tests/data"
        os.makedirs(self.dataset_path, exist_ok=True)

    def test_train_model_runs(self):
        """Test if train_model runs without errors."""
        try:
            model = train_model(self.dataset_path, batch_size=2, num_epochs=1, lr=0.001)
            self.assertIsNotNone(model, "Model should not be None after training.")
        except Exception as e:
            self.fail(f"train_model raised an exception: {e}")

    def test_model_saving(self):
        """Test if the best model is saved correctly."""
        train_model(self.dataset_path, batch_size=2, num_epochs=1, lr=0.001)
        saved_model_path = os.path.join(self.dataset_path, 'best_model.pth')
        self.assertTrue(os.path.exists(saved_model_path), "Best model file should be saved.")

if __name__ == "__main__":
    unittest.main()

