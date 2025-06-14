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
            
            # Call train_model with all the explicit parameters
            model = train_model(
                dataset_path=self.dataset_path,
                window_size=20,  # For 1s of driving data at 20fps
                target_length=1200,  # Length of each segment in frames
                stride=20,  # Non-overlapping windows
                batch_size=20,  # Batch size
                num_workers=2,  # Use two workers for tests
                num_epochs=5,  # Reduced for testing
                lr=0.0001,  # Learning rate
                img_size=(240, 320),  # Image dimensions
                frame_delay=2,  # Frames for T-100ms lookback
                future_steps=5,  # Number of future steps to predict
                future_step_size=2,  # Frames between future predictions
                fps=20,  # Original video frames per second
                debug=False  # Enable debugging output
            )
            
            self.assertIsNotNone(model, "Model should not be None after training.")
            saved_model_path = 'models/best_model.pth'
            self.assertTrue(os.path.exists(saved_model_path), "Best model file should be saved.")
            end_time = time.time()
            print(f"Training completed in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            self.fail(f"train_model raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()

