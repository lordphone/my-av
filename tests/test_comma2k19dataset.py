import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.comma2k19dataset import Comma2k19Dataset

# Define the base path to your dataset
BASE_PATH = "/home/lordphone/my-av/data/raw/comma2k19"

def test_dataset():
    # Initialize the dataset
    dataset = Comma2k19Dataset(base_path=BASE_PATH)

    # Check the number of samples
    print(f"Number of samples: {len(dataset)}")

    # Try loading the first sample
    if len(dataset) > 0:
        sample = dataset[0]
        print("First sample loaded successfully:")
        print(f"Image: {sample['image']}")
        print(f"Log Data: {sample['log_data']}")
        print(f"Position Data: {sample['position_data']}")
        print(f"Metadata: {sample['metadata']}")
    else:
        print("No samples found in the dataset.")

if __name__ == "__main__":
    test_dataset()