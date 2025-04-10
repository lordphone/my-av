# data_loader.py

from torch.utils.data import DataLoader
from src.data.comma2k19dataset import Comma2k19Dataset

def create_data_loader(data_path, batch_size=8, shuffle=True, num_workers=4):
    """
    Create a DataLoader for the Comma2k19 dataset.

    Args:
        data_path (str): Path to the dataset.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of worker threads for loading data.

    Returns:
        DataLoader: DataLoader for the Comma2k19 dataset, dataset for the data.
    """
    dataset = Comma2k19Dataset(data_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader, dataset