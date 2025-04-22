# train.py
# training script

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import random
import src.utils.data_utils as data_utils

from torch.utils.data import DataLoader
from src.data.comma2k19dataset import Comma2k19Dataset
from src.data.processed_dataset import ProcessedDataset
from src.models.model import Model
from src.data.chunk_shuffling_sampler import ChunkShufflingSampler

def train_model(dataset_path, batch_size=8, num_epochs=50, lr=0.001):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    base_dataset = Comma2k19Dataset(dataset_path)
    processed_dataset = ProcessedDataset(base_dataset)

    # Group dataset by video
    video_indices = data_utils.get_video_indices(processed_dataset)  # Assuming this method exists
    random.shuffle(video_indices)  # Shuffle videos
    # print(f"Video indices: {video_indices}")  # Print video indices

    # Split videos into train and validation sets
    num_videos = len(video_indices)
    split_ratio = 0.8
    split_index = int(num_videos * split_ratio)
    train_videos_indices = video_indices[:split_index]
    val_videos_indices = video_indices[split_index:]
    # print(f"Train videos: {train_videos_indices}")  # Print train videos
    # print(f"Validation videos: {val_videos_indices}")  # Print validation videos

    """ Unfinished implementation of seperating dataset into train and validation sets, for now feeding the entire dataset into the model"""
    # # Create a mapping of video indices to dataset indices
    # train_indices = [idx for indices in data_utils.group_by_video(processed_dataset, train_videos_indices).values() for idx in indices]
    # val_indices = [idx for indices in data_utils.group_by_video(processed_dataset, val_videos_indices).values() for idx in indices]
    # print(f"Train indices: {train_indices}")  # Print train indices
    # print(f"Validation indices: {val_indices}")  # Print validation indices

    # # Create Subset datasets
    # train_dataset = torch.utils.data.Subset(processed_dataset, train_indices)
    # val_dataset = torch.utils.data.Subset(processed_dataset, val_indices)
    # print(f"Train dataset: {train_dataset}")  # Print train dataset
    # print(f"Validation dataset: {val_dataset}")  # Print validation dataset
    # print(f"Training dataset size: {len(train_dataset)}")
    # print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        processed_dataset, 
        batch_size=batch_size, 
        sampler=ChunkShufflingSampler(processed_dataset, video_indices=train_videos_indices, shuffle=True), 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        processed_dataset, 
        batch_size=batch_size, 
        sampler=ChunkShufflingSampler(processed_dataset, video_indices=val_videos_indices, shuffle=True), 
        num_workers=4, 
        pin_memory=True
    )

    """ Uncomment to check dataset sizes and batch shapes"""
    # # Print dataset sizes
    # print(f"Training dataset size: {len(train_dataset)}")
    # print(f"Validation dataset size: {len(val_dataset)}")
    # for batch in train_loader:
    #     if batch is None:
    #         print("Batch is None, skipping...")
    #         continue
    #     frames = batch['frames']
    #     steering = batch['steering']
    #     speed = batch['speed']
    #     print(f"Frames shape: {frames.shape}, Steering shape: {steering.shape}, Speed shape: {speed.shape}")
    #     break
    # for i, batch in enumerate(train_loader):
    #     print(f"Batch {i}: {batch['frames'].shape}, {batch['steering'].shape}, {batch['speed'].shape}")

    # Initialize model
    model = Model().to(device)

    # Define loss function and optimizer
    criterion_steering = nn.MSELoss()
    criterion_speed = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for i, batch in enumerate(train_loader):
            frames = batch['frames'].to(device)  # Shape: [batch_size, 12, 3, 160, 320]
            steering = batch['steering'].to(device)  # Shape: [batch_size, 12]
            speed = batch['speed'].to(device)  # Shape: [batch_size, 12]

            # Forward pass
            steering_pred, speed_pred = model(frames)

            # Compute loss
            loss_steering = criterion_steering(steering_pred, steering)
            loss_speed = criterion_speed(speed_pred, speed)
            loss = loss_steering + loss_speed
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        # Calculate average loss for the epoch
        running_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data in val_loader:
                if data is None:
                    continue

                frames = data['frames'].to(device)
                steering = data['steering'].to(device)
                speed = data['speed'].to(device)

                steering_pred, speed_pred = model(frames)

                loss_steering = criterion_steering(steering_pred, steering)
                loss_speed = criterion_speed(speed_pred, speed)
                loss = loss_steering + loss_speed

                val_loss += loss.item()

        # Calculate average validation loss
        val_loss /= len(val_loader)

        # Update learning rate
        scheduler.step(val_loss)

        # Print each epoch summary
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f} seconds")
        print(f"Training Loss: {running_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(dataset_path, 'best_model.pth'))
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")
        
    print("Training completed.")
    return model

if __name__ == "__main__":

    # Example usage on placeholder dataset path
    dataset_path = "/home/lordphone/my-av/data/raw/comma2k19"
    train_model(dataset_path)