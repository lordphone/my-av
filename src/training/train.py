# train.py
# training script

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.comma2k19dataset import Comma2k19Dataset
from src.data.processed_dataset import ProcessedDataset
from src.models.model import Model
import os
import time

def train_model(dataset_path, batch_size=1, num_epochs=50, lr=0.001):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    base_dataset = Comma2k19Dataset(dataset_path)
    processed_dataset = ProcessedDataset(base_dataset)

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(processed_dataset))
    val_size = len(processed_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(processed_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # # Print examples from train_loader and val_loader
    # print("Examples from train_loader:")
    # for i, data in enumerate(train_loader):
    #     if i >= 2:  # Print only the first 2 batches
    #         break
    #     print(f"Batch {i+1}:")
    #     print(f"Frames shape: {data['frames'].shape}")
    #     print(f"Steering shape: {data['steering'].shape}")
    #     print(f"Speed shape: {data['speed'].shape}")
    #     print()

    #     print("Examples from val_loader:")
    # for i, data in enumerate(val_loader):
    #     if i >= 2:  # Print only the first 2 batches
    #         break
    #     print(f"Batch {i+1}:")
    #     print(f"Frames shape: {data['frames'].shape}")
    #     print(f"Steering shape: {data['steering'].shape}")
    #     print(f"Speed shape: {data['speed'].shape}")
    #     print()

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

        for i, data in enumerate(train_loader):
            if data is None:
                continue

            frames = data['frames'].to(device)
            steering = data['steering'].to(device)
            speed = data['speed'].to(device)

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

            # Print training progress
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            
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
    dataset_path = "/home/lordphone/my-av/tests/data"
    train_model(dataset_path)