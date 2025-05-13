# train.py
# training script

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import random
import src.utils.data_utils as data_utils
import datetime

# for logging
import logging
import time
import gc

# Update logging configuration to save logs in a dedicated 'logs' directory
log_dir = "logs"  # Directory to save training logs
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    logging.info(f"Created log directory: {log_dir}")

log_filename = os.path.join(log_dir, f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# A handler to print logs to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

from torch.utils.data import DataLoader
from src.data.comma2k19dataset import Comma2k19Dataset
from src.data.processed_dataset import ProcessedDataset
from src.models.model import Model
from src.data.chunk_shuffling_sampler import ChunkShufflingSampler

from collections import OrderedDict

# Function to save checkpoints
def save_checkpoint(state, filename="checkpoint.pth"):
    """Save the model checkpoint."""
    logging.info(f"Saving checkpoint to {filename}")
    torch.save(state, filename)

def train_model(
    dataset_path, 
    checkpoint_dir="checkpoints", # Directory to save checkpoints
    resume_from=None, # Path to resume from checkpoint
    window_size=10, 
    batch_size=8, 
    num_epochs=30, 
    lr=0.001):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logging.info(f"Using device: {device}")

    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        logging.info(f"Created checkpoint directory: {checkpoint_dir}")

    # Create model directory if it doesn't exist
    model_dir = "models"  # Directory to save models
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logging.info(f"Created model directory: {model_dir}")

    # Load dataset
    base_dataset = Comma2k19Dataset(dataset_path)
    processed_dataset = ProcessedDataset(base_dataset, window_size=window_size)

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
    model = Model(window_size=window_size).to(device)

    # Define loss function and optimizer
    criterion_steering = nn.MSELoss()
    criterion_speed = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Reduced learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Loss weights
    steering_weight = 1.0
    speed_weight = 0.5

    # Gradient clipping
    max_grad_norm = 1.0

    # --- Checkpoint Loading ---
    start_epoch = 0
    best_val_loss = float('inf')
    if resume_from and os.path.isfile(resume_from):
        logging.info(f"Loading checkpoint from {resume_from}")
        # Load checkpoint on the CPU first
        checkpoint = torch.load(resume_from, map_location='cpu')

        # Load model state
        model_state_dict = checkpoint['model_state_dict']
        # If the checkpoint was saved with DataParallel, keys might start with 'module.'
        if all(key.startswith('module.') for key in model_state_dict.keys()):
            # Create a new state dict without 'module.' prefix
            new_state_dict = OrderedDict()
            for key, value in model_state_dict.items():
                name = key[7:]  # Remove 'module.' prefix
                new_state_dict[name] = value
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(model_state_dict)

        # Move model to the correct device after loading the state dict
        model.to(device)

        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Manually move the optimizer state to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
        # Load scheduler state
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load other training parameters
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
    else:
        if resume_from:
             logging.warning(f"Checkpoint file not found at '{resume_from}'. Starting training from scratch.")
        else:
             logging.info("No checkpoint specified. Starting training from scratch.")

    # Set log interval
    log_interval = 1000  # Log every 1000 batches

    # Training loop
    logging.info(f"Starting training for {num_epochs} epochs.")

    # --- Training Phase ---
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            frames = batch['frames'].to(device)  # Shape: [batch_size, window_length, 3, 160, 320]
            steering = batch['steering'].to(device)  # Shape: [batch_size, window_length]
            speed = batch['speed'].to(device)  # Shape: [batch_size, window_length]

            # Get only the last frame's ground truth values (current frame we're predicting)
            current_steering = steering[:, -1].unsqueeze(1)  # Shape: [batch_size, 1]
            current_speed = speed[:, -1].unsqueeze(1)  # Shape: [batch_size, 1]

            # Forward pass
            steering_pred, speed_pred = model(frames)

            # Compute loss - now comparing single predictions with the last frame's values
            loss_steering = criterion_steering(steering_pred, current_steering)
            loss_speed = criterion_speed(speed_pred, current_speed)
            loss = steering_weight * loss_steering + speed_weight * loss_speed
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            running_loss += loss.item()

            # Log progress every 1000 batches
            if (i + 1) % log_interval == 0:
                            batch_time = time.time()
                            batches_processed = i + 1
                            total_batches = len(train_loader)
                            time_per_batch = (batch_time - epoch_start_time) / batches_processed
                            eta_seconds = time_per_batch * (total_batches - batches_processed)
                            eta_formatted = str(datetime.timedelta(seconds=int(eta_seconds)))
                            logging.info(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batches_processed}/{total_batches}] - Loss: {loss.item():.4f} - ETA: {eta_formatted}")

        # Calculate average loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        # --- End of Training Phase ---

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data in val_loader:
                if data is None:
                    continue

                frames = data['frames'].to(device)
                steering = data['steering'].to(device)
                speed = data['speed'].to(device)

                # Get only the last frame's ground truth values
                current_steering = steering[:, -1].unsqueeze(1)
                current_speed = speed[:, -1].unsqueeze(1)

                steering_pred, speed_pred = model(frames)

                loss_steering = criterion_steering(steering_pred, current_steering)
                loss_speed = criterion_speed(speed_pred, current_speed)
                loss = steering_weight * loss_steering + speed_weight * loss_speed

                val_loss += loss.item()

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        # --- End of Validation Phase ---

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Print each epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f} seconds")
        print(f"Training Loss: {running_loss:.4f}, Validation Loss: {val_loss:.4f}")
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f} seconds")
        logging.info(f"Training Loss: {running_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # --- Checkpoint Saving ---
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(model_dir, 'best_model.pth')
            logging.info(f"New best model found! Saving to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)
        
        # Save the latest checkpoint including optimizer state etc.
        latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }, filename=latest_checkpoint_path)
        # --- End of Checkpoint Saving ---
        
    print("Training completed.")
    logging.info("Training completed.")
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a model with checkpointing.')
    parser.add_argument('--data_path', type=str, default="/home/lordphone/my-av/data/raw/comma2k19", help='Path to the dataset.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints.')
    parser.add_argument('--resume_from', type=str, default='checkpoints/latest_checkpoint.pth', help='Path to checkpoint file to resume training from (e.g., checkpoints/latest_checkpoint.pth).')
    parser.add_argument('--window_size', type=int, default=10, help='Sequence length for model input.')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size.')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')

    args = parser.parse_args()

    train_model(
        dataset_path=args.data_path,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        window_size=args.window_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr
    )
