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

# Set multiprocessing start method to 'spawn' for CUDA compatibility
# This must be done at the module level before creating any DataLoaders
torch.multiprocessing.set_start_method('spawn', force=True)

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
from torch.amp import GradScaler, autocast
from src.data.comma2k19dataset import Comma2k19Dataset
from src.data.processed_dataset import ProcessedDataset
from src.models.model import Model
from src.data.video_batch_sampler import VideoBatchSampler
from src.data.video_window_dataset import VideoWindowIterableDataset
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
    window_size=20,  # For 1s of driving data at 20fps
    target_length=1200,  # Length of each segment in frames
    stride=20,  # Non-overlapping windows
    batch_size=16, 
    num_epochs=30, 
    lr=0.0001,
    img_size=(240, 320),  # Image dimensions
    frame_delay=2,  # Frames for T-100ms lookback (2 for 100ms at 20fps)
    future_steps=5,  # Number of future steps to predict
    future_step_size=2,  # Frames between future predictions (2 = 100ms at 20fps)
    fps=20,  # Original video frames per second
    debug=False):  # Add debug flag
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    processed_dataset = ProcessedDataset(
        base_dataset, 
        window_size=window_size,
        target_length=target_length,
        stride=stride,
        img_size=img_size,
        frame_delay=frame_delay,
        future_steps=future_steps,
        future_step_size=future_step_size,
        fps=fps
    )

    # Group dataset by video
    video_indices = data_utils.get_video_indices(processed_dataset)  # Assuming this method exists
    random.seed(42)  # Fixed seed for reproducibility when using checkpoints
    random.shuffle(video_indices)  # Shuffle videos
    
    if debug:
        logging.info(f"Total number of videos: {len(video_indices)}")

    # Split videos into train and validation sets
    num_videos = len(video_indices)
    split_ratio = 0.8
    split_index = int(num_videos * split_ratio)
    train_videos_indices = video_indices[:split_index]
    val_videos_indices = video_indices[split_index:]
    
    if debug:
        logging.info(f"Train videos: {len(train_videos_indices)}, Val videos: {len(val_videos_indices)}")

    # Number of workers to use for data loading
    num_workers = 2  # Adjust based on your system's CPU cores
    
    # Create DataLoaders with batch samplers
    train_dataset = VideoWindowIterableDataset(processed_dataset, video_indices=train_videos_indices, shuffle=True)
    val_dataset = VideoWindowIterableDataset(processed_dataset, video_indices=val_videos_indices, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # Now works as expected
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,
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

    # Initialize model with appropriate parameters for the new structure
    # window_size=19 for 1 second of driving data
    num_future_predictions = 5  # Number of future timesteps to predict
    num_vehicle_state_features = 2  # Speed and steering
    model = Model(
        num_future_predictions=num_future_predictions,
        num_vehicle_state_features=num_vehicle_state_features,
    ).to(device)

    # Define loss function and optimizer
    criterion_steering = nn.MSELoss()
    criterion_speed = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)  # Reduced learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Loss weights
    steering_weight = 1.0
    speed_weight = 0.5

    # Gradient clipping
    max_grad_norm = 1.0

    # Mixed precision training
    scaler = GradScaler()

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

    # Log training start
    if debug:
        logging.info(f"Starting training with {num_workers} workers")

    # --- Training Phase ---
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0

        # For profiling data loading vs model training
        total_data_loading_time = 0
        total_model_time = 0

        for i, batch in enumerate(train_loader):
            # Start timing data loading
            data_loading_end_time = time.time()
            
            frames = batch['frames'].to(device)  # Shape: [batch_size, window_size, 6, H, W]
            steering = batch['steering'].to(device)  # Shape: [batch_size, window_size]
            speed = batch['speed'].to(device)  # Shape: [batch_size, window_size]
            
            # Create vehicle state tensor by combining speed and steering
            veh_states = torch.stack([speed, steering], dim=2)  # Shape: [batch_size, window_size, 2]

            # Data loading finished, now start timing model computation
            model_start_time = time.time()

            # Reset hidden state for each new batch to avoid information leakage between sequences
            hidden_state = None
                
            # Forward pass with mixed precision
            with autocast(device_type='cuda'):
                # Pass both frames and vehicle states to the model
                # The model now expects both frame data and vehicle state data
                steering_pred, speed_pred, _ = model(frames, veh_states, hidden_state)
                
                # Get the future ground truth values from the dataset
                # These are 5 future values (T+100ms to T+500ms) that we'll use as targets
                future_steering_targets = batch['future_steering'].to(device)  # [batch_size, 5]
                future_speed_targets = batch['future_speed'].to(device)  # [batch_size, 5]
                
                loss_steering = criterion_steering(steering_pred, future_steering_targets)
                loss_speed = criterion_speed(speed_pred, future_speed_targets)
                loss = steering_weight * loss_steering + speed_weight * loss_speed

            # Backward pass and optimization with scaler
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # Apply gradient clipping to unscaled gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            # End of model computation timing
            model_end_time = time.time()
            
            # For the first batch, data loading time includes initialization
            if i == 0:
                next_data_loading_start_time = model_end_time
            
            # Calculate times for subsequent batches
            if i > 0:
                data_loading_time = data_loading_end_time - next_data_loading_start_time
                total_data_loading_time += data_loading_time
            
            model_time = model_end_time - model_start_time
            total_model_time += model_time
            
            # Set the start time for the next batch data loading
            next_data_loading_start_time = model_end_time

            running_loss += loss.item()

            # Log progress every 1000 batches
            if (i + 1) % log_interval == 0:
                batch_time = time.time()
                batches_processed = i + 1
                total_batches = len(train_loader)
                time_per_batch = (batch_time - epoch_start_time) / batches_processed
                eta_seconds = time_per_batch * (total_batches - batches_processed)
                eta_formatted = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                # Calculate current average loss
                current_avg_loss = running_loss / batches_processed
                
                if debug and i > 0:
                    avg_data_loading_time = total_data_loading_time / i
                    avg_model_time = total_model_time / (i + 1)
                    logging.info(f"Avg data loading time: {avg_data_loading_time:.4f}s, Avg model time: {avg_model_time:.4f}s")
                
                logging.info(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batches_processed}/{total_batches}] - Batch Loss: {loss.item():.4f} - Avg Loss: {current_avg_loss:.4f} - ETA: {eta_formatted}")

        # End of epoch timing summary
        if debug:
            total_batches = len(train_loader)
            if total_batches > 1:
                avg_data_loading_time = total_data_loading_time / (total_batches - 1)  # Exclude first batch
                avg_model_time = total_model_time / total_batches
                data_percentage = (avg_data_loading_time / (avg_data_loading_time + avg_model_time)) * 100
                model_percentage = (avg_model_time / (avg_data_loading_time + avg_model_time)) * 100
                
                logging.info(f"Epoch time breakdown - Data loading: {avg_data_loading_time:.4f}s ({data_percentage:.1f}%), Model: {avg_model_time:.4f}s ({model_percentage:.1f}%)")

        # Calculate average loss for the epoch
        avg_train_loss = running_loss / len(train_loader) / batch_size
        
        # Decompose losses for more detailed monitoring
        with torch.no_grad():
            model.eval()
            detailed_samples = min(100, len(train_loader))  # Limit to 100 batches for speed
            steering_loss_sum = 0.0
            speed_loss_sum = 0.0
            
            for i, batch in enumerate(train_loader):
                if i >= detailed_samples:
                    break
                    
                frames = batch['frames'].to(device)  # Shape: [batch_size, window_size, 6, H, W]
                steering = batch['steering'].to(device)  # Shape: [batch_size, window_size]
                speed = batch['speed'].to(device)  # Shape: [batch_size, window_size]
                
                # Get future ground truth values
                future_steering_targets = batch['future_steering'].to(device)  # [batch_size, 5]
                future_speed_targets = batch['future_speed'].to(device)  # [batch_size, 5]
                
                # Create vehicle state tensor
                veh_states = torch.stack([speed, steering], dim=2)  # Shape: [batch_size, window_size, 2]
                
                # Reset hidden state for each evaluation batch
                hidden_state = None
                
                # Forward pass with the new model format
                steering_pred, speed_pred, _ = model(frames, veh_states, hidden_state)
                
                # Get individual losses using the future predictions
                steering_loss = criterion_steering(steering_pred, future_steering_targets).item()
                speed_loss = criterion_speed(speed_pred, future_speed_targets).item()
                
                steering_loss_sum += steering_loss
                speed_loss_sum += speed_loss
            
            avg_steering_loss = steering_loss_sum / detailed_samples
            avg_speed_loss = speed_loss_sum / detailed_samples
        
        # Log individual loss components
        logging.info(f"Component Losses - Steering: {avg_steering_loss:.4f}, Speed: {avg_speed_loss:.4f}")
        # --- End of Training Phase ---

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data in val_loader:
                if data is None:
                    continue

                frames = data['frames'].to(device)  # Shape: [batch_size, window_size, 6, H, W]
                steering = data['steering'].to(device)  # Shape: [batch_size, window_size]
                speed = data['speed'].to(device)  # Shape: [batch_size, window_size]
                
                # Create vehicle state tensor by combining speed and steering
                veh_states = torch.stack([speed, steering], dim=2)  # Shape: [batch_size, window_size, 2]

                # Reset hidden state for each validation batch
                hidden_state = None

                # Pass both frames and vehicle states to the model
                steering_pred, speed_pred, _ = model(frames, veh_states, hidden_state)
                
                # Use the actual future ground truth values from the dataset
                future_steering_targets = data['future_steering'].to(device)  # [batch_size, 5]
                future_speed_targets = data['future_speed'].to(device)  # [batch_size, 5]

                loss_steering = criterion_steering(steering_pred, future_steering_targets)
                loss_speed = criterion_speed(speed_pred, future_speed_targets)
                loss = steering_weight * loss_steering + speed_weight * loss_speed

                val_loss += loss.item()

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader) / batch_size
        # --- End of Validation Phase ---

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Log epoch summary
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f} seconds")
        logging.info(f"Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")
        logging.info(f"Total Train Loss: {running_loss:.4f}, Total Val Loss: {val_loss:.4f}")

        # --- Checkpoint Saving ---
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_model_path = os.path.join(model_dir, 'best_model.pth')
            logging.info(f"New best model found! Avg Val Loss: {avg_val_loss:.4f} (previous best: {best_val_loss if epoch > 0 else 'N/A'})")
            best_val_loss = avg_val_loss
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
        
    logging.info("Training completed.")
    return model

if __name__ == "__main__":
    import argparse
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Train the model in different modes')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help='Training mode: "train" for real training, "test" for test training')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train (default: 30 for train, 5 for test)')
    parser.add_argument('--batch-size', type=int, default=20, help='Batch size for training')
    parser.add_argument('--window-size', type=int, default=20, help='Window size for temporal data')
    
    args = parser.parse_args()
    
    # Set default epochs based on mode if not specified
    if args.epochs is None:
        if args.mode == 'train':
            args.epochs = 50
        else:  # test mode
            args.epochs = 5
    
    # Common parameters for both modes
    common_params = {
        'debug': args.debug,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'window_size': args.window_size
    }
    
    if args.mode == 'train':
        # Real training mode - use full dataset
        logging.info(f"Starting real training mode with full dataset (epochs: {args.epochs})")
        
        # Check for existing checkpoint
        checkpoint_dir = "checkpoints"
        latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        resume_from = latest_checkpoint_path if os.path.exists(latest_checkpoint_path) else None
        
        if resume_from:
            logging.info(f"Resuming training from checkpoint: {resume_from}")
        else:
            logging.info("No checkpoint found. Starting training from scratch.")
        
        train_model(
            dataset_path="/home/lordphone/my-av/data/raw/comma2k19",
            resume_from=resume_from,
            **common_params
        )
    else:
        # Test training mode - use test dataset
        logging.info(f"Starting test training mode with test dataset (epochs: {args.epochs})")
        train_model(
            dataset_path="/home/lordphone/my-av/tests/data",
            **common_params
        )
