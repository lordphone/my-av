# train.py
# training script
import os
import logging
import datetime
import random
import gc
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Set multiprocessing start method to 'spawn' for CUDA compatibility
# This must be done at the module level before creating any DataLoaders
torch.multiprocessing.set_start_method('spawn', force=True)

# Import other modules after setting multiprocessing method
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from src.data.comma2k19dataset import Comma2k19Dataset
from src.data.processed_dataset import ProcessedDataset
from src.models.model import Model
from src.data.video_batch_sampler import VideoBatchSampler
from src.data.video_window_dataset import VideoWindowIterableDataset
from collections import OrderedDict
import src.utils.data_utils as data_utils

def setup_logging(log_dir="logs"):
    """Initialize logging configuration."""
    # Ensure logs directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")),
            logging.StreamHandler()
        ],
        force=True  # Override any existing handlers
    )
    
    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")
    logger.info(f"Starting training script at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return logger

# Function to save checkpoints
def save_checkpoint(state, filename="checkpoint.pth"):
    """Save the model checkpoint."""
    logging.info(f"Saving checkpoint to {filename}")
    torch.save(state, filename)

def calculate_dynamic_weights(loss_history, starting_steering_weight, starting_speed_weight, smoothing_factor=0.1, steering_importance=2.0, speed_importance=1.0, max_ratio=10.0):
    """
    Calculate dynamic weights based on loss history with normalization and importance preferences.
    
    Args:
        loss_history: List of dictionaries containing 'steering_loss' and 'speed_loss' for each epoch
        starting_steering_weight: Initial steering weight (used as default and for target sum)
        starting_speed_weight: Initial speed weight (used as default and for target sum)
        smoothing_factor: Factor for smoothing the weight updates (0 = no change, 1 = full change)
        steering_importance: Relative importance of steering (default: 2.0, meaning 2x more important than speed)
        speed_importance: Relative importance of speed (default: 1.0, baseline importance)
        max_ratio: Maximum ratio between weights (default: 10.0, meaning one weight can't be more than 10x the other)
    
    Returns:
        tuple: (steering_weight, speed_weight)
    """
    # Target sum for normalized weights based on starting weights
    TARGET_WEIGHT_SUM = starting_steering_weight + starting_speed_weight
    
    if not loss_history:
        # Return starting weights if no history
        return starting_steering_weight, starting_speed_weight
    
    # Use the last 2 epochs if available, otherwise just the last epoch
    recent_losses = loss_history[-2:] if len(loss_history) >= 2 else loss_history[-1:]
    
    # Calculate average losses over recent epochs
    avg_steering_loss = sum(epoch['steering_loss'] for epoch in recent_losses) / len(recent_losses)
    avg_speed_loss = sum(epoch['speed_loss'] for epoch in recent_losses) / len(recent_losses)
    
    # Prevent division by zero
    if avg_steering_loss == 0 or avg_speed_loss == 0:
        return starting_steering_weight, starting_speed_weight
    
    # Calculate weights directly proportional to loss magnitudes
    # Higher loss gets higher weight to focus more on that component
    direct_steering_loss = avg_steering_loss
    direct_speed_loss = avg_speed_loss
    
    # Apply importance preferences to bias the weights
    # Higher importance means the loss gets more weight even when balanced
    biased_steering_weight = direct_steering_loss * steering_importance
    biased_speed_weight = direct_speed_loss * speed_importance
    
    # Calculate raw weights (normalized to sum to 1)
    total_biased = biased_steering_weight + biased_speed_weight
    raw_steering_weight = biased_steering_weight / total_biased
    raw_speed_weight = biased_speed_weight / total_biased
    
    # Scale to target sum to maintain consistent magnitude
    steering_weight = raw_steering_weight * TARGET_WEIGHT_SUM
    speed_weight = raw_speed_weight * TARGET_WEIGHT_SUM
    
    # Apply smoothing to prevent dramatic weight changes
    if len(loss_history) > 1:
        prev_steering_weight = loss_history[-1].get('steering_weight', starting_steering_weight)
        prev_speed_weight = loss_history[-1].get('speed_weight', starting_speed_weight)
        
        steering_weight = (1 - smoothing_factor) * prev_steering_weight + smoothing_factor * steering_weight
        speed_weight = (1 - smoothing_factor) * prev_speed_weight + smoothing_factor * speed_weight
    
    # Normalize again after smoothing to ensure consistent sum
    current_sum = steering_weight + speed_weight
    steering_weight = (steering_weight / current_sum) * TARGET_WEIGHT_SUM
    speed_weight = (speed_weight / current_sum) * TARGET_WEIGHT_SUM
    
    # Apply relative ratio constraints to prevent one weight from dominating
    # Ensure steering_weight / speed_weight and speed_weight / steering_weight are both <= max_ratio
    
    # First normalize to target sum
    current_sum = steering_weight + speed_weight
    if current_sum != TARGET_WEIGHT_SUM:
        steering_weight = (steering_weight / current_sum) * TARGET_WEIGHT_SUM
        speed_weight = (speed_weight / current_sum) * TARGET_WEIGHT_SUM
    
    # Check and enforce ratio constraints
    if steering_weight > 0 and speed_weight > 0:
        steering_to_speed_ratio = steering_weight / speed_weight
        speed_to_steering_ratio = speed_weight / steering_weight
        
        # If steering is more than max_ratio times speed, clamp it
        if steering_to_speed_ratio > max_ratio:
            steering_weight = speed_weight * max_ratio
            logging.info(f"Clamped steering weight: ratio was {steering_to_speed_ratio:.2f}, max allowed is {max_ratio:.2f}")
        
        # If speed is more than max_ratio times steering, clamp it  
        elif speed_to_steering_ratio > max_ratio:
            speed_weight = steering_weight * max_ratio
            logging.info(f"Clamped speed weight: ratio was {speed_to_steering_ratio:.2f}, max allowed is {max_ratio:.2f}")
    
    # Re-normalize to maintain target sum after ratio clamping
    current_sum = steering_weight + speed_weight
    steering_weight = (steering_weight / current_sum) * TARGET_WEIGHT_SUM
    speed_weight = (speed_weight / current_sum) * TARGET_WEIGHT_SUM
    
    logging.info(f"Dynamic weights: Steering: {steering_weight:.4f}, Speed: {speed_weight:.4f} (sum={steering_weight + speed_weight:.4f}) | Based on losses Steering: {avg_steering_loss:.5f}, Speed: {avg_speed_loss:.5f} | Importance bias Steering: {steering_importance:.1f}x, Speed: {speed_importance:.1f}x")
    
    return steering_weight, speed_weight

def train_model(
    dataset_path,
    checkpoint_dir="checkpoints",  # Directory to save checkpoints
    model_dir="models",  # Directory to save models
    resume_from=None,  # Path to resume from checkpoint
    window_size=20,  # For 1s of driving data at 20fps
    target_length=1200,  # Length of each segment in frames
    stride=20,  # Non-overlapping windows
    batch_size=16, 
    num_epochs=30,
    num_workers=2,
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

    # Number of workers to use for data loading is controlled by the
    # num_workers function argument so it can be tuned for the available hardware
    
    # Store normalization constants for inference
    normalization_mean = processed_dataset.preprocessor.transform.transforms[-1].mean
    normalization_std = processed_dataset.preprocessor.transform.transforms[-1].std
    
    # Create DataLoaders with batch samplers
    train_dataset = VideoWindowIterableDataset(processed_dataset, video_indices=train_videos_indices, shuffle=True)
    val_dataset = VideoWindowIterableDataset(processed_dataset, video_indices=val_videos_indices, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # Now works as expected
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
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

    # Dynamic weight averaging initialization
    loss_history = []  # Store loss history for dynamic weight calculation
    STARTING_STEERING_WEIGHT = 1.0
    STARTING_SPEED_WEIGHT = 0.5
    steering_weight = STARTING_STEERING_WEIGHT
    speed_weight = STARTING_SPEED_WEIGHT

    # Gradient clipping
    max_grad_norm = 1.0

    # Mixed precision training
    scaler = GradScaler('cuda')

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
        
        # Load loss history and dynamic weights if available
        if 'loss_history' in checkpoint:
            loss_history = checkpoint['loss_history']
            logging.info(f"Loaded loss history with {len(loss_history)} epochs")
            
            # Calculate current weights based on loaded history
            if loss_history:
                steering_weight, speed_weight = calculate_dynamic_weights(
                    loss_history,
                    STARTING_STEERING_WEIGHT,
                    STARTING_SPEED_WEIGHT,
                    steering_importance=2.0,  # Steering is 2x more important than speed
                    speed_importance=1.0
                )
                logging.info(f"Resumed with dynamic weights - Steering: {steering_weight:.4f}, Speed: {speed_weight:.4f}")
            else:
                logging.info(f"Using starting weights - Steering: {steering_weight:.4f}, Speed: {speed_weight:.4f}")
        else:
            logging.info("No loss history found in checkpoint, starting with default weights")
        
        # Load normalization constants if available (for backward compatibility)
        if 'normalization_mean' in checkpoint and 'normalization_std' in checkpoint:
            loaded_mean = checkpoint['normalization_mean']
            loaded_std = checkpoint['normalization_std']
            logging.info(f"Loaded normalization constants from checkpoint: mean={loaded_mean}, std={loaded_std}")
            # Verify they match current preprocessor settings
            if (loaded_mean != normalization_mean or loaded_std != normalization_std):
                logging.warning(f"Checkpoint normalization constants differ from current settings!")
                logging.warning(f"Checkpoint: mean={loaded_mean}, std={loaded_std}")
                logging.warning(f"Current: mean={normalization_mean}, std={normalization_std}")
        else:
            logging.info("No normalization constants found in checkpoint (older checkpoint format)")
    else:
        if resume_from:
             logging.warning(f"Checkpoint file not found at '{resume_from}'. Starting training from scratch.")
        else:
             logging.info("No checkpoint specified. Starting training from scratch.")

    # Set log interval
    log_interval = 1000  # Log every 1000 batches

    # Training loop
    logging.info(f"Starting training for {num_epochs} epochs.")
    logging.info(f"Initial weights - Steering: {steering_weight:.4f}, Speed: {speed_weight:.4f}")

    # Log training start
    if debug:
        logging.info(f"Starting training with {num_workers} workers")

    # --- Training Phase ---
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
        logging.info(f"Current epoch weights - Steering: {steering_weight:.4f}, Speed: {speed_weight:.4f}")
        model.train()
        running_loss = 0.0
        
        # For profiling data loading vs model training
        total_data_loading_time = 0
        total_model_time = 0
        
        # Accumulate component losses during training
        running_steering_loss = 0.0
        running_speed_loss = 0.0

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
                # Use dynamic weights
                loss = steering_weight * loss_steering + speed_weight * loss_speed
                
                # Accumulate component losses for monitoring
                running_steering_loss += loss_steering.item()
                running_speed_loss += loss_speed.item()

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

            # Log detailed batch information
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(
                f"Epoch: {epoch+1:03d}/{num_epochs}, "
                f"Batch: {i+1:05d}/{len(train_loader)}, "
                f"Total Loss: {loss.item():.6f}, "
                f"Steering Loss: {loss_steering.item():.6f}, "
                f"Speed Loss: {loss_speed.item():.6f}, "
                f"Weights: Steering: {steering_weight:.4f}, Speed: {speed_weight:.4f}, "
                f"LR: {current_lr:.8f}"
            )
            
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
        avg_train_loss = running_loss / len(train_loader)  # Mean across batches, no need to divide by batch_size again
        
        # Calculate component loss averages from accumulated values
        avg_steering_loss = running_steering_loss / len(train_loader)
        avg_speed_loss = running_speed_loss / len(train_loader)
        # --- End of Training Phase ---

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        
        # Accumulate validation component losses
        val_steering_loss = 0.0
        val_speed_loss = 0.0

        with torch.no_grad():
            for val_i, data in enumerate(val_loader):
                if data is None:
                    continue

                frames = data['frames'].to(device)  # Shape: [batch_size, window_size, 6, H, W]
                steering = data['steering'].to(device)  # Shape: [batch_size, window_size]
                speed = data['speed'].to(device)  # Shape: [batch_size, window_size]
                
                # Create vehicle state tensor by combining speed and steering
                veh_states = torch.stack([speed, steering], dim=2)  # Shape: [batch_size, window_size, 2]

                # Reset hidden state for each validation batch
                hidden_state = None

                # Forward pass with mixed precision for validation
                with autocast(device_type='cuda'):
                    # Pass both frames and vehicle states to the model
                    steering_pred, speed_pred, _ = model(frames, veh_states, hidden_state)
                    
                    # Use the actual future ground truth values from the dataset
                    future_steering_targets = data['future_steering'].to(device)  # [batch_size, 5]
                    future_speed_targets = data['future_speed'].to(device)  # [batch_size, 5]

                    loss_steering = criterion_steering(steering_pred, future_steering_targets)
                    loss_speed = criterion_speed(speed_pred, future_speed_targets)
                    # Use dynamic weights
                    loss = steering_weight * loss_steering + speed_weight * loss_speed

                val_loss += loss.item()
                
                # Accumulate validation component losses
                val_steering_loss += loss_steering.item()
                val_speed_loss += loss_speed.item()

                # Log detailed batch information for validation
                current_val_lr = optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else None
                logging.info(
                    f"Validation - Epoch: {epoch+1:03d}/{num_epochs}, "
                    f"Batch: {val_i+1:05d}/{len(val_loader)}, "
                    f"Total Loss: {loss.item():.6f}, "
                    f"Steering Loss: {loss_steering.item():.6f}, "
                    f"Speed Loss: {loss_speed.item():.6f}, "
                    f"Weights: Steering: {steering_weight:.4f}, Speed: {speed_weight:.4f}, "
                    f"LR: {current_val_lr:.8f}" if current_val_lr is not None else ""
                )

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)  # Mean across batches, no need to divide by batch_size again
        
        # Calculate validation component loss averages
        avg_val_steering_loss = val_steering_loss / len(val_loader)
        avg_val_speed_loss = val_speed_loss / len(val_loader)
        # --- End of Validation Phase ---

        # --- Update Loss History and Calculate New Weights ---
        # Store this epoch's losses and weights in history
        epoch_loss_info = {
            'epoch': epoch + 1,
            'train_steering_loss': avg_steering_loss,
            'train_speed_loss': avg_speed_loss,
            'val_steering_loss': avg_val_steering_loss,
            'val_speed_loss': avg_val_speed_loss,
            'steering_loss': avg_val_steering_loss,  # Use validation loss for weight calculation
            'speed_loss': avg_val_speed_loss,
            'steering_weight': steering_weight,
            'speed_weight': speed_weight
        }
        loss_history.append(epoch_loss_info)
        
        # Calculate new weights for the next epoch based on validation losses
        if epoch + 1 < num_epochs:  # Only calculate new weights if there are more epochs
            new_steering_weight, new_speed_weight = calculate_dynamic_weights(
                loss_history,
                STARTING_STEERING_WEIGHT,
                STARTING_SPEED_WEIGHT,
                steering_importance=2.0,  # Steering is 2x more important than speed
                speed_importance=1.0
            )
            logging.info(f"Weight update: Steering: {steering_weight:.4f}, Speed: {speed_weight:.4f} → Steering: {new_steering_weight:.4f}, Speed: {new_speed_weight:.4f}")
            steering_weight, speed_weight = new_steering_weight, new_speed_weight
        # --- End of Weight Update ---

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Log epoch summary
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f} seconds")
        logging.info(f"Epoch Summary - Average Train Loss: {avg_train_loss:.4f} (Steering: {avg_steering_loss:.4f}, Speed: {avg_speed_loss:.4f}) | Val Loss: {avg_val_loss:.4f} (Steering: {avg_val_steering_loss:.4f}, Speed: {avg_val_speed_loss:.4f})")

        # --- Checkpoint Saving ---
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_model_path = os.path.join(model_dir, 'best_model.pth')
            logging.info(f"New best model found! Avg Val Loss: {avg_val_loss:.4f} (previous best: {best_val_loss if epoch > 0 else 'N/A'})")
            best_val_loss = avg_val_loss
            # Save best model with normalization constants for inference
            torch.save({
                'model_state_dict': model.state_dict(),
                'normalization_mean': normalization_mean,
                'normalization_std': normalization_std,
                'val_loss': avg_val_loss,
                'loss_history': loss_history,  # Save loss history in best model too
            }, best_model_path)
        
        # Save the latest checkpoint including optimizer state etc.
        latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'normalization_mean': normalization_mean,
            'normalization_std': normalization_std,
            'loss_history': loss_history,  # Save loss history for dynamic weight calculation
        }, filename=latest_checkpoint_path)
        # --- End of Checkpoint Saving ---
        
    logging.info("Training completed.")
    return model

if __name__ == "__main__":
    # Initialize logging
    import argparse

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Train the model in different modes')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                        help='Training mode: "train" for real training, "test" for test training')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train (default: 30 for train, 5 for test)')
    parser.add_argument('--batch-size', type=int, default=20, help='Batch size for training')
    parser.add_argument('--window-size', type=int, default=20, help='Window size for temporal data')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of DataLoader worker processes')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to store checkpoints')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory to store trained models')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory to store logs')
    args = parser.parse_args()

    # Initialize logging using provided log directory
    logger = setup_logging(args.log_dir)
    
    # Set default epochs based on mode if not specified
    if args.epochs is None:
        if args.mode == 'train':
            args.epochs = 50
        else:  # test mode
            args.epochs = 10
    
    # Common parameters for both modes
    common_params = {
        'debug': args.debug,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'window_size': args.window_size,
        'num_workers': args.num_workers
    }

    if args.mode == 'train':
        # Real training mode - use full dataset
        logger.info(f"Starting real training mode with full dataset (epochs: {args.epochs})")

        # Check for existing checkpoint
        latest_checkpoint_path = os.path.join(args.checkpoint_dir, 'latest_checkpoint.pth')
        resume_from = latest_checkpoint_path if os.path.exists(latest_checkpoint_path) else None

        if resume_from:
            logger.info(f"Resuming training from checkpoint: {resume_from}")
        else:
            logger.info("No checkpoint found. Starting training from scratch.")

        train_model(
            dataset_path=args.dataset_path,
            checkpoint_dir=args.checkpoint_dir,
            model_dir=args.model_dir,
            resume_from=resume_from,
            **common_params
        )
    else:
        # Test training mode - use test dataset
        logging.info(f"Starting test training mode with test dataset (epochs: {args.epochs})")
        train_model(
            dataset_path=args.dataset_path,
            checkpoint_dir=args.checkpoint_dir,
            model_dir=args.model_dir,
            **common_params
        )
