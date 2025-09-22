# Evaluation script using ProcessedDataset infrastructure for proper ground truth integration

import torch
import numpy as np
import cv2
import os
from tqdm import tqdm

from src.models.model import Model
from src.data.comma2k19dataset import Comma2k19Dataset
from src.data.data_preprocessor import DataPreprocessor
from src.utils.frame_reader import FrameReader

def process_video_with_dataset(model_path, data_path, output_path, segment_index=0, visualization_type="text"):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model and move to device
    model = Model(
        num_future_predictions=5,
        num_vehicle_state_features=2,
        cnn_feature_dim_expected=512,
        rnn_hidden_size=256,
        rnn_num_layers=2
    )
    model = model.to(device)
    model.eval()
    
    # Load model weights if path is provided
    if model_path and os.path.exists(model_path):
        print(f"Loading model weights from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            print("Found 'model_state_dict' in checkpoint, loading model weights...")
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load normalization stats if available
            if 'normalization_mean' in checkpoint and 'normalization_std' in checkpoint:
                print("Found normalization stats in checkpoint")
                # You might want to save these for preprocessing if needed
        else:
            # Try loading directly (for backward compatibility)
            try:
                model.load_state_dict(checkpoint)
                print("Loaded model weights directly from checkpoint")
            except Exception as e:
                print(f"Error loading model weights: {e}")
                print("Available keys in checkpoint:", checkpoint.keys())
                raise

    # Load the evaluation dataset
    base_dataset = Comma2k19Dataset(data_path)
    processor = DataPreprocessor()
    
    print(f"Loading segment {segment_index} from dataset (total segments: {len(base_dataset)})")
    segment = base_dataset[segment_index]
    processed_dataset = processor.preprocess_segment(segment)
    frames, steering, speed = processed_dataset
    
    # Get video path from the segment
    video_path = segment['video_path']
    print(f"Using video from: {video_path}")

    target_length = 1150
    
    # Format the data for the model
    frame_pairs = []
    for i in range(2, target_length):
        # Create a pair of current and past frame (6 channels)
        pair = torch.cat([frames[i], frames[i-2]], dim=0)
        frame_pairs.append(pair)
    
    # Ground truth steering and speed values
    gt_steering = steering[2:target_length]
    gt_speed = speed[2:target_length]

    # Feed the data to the model
    all_steering_preds = []  # Will store all 5 predictions for each frame
    all_speed_preds = []     # Will store all 5 predictions for each frame
    t_100ms_steering_preds = []
    t_100ms_speed_preds = []
    ground_truth = []
    hidden_state = None  # Initialize hidden state
    
    with torch.no_grad():
        for i in tqdm(range(len(frame_pairs)), desc="Processing frames"):
            frame_pair = frame_pairs[i]
            current_gt_steering = gt_steering[i]
            current_gt_speed = gt_speed[i]
            
            # Add batch and sequence dimensions and move to device
            frame_pair = frame_pair.unsqueeze(0).unsqueeze(0)  # [1, 1, 6, H, W]
            frame_pair = frame_pair.to(device)
            
            # Create vehicle state input (current steering and speed)
            veh_states = torch.tensor([[[current_gt_steering, current_gt_speed]]], dtype=torch.float32).to(device)
            
            # Get model predictions for 5 future time steps
            # Model returns (steering, speed, hidden_state). Preserve this order
            pred_steering, pred_speed, hidden_state = model(frame_pair, veh_states, hidden_state)
            
            # Extract all 5 predictions
            all_5_steering = pred_steering.cpu().numpy()[0]  # All 5 predictions [T+100ms, T+200ms, ..., T+500ms]
            all_5_speed = pred_speed.cpu().numpy()[0]        # All 5 predictions [T+100ms, T+200ms, ..., T+500ms]
            
            # Store all predictions
            all_steering_preds.append(all_5_steering)
            all_speed_preds.append(all_5_speed)
            
            # Also keep T+100ms for backward compatibility
            t_100ms_steering_preds.append(all_5_steering[0])  # First prediction (T+100ms)
            t_100ms_speed_preds.append(all_5_speed[0])        # First prediction (T+100ms)
            ground_truth.append((current_gt_steering, current_gt_speed))
    

    frames = []
    frame_reader = FrameReader(video_path)
    for i, frame in enumerate(frame_reader):
        frames.append(frame)
    frames = frames[2:target_length]

    p_speed = [0, 0] + t_100ms_speed_preds[:-2]
    p_steering = [0, 0] + t_100ms_steering_preds[:-2]


    # Create visualization video
    if visualization_type in ["trajectory", "combined"]:
        # For trajectory and combined visualization, pass all predictions and current values
        create_visualization(frames, all_speed_preds, all_steering_preds, ground_truth, output_path, visualization_type=visualization_type)
    else:
        # For text visualization, use the T+100ms predictions as before
        create_visualization(frames, p_speed, p_steering, ground_truth, output_path, visualization_type=visualization_type)
    
    return t_100ms_steering_preds, t_100ms_speed_preds, ground_truth

def mps_to_mph(speed_mps):
    """Convert speed from meters per second to miles per hour."""
    return speed_mps * 2.23694


def denormalize_steering(steering_norm):
    """Denormalize steering angle from [-1, 1] to [-25, 25] degrees"""
    STEER_CLIP = 25  # degrees
    return steering_norm * STEER_CLIP


def denormalize_speed(speed_norm):
    """Denormalize speed from [0, 1] to [0, 50] m/s"""
    SPEED_CLIP = 50  # m/s
    return speed_norm * SPEED_CLIP

def create_text_visualization(frames, steering_preds, speed_preds, ground_truth, output_path):
    """Create a video visualization with text overlay showing predictions vs ground truth."""
    # Create a video that shows the frames with overlaid predictions and ground truth
    if len(frames) == 0:
        print("No frames to process for visualization")
        return
        
    # Get frame dimensions (assuming CHW format)
    if isinstance(frames[0], torch.Tensor):
        height, width = frames[0].shape[1:]
    else:
        height, width = frames[0].shape[0], frames[0].shape[1]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
    
    # Initialize progress bar
    from tqdm import tqdm
    
    # Process frames with progress bar
    for i, (frame, pred_speed, pred_steering, gt) in enumerate(tqdm(zip(frames, speed_preds, steering_preds, ground_truth), 
                                                                  total=len(frames), 
                                                                  desc="Creating text visualization")):
        # Convert frame to numpy array if it's a tensor
        if isinstance(frame, torch.Tensor):
            frame = frame.permute(1, 2, 0).cpu().numpy()
        
        # Ensure the frame is in the correct format (HWC, uint8, 0-255)
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            if frame.max() <= 1.0:  # If values are in [0,1] range
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Convert grayscale to BGR if needed
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # If it's RGB, convert to BGR for OpenCV
        elif len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Denormalize ground truth values
        gt_steering_denorm = denormalize_steering(gt[0])
        gt_speed_denorm = denormalize_speed(gt[1])
        
        # Convert speed to mph
        gt_speed_mph = mps_to_mph(gt_speed_denorm)
        
        # Format steering angle display
        if abs(gt_steering_denorm) < 1.0:
            steering_display = "Straight"
        else:
            direction = "Left" if gt_steering_denorm > 0 else "Right"
            steering_display = f"{direction} {abs(gt_steering_denorm):.1f}"
            
        # Add current ground truth overlay (in white)
        cv2.putText(frame, f"Speed: {gt_speed_mph:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Steering: {steering_display}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Denormalize and convert predicted speed to mph
        pred_speed_denorm = denormalize_speed(pred_speed)
        pred_speed_mph = mps_to_mph(pred_speed_denorm)
        
        # Denormalize and format predicted steering display
        pred_steering_denorm = denormalize_steering(pred_steering)
        if abs(pred_steering_denorm) < 1.0:
            pred_steering_display = "Straight"
        else:
            direction = "Left" if pred_steering_denorm > 0 else "Right"
            pred_steering_display = f"{direction} {abs(pred_steering_denorm):.1f}"
            
        # Add T+100ms prediction overlay (predictions made from T-100ms ago)
        cv2.putText(frame, f"Pred Speed (from T-100ms): {pred_speed_mph:.1f}", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Pred Steering (from T-100ms): {pred_steering_display}", (10, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add difference between ground truth and prediction, with color based on accuracy
        diff_speed = gt_speed_mph - pred_speed_mph
        diff_steering = gt_steering_denorm - pred_steering_denorm
        
        # Calculate absolute differences for color determination
        abs_diff_speed = abs(diff_speed)
        abs_diff_steering = abs(diff_steering)
        
        # Calculate speed percentage error
        speed_percentage_error = (abs_diff_speed / gt_speed_mph * 100) if gt_speed_mph > 0 else 0
        
        # Determine color for steering difference
        # Green: within 1 degree, Yellow: 1-5 degrees, Red: above 5 degrees
        if abs_diff_steering <= 1.0:
            steering_color = (0, 255, 0)  # Green
        elif abs_diff_steering <= 5.0:
            steering_color = (0, 255, 255)  # Yellow (BGR format)
        else:
            steering_color = (0, 0, 255)  # Red
        
        # Determine color for speed difference
        # Green: within 10%, Yellow: 10-30%, Red: above 30%
        if speed_percentage_error <= 10.0:
            speed_color = (0, 255, 0)  # Green
        elif speed_percentage_error <= 30.0:
            speed_color = (0, 255, 255)  # Yellow (BGR format)
        else:
            speed_color = (0, 0, 255)  # Red
        
        cv2.putText(frame, f"Diff Speed: {diff_speed:.1f}", (10, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, speed_color, 2)
        cv2.putText(frame, f"Diff Steering: {diff_steering:.1f}", (10, 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, steering_color, 2)
        
        out.write(frame)
    
    out.release()
    print(f"Text visualization saved to {output_path}")


def create_trajectory_visualization(frames, steering_preds, speed_preds, ground_truth, output_path):
    """Create a video visualization with trajectory lines showing predicted vs actual path using all 5 predictions."""
    if len(frames) == 0:
        print("No frames to process for visualization")
        return
        
    # Get frame dimensions (assuming CHW format)
    if isinstance(frames[0], torch.Tensor):
        height, width = frames[0].shape[1:]
    else:
        height, width = frames[0].shape[0], frames[0].shape[1]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
    
    # Initialize progress bar
    from tqdm import tqdm
    
    # Vehicle parameters for trajectory calculation
    wheelbase = 2.7  # meters (approximate for typical car)
    dt = 0.1  # time step (100ms)
    steering_scale = 0.2  # Scale down steering sensitivity
    
    # Starting position (center bottom of frame)
    start_x = width // 2
    start_y = int(height * 0.9)
    
    # Pixel scale (pixels per meter) - adjust based on camera view
    pixels_per_meter = 18 
    
    # Process frames with progress bar
    for i, (frame, pred_speed_5, pred_steering_5, gt) in enumerate(tqdm(zip(frames, speed_preds, steering_preds, ground_truth), 
                                                                  total=len(frames), 
                                                                  desc="Creating trajectory visualization")):
        # Convert frame to numpy array if it's a tensor
        if isinstance(frame, torch.Tensor):
            frame = frame.permute(1, 2, 0).cpu().numpy()
        
        # Ensure the frame is in the correct format (HWC, uint8, 0-255)
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            if frame.max() <= 1.0:  # If values are in [0,1] range
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Convert grayscale to BGR if needed
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # If it's RGB, convert to BGR for OpenCV
        elif len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Current ground truth values (denormalized)
        gt_steering_denorm = denormalize_steering(gt[0])
        gt_speed_denorm = denormalize_speed(gt[1])
        
        # Calculate trajectory using 7 points total:
        # Point 0: BASE - Current position (starting point)
        # Point 1: T+100ms using current ground truth speed/steering  
        # Points 2-6: T+200ms to T+600ms using the 5 model predictions (which predict T+100ms to T+500ms)
        trajectory_points = [(start_x, start_y)]  # Point 0: BASE
        current_x, current_y = start_x, start_y
        current_theta = -np.pi/2  # Initially pointing up (negative y direction)
        
        # Point 1: Move using current ground truth values
        gt_steering_rad = np.radians(-gt_steering_denorm * steering_scale)  # Flip steering direction and scale
        if abs(gt_steering_rad) < 0.001:  # Straight line
            current_x += gt_speed_denorm * dt * np.cos(current_theta) * pixels_per_meter
            current_y += gt_speed_denorm * dt * np.sin(current_theta) * pixels_per_meter
        else:
            # Calculate radius of curvature
            radius = wheelbase / np.tan(abs(gt_steering_rad))
            
            # Angular velocity
            omega = gt_speed_denorm / radius
            if gt_steering_rad < 0:  # Right turn
                omega = -omega
            
            # Update position using circular motion
            cx = current_x - radius * np.sin(current_theta) * (1 if gt_steering_rad > 0 else -1) * pixels_per_meter
            cy = current_y + radius * np.cos(current_theta) * (1 if gt_steering_rad > 0 else -1) * pixels_per_meter
            
            # Update angle
            current_theta += omega * dt
            
            # Update position
            current_x = cx + radius * np.sin(current_theta) * (1 if gt_steering_rad > 0 else -1) * pixels_per_meter
            current_y = cy - radius * np.cos(current_theta) * (1 if gt_steering_rad > 0 else -1) * pixels_per_meter
        
        # Add Point 1
        if 0 <= current_x < width*2 and 0 <= current_y < height*2:
            trajectory_points.append((int(current_x), int(current_y)))
        
        # Points 2-6: Use the 5 model predictions
        for step in range(5):
            # Denormalize predictions for this time step
            pred_steering_denorm = denormalize_steering(pred_steering_5[step])
            pred_speed_denorm = denormalize_speed(pred_speed_5[step])
            
            # Convert to radians (flip steering direction and scale)
            pred_steering_rad = np.radians(-pred_steering_denorm * steering_scale)
            
            # Calculate next position using bicycle model
            if abs(pred_steering_rad) < 0.001:  # Straight line
                current_x += pred_speed_denorm * dt * np.cos(current_theta) * pixels_per_meter
                current_y += pred_speed_denorm * dt * np.sin(current_theta) * pixels_per_meter
            else:
                # Calculate radius of curvature
                radius = wheelbase / np.tan(abs(pred_steering_rad))
                
                # Angular velocity
                omega = pred_speed_denorm / radius
                if pred_steering_rad < 0:  # Right turn
                    omega = -omega
                
                # Update position using circular motion
                # Center of rotation
                cx = current_x - radius * np.sin(current_theta) * (1 if pred_steering_rad > 0 else -1) * pixels_per_meter
                cy = current_y + radius * np.cos(current_theta) * (1 if pred_steering_rad > 0 else -1) * pixels_per_meter
                
                # Update angle
                current_theta += omega * dt
                
                # Update position
                current_x = cx + radius * np.sin(current_theta) * (1 if pred_steering_rad > 0 else -1) * pixels_per_meter
                current_y = cy - radius * np.cos(current_theta) * (1 if pred_steering_rad > 0 else -1) * pixels_per_meter
            
            # Add point if within reasonable bounds
            if 0 <= current_x < width*2 and 0 <= current_y < height*2:  # Allow some margin
                trajectory_points.append((int(current_x), int(current_y)))
            else:
                break  # Stop if trajectory goes out of bounds
        
        # Create REAL ground truth trajectory using actual future ground truth values
        gt_trajectory = [(start_x, start_y)]  # Point 0: BASE
        gt_x, gt_y = start_x, start_y
        gt_theta = -np.pi/2
        
        # Use actual ground truth values for the next 6 time steps (every 100ms = 2 frames at 20fps)
        for step in range(6):
            # Get the actual ground truth values for this time step
            future_frame_idx = i + (step + 1) * 2  # Every 100ms = 2 frames at 20fps
            
            if future_frame_idx < len(ground_truth):
                # Use actual future ground truth values
                future_gt = ground_truth[future_frame_idx]
                future_gt_steering_denorm = denormalize_steering(future_gt[0])
                future_gt_speed_denorm = denormalize_speed(future_gt[1])
            else:
                # If we don't have future ground truth, use current values (edge case)
                future_gt_steering_denorm = gt_steering_denorm
                future_gt_speed_denorm = gt_speed_denorm
            
            # Apply the same scaling and direction as predictions
            future_gt_steering_rad = np.radians(-future_gt_steering_denorm * steering_scale)
            
            # Calculate next position using actual ground truth values
            if abs(future_gt_steering_rad) < 0.001:  # Straight line
                gt_x += future_gt_speed_denorm * dt * np.cos(gt_theta) * pixels_per_meter
                gt_y += future_gt_speed_denorm * dt * np.sin(gt_theta) * pixels_per_meter
            else:
                radius = wheelbase / np.tan(abs(future_gt_steering_rad))
                omega = future_gt_speed_denorm / radius
                if future_gt_steering_rad < 0:
                    omega = -omega
                cx = gt_x - radius * np.sin(gt_theta) * (1 if future_gt_steering_rad > 0 else -1) * pixels_per_meter
                cy = gt_y + radius * np.cos(gt_theta) * (1 if future_gt_steering_rad > 0 else -1) * pixels_per_meter
                gt_theta += omega * dt
                gt_x = cx + radius * np.sin(gt_theta) * (1 if future_gt_steering_rad > 0 else -1) * pixels_per_meter
                gt_y = cy - radius * np.cos(gt_theta) * (1 if future_gt_steering_rad > 0 else -1) * pixels_per_meter
            
            if 0 <= gt_x < width*2 and 0 <= gt_y < height*2:
                gt_trajectory.append((int(gt_x), int(gt_y)))
            else:
                break
        
        # Draw trajectories
        # Ground truth trajectory in Tesla/Waymo style - wide semi-transparent green path
        if len(gt_trajectory) > 1:
            gt_points = np.array(gt_trajectory, dtype=np.int32)
            
            # Create overlay for transparency
            overlay = frame.copy()
            
            # Draw multiple thick lines to create a path effect
            for thickness in [16, 12, 8, 4]:
                alpha = 0.3 if thickness > 8 else 0.6  # More transparent for wider lines
                cv2.polylines(overlay, [gt_points], False, (0, 255, 0), thickness)
            
            # Blend the overlay with the original frame
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            # Add a sharp center line
            cv2.polylines(frame, [gt_points], False, (0, 200, 0), 2)
            
        # Predicted trajectory in blue (keep as is)
        if len(trajectory_points) > 1:
            pred_points = np.array(trajectory_points, dtype=np.int32)
            cv2.polylines(frame, [pred_points], False, (255, 0, 0), 3)  # Blue line
        
        # Draw predicted trajectory points as circles (blue)
        for j, point in enumerate(trajectory_points):
            if j == 0:
                # BASE position (larger white circle)
                cv2.circle(frame, point, 8, (255, 255, 255), -1)
                cv2.circle(frame, point, 8, (0, 0, 0), 2)
                cv2.putText(frame, "BASE", (point[0]-15, point[1]-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            elif j == 1:
                # T+100ms point (both trajectories use this timing)
                cv2.circle(frame, point, 4, (255, 0, 0), -1)
                cv2.circle(frame, point, 4, (0, 0, 0), 1)
                cv2.putText(frame, "T+100ms", (point[0]-25, point[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            else:
                # Model prediction points (blue circles)
                cv2.circle(frame, point, 4, (255, 0, 0), -1)
                cv2.circle(frame, point, 4, (0, 0, 0), 1)
                # Add blue labels on the RIGHT side of blue dots
                cv2.putText(frame, f"T+{j}00ms", (point[0]+10, point[1]-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)  # Blue text
        
        # Draw ground truth trajectory points as circles (green)
        for j, gt_point in enumerate(gt_trajectory):
            if j == 0:
                # BASE already drawn above
                pass
            elif j == 1:
                # T+100ms ground truth point (green circle)
                cv2.circle(frame, gt_point, 4, (0, 255, 0), -1)
                cv2.circle(frame, gt_point, 4, (0, 0, 0), 1)
                # Label already added above since both use T+100ms
            else:
                # Ground truth points (green circles)
                cv2.circle(frame, gt_point, 4, (0, 255, 0), -1)
                cv2.circle(frame, gt_point, 4, (0, 0, 0), 1)
                # Add green labels on the LEFT side of green dots
                cv2.putText(frame, f"T+{j}00ms", (gt_point[0]-45, gt_point[1]-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)  # Green text
        
        # Add legend in bottom 1/4th of screen, towards center
        legend_x = width // 4  # 1/4 from left edge
        legend_y_start = int(height * 0.85)  # Start at 85% down the screen
        cv2.putText(frame, "Ground Truth", (legend_x, legend_y_start), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Predicted", (legend_x, legend_y_start + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Trajectory visualization saved to {output_path}")


def calculate_trajectory(start_x, start_y, speed, steering_angle, wheelbase, dt, steps, pixels_per_meter):
    """Calculate vehicle trajectory using bicycle model."""
    trajectory = [(start_x, start_y)]
    
    # Current state
    x, y = start_x, start_y
    theta = -np.pi/2  # Initially pointing up (negative y direction)
    
    for _ in range(steps):
        if abs(steering_angle) < 0.001:  # Straight line
            # Move straight
            x += speed * dt * np.cos(theta) * pixels_per_meter
            y += speed * dt * np.sin(theta) * pixels_per_meter
        else:
            # Calculate radius of curvature
            radius = wheelbase / np.tan(abs(steering_angle))
            
            # Angular velocity
            omega = speed / radius
            if steering_angle < 0:  # Right turn
                omega = -omega
            
            # Update position using circular motion
            # Center of rotation
            cx = x - radius * np.sin(theta) * (1 if steering_angle > 0 else -1) * pixels_per_meter
            cy = y + radius * np.cos(theta) * (1 if steering_angle > 0 else -1) * pixels_per_meter
            
            # Update angle
            theta += omega * dt
            
            # Update position
            x = cx + radius * np.sin(theta) * (1 if steering_angle > 0 else -1) * pixels_per_meter
            y = cy - radius * np.cos(theta) * (1 if steering_angle > 0 else -1) * pixels_per_meter
        
        # Only add points that are within the frame
        if 0 <= x < 1000 and 0 <= y < 1000:  # Reasonable bounds
            trajectory.append((int(x), int(y)))
        else:
            break
    
    return trajectory


def create_combined_visualization(frames, steering_preds, speed_preds, ground_truth, output_path):
    """Create a video visualization combining both text overlay AND trajectory lines."""
    if len(frames) == 0:
        print("No frames to process for visualization")
        return
        
    # Get frame dimensions (assuming CHW format)
    if isinstance(frames[0], torch.Tensor):
        height, width = frames[0].shape[1:]
    else:
        height, width = frames[0].shape[0], frames[0].shape[1]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
    
    # Initialize progress bar
    from tqdm import tqdm
    
    # Vehicle parameters for trajectory calculation
    wheelbase = 2.7  # meters (approximate for typical car)
    dt = 0.1  # time step (100ms)
    steering_scale = 0.2  # Scale down steering sensitivity
    
    # Starting position (center bottom of frame)
    start_x = width // 2
    start_y = int(height * 0.9)
    
    # Pixel scale (pixels per meter) - adjust based on camera view
    pixels_per_meter = 18 
    
    # Process frames with progress bar
    for i, (frame, pred_speed_5, pred_steering_5, gt) in enumerate(tqdm(zip(frames, speed_preds, steering_preds, ground_truth), 
                                                                  total=len(frames), 
                                                                  desc="Creating combined visualization")):
        # Convert frame to numpy array if it's a tensor
        if isinstance(frame, torch.Tensor):
            frame = frame.permute(1, 2, 0).cpu().numpy()
        
        # Ensure the frame is in the correct format (HWC, uint8, 0-255)
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            if frame.max() <= 1.0:  # If values are in [0,1] range
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Convert grayscale to BGR if needed
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # If it's RGB, convert to BGR for OpenCV
        elif len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Current ground truth values (denormalized)
        gt_steering_denorm = denormalize_steering(gt[0])
        gt_speed_denorm = denormalize_speed(gt[1])
        
        # === TRAJECTORY CALCULATION (same as trajectory visualization) ===
        # Calculate predicted trajectory using 7 points
        trajectory_points = [(start_x, start_y)]  # Point 0: BASE
        current_x, current_y = start_x, start_y
        current_theta = -np.pi/2  # Initially pointing up (negative y direction)
        
        # Point 1: Move using current ground truth values
        gt_steering_rad = np.radians(-gt_steering_denorm * steering_scale)
        if abs(gt_steering_rad) < 0.001:  # Straight line
            current_x += gt_speed_denorm * dt * np.cos(current_theta) * pixels_per_meter
            current_y += gt_speed_denorm * dt * np.sin(current_theta) * pixels_per_meter
        else:
            radius = wheelbase / np.tan(abs(gt_steering_rad))
            omega = gt_speed_denorm / radius
            if gt_steering_rad < 0:
                omega = -omega
            cx = current_x - radius * np.sin(current_theta) * (1 if gt_steering_rad > 0 else -1) * pixels_per_meter
            cy = current_y + radius * np.cos(current_theta) * (1 if gt_steering_rad > 0 else -1) * pixels_per_meter
            current_theta += omega * dt
            current_x = cx + radius * np.sin(current_theta) * (1 if gt_steering_rad > 0 else -1) * pixels_per_meter
            current_y = cy - radius * np.cos(current_theta) * (1 if gt_steering_rad > 0 else -1) * pixels_per_meter
        
        if 0 <= current_x < width*2 and 0 <= current_y < height*2:
            trajectory_points.append((int(current_x), int(current_y)))
        
        # Points 2-6: Use the 5 model predictions
        for step in range(5):
            pred_steering_denorm = denormalize_steering(pred_steering_5[step])
            pred_speed_denorm = denormalize_speed(pred_speed_5[step])
            pred_steering_rad = np.radians(-pred_steering_denorm * steering_scale)
            
            if abs(pred_steering_rad) < 0.001:
                current_x += pred_speed_denorm * dt * np.cos(current_theta) * pixels_per_meter
                current_y += pred_speed_denorm * dt * np.sin(current_theta) * pixels_per_meter
            else:
                radius = wheelbase / np.tan(abs(pred_steering_rad))
                omega = pred_speed_denorm / radius
                if pred_steering_rad < 0:
                    omega = -omega
                cx = current_x - radius * np.sin(current_theta) * (1 if pred_steering_rad > 0 else -1) * pixels_per_meter
                cy = current_y + radius * np.cos(current_theta) * (1 if pred_steering_rad > 0 else -1) * pixels_per_meter
                current_theta += omega * dt
                current_x = cx + radius * np.sin(current_theta) * (1 if pred_steering_rad > 0 else -1) * pixels_per_meter
                current_y = cy - radius * np.cos(current_theta) * (1 if pred_steering_rad > 0 else -1) * pixels_per_meter
            
            if 0 <= current_x < width*2 and 0 <= current_y < height*2:
                trajectory_points.append((int(current_x), int(current_y)))
            else:
                break
        
        # Calculate ground truth trajectory using actual future values
        gt_trajectory = [(start_x, start_y)]
        gt_x, gt_y = start_x, start_y
        gt_theta = -np.pi/2
        
        for step in range(6):
            future_frame_idx = i + (step + 1) * 2
            
            if future_frame_idx < len(ground_truth):
                future_gt = ground_truth[future_frame_idx]
                future_gt_steering_denorm = denormalize_steering(future_gt[0])
                future_gt_speed_denorm = denormalize_speed(future_gt[1])
            else:
                future_gt_steering_denorm = gt_steering_denorm
                future_gt_speed_denorm = gt_speed_denorm
            
            future_gt_steering_rad = np.radians(-future_gt_steering_denorm * steering_scale)
            
            if abs(future_gt_steering_rad) < 0.001:
                gt_x += future_gt_speed_denorm * dt * np.cos(gt_theta) * pixels_per_meter
                gt_y += future_gt_speed_denorm * dt * np.sin(gt_theta) * pixels_per_meter
            else:
                radius = wheelbase / np.tan(abs(future_gt_steering_rad))
                omega = future_gt_speed_denorm / radius
                if future_gt_steering_rad < 0:
                    omega = -omega
                cx = gt_x - radius * np.sin(gt_theta) * (1 if future_gt_steering_rad > 0 else -1) * pixels_per_meter
                cy = gt_y + radius * np.cos(gt_theta) * (1 if future_gt_steering_rad > 0 else -1) * pixels_per_meter
                gt_theta += omega * dt
                gt_x = cx + radius * np.sin(gt_theta) * (1 if future_gt_steering_rad > 0 else -1) * pixels_per_meter
                gt_y = cy - radius * np.cos(gt_theta) * (1 if future_gt_steering_rad > 0 else -1) * pixels_per_meter
            
            if 0 <= gt_x < width*2 and 0 <= gt_y < height*2:
                gt_trajectory.append((int(gt_x), int(gt_y)))
            else:
                break
        
        # === DRAW TRAJECTORIES ===
        # Ground truth trajectory in Tesla/Waymo style
        if len(gt_trajectory) > 1:
            gt_points = np.array(gt_trajectory, dtype=np.int32)
            overlay = frame.copy()
            for thickness in [16, 12, 8, 4]:
                cv2.polylines(overlay, [gt_points], False, (0, 255, 0), thickness)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.polylines(frame, [gt_points], False, (0, 200, 0), 2)
            
        # Predicted trajectory in blue
        if len(trajectory_points) > 1:
            pred_points = np.array(trajectory_points, dtype=np.int32)
            cv2.polylines(frame, [pred_points], False, (255, 0, 0), 3)
        
        # === DRAW TRAJECTORY POINTS ===
        # Predicted trajectory points
        for j, point in enumerate(trajectory_points):
            if j == 0:
                cv2.circle(frame, point, 8, (255, 255, 255), -1)
                cv2.circle(frame, point, 8, (0, 0, 0), 2)
                cv2.putText(frame, "BASE", (point[0]-15, point[1]-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            elif j == 1:
                cv2.circle(frame, point, 4, (255, 0, 0), -1)
                cv2.circle(frame, point, 4, (0, 0, 0), 1)
                cv2.putText(frame, "T+100ms", (point[0]-25, point[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            else:
                cv2.circle(frame, point, 4, (255, 0, 0), -1)
                cv2.circle(frame, point, 4, (0, 0, 0), 1)
                cv2.putText(frame, f"T+{j}00ms", (point[0]+10, point[1]-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        # Ground truth trajectory points
        for j, gt_point in enumerate(gt_trajectory):
            if j == 0:
                pass  # BASE already drawn
            elif j == 1:
                cv2.circle(frame, gt_point, 4, (0, 255, 0), -1)
                cv2.circle(frame, gt_point, 4, (0, 0, 0), 1)
            else:
                cv2.circle(frame, gt_point, 4, (0, 255, 0), -1)
                cv2.circle(frame, gt_point, 4, (0, 0, 0), 1)
                cv2.putText(frame, f"T+{j}00ms", (gt_point[0]-45, gt_point[1]-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # === ADD TEXT OVERLAY (from text visualization) ===
        # Convert speed to mph
        gt_speed_mph = mps_to_mph(gt_speed_denorm)
        
        # Format steering angle display
        if abs(gt_steering_denorm) < 1.0:
            steering_display = "Straight"
        else:
            direction = "Left" if gt_steering_denorm > 0 else "Right"
            steering_display = f"{direction} {abs(gt_steering_denorm):.1f}"
            
        # Add current ground truth overlay (in white)
        cv2.putText(frame, f"Speed: {gt_speed_mph:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Steering: {steering_display}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Denormalize and convert predicted speed to mph (T+100ms prediction)
        pred_speed_denorm = denormalize_speed(pred_speed_5[0])
        pred_speed_mph = mps_to_mph(pred_speed_denorm)
        
        # Denormalize and format predicted steering display
        pred_steering_denorm = denormalize_steering(pred_steering_5[0])
        if abs(pred_steering_denorm) < 1.0:
            pred_steering_display = "Straight"
        else:
            direction = "Left" if pred_steering_denorm > 0 else "Right"
            pred_steering_display = f"{direction} {abs(pred_steering_denorm):.1f}"
            
        # Add T+100ms prediction overlay (predictions made from T-100ms ago)
        cv2.putText(frame, f"Pred Speed (from T-100ms): {pred_speed_mph:.1f}", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Pred Steering (from T-100ms): {pred_steering_display}", (10, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add difference analysis with color coding
        diff_speed = gt_speed_mph - pred_speed_mph
        diff_steering = gt_steering_denorm - pred_steering_denorm
        
        abs_diff_speed = abs(diff_speed)
        abs_diff_steering = abs(diff_steering)
        
        speed_percentage_error = (abs_diff_speed / gt_speed_mph * 100) if gt_speed_mph > 0 else 0
        
        # Color coding for accuracy
        if abs_diff_steering <= 1.0:
            steering_color = (0, 255, 0)  # Green
        elif abs_diff_steering <= 5.0:
            steering_color = (0, 255, 255)  # Yellow
        else:
            steering_color = (0, 0, 255)  # Red
        
        if speed_percentage_error <= 10.0:
            speed_color = (0, 255, 0)  # Green
        elif speed_percentage_error <= 30.0:
            speed_color = (0, 255, 255)  # Yellow
        else:
            speed_color = (0, 0, 255)  # Red
        
        cv2.putText(frame, f"Diff Speed: {diff_speed:.1f}", (10, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, speed_color, 2)
        cv2.putText(frame, f"Diff Steering: {diff_steering:.1f}deg", (10, 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, steering_color, 2)
        
        # Add trajectory legend
        legend_x = width // 4
        legend_y_start = int(height * 0.85)
        cv2.putText(frame, "Ground Truth", (legend_x, legend_y_start), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Predicted", (legend_x, legend_y_start + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Combined visualization saved to {output_path}")


def create_visualization(frames, steering_preds, speed_preds, ground_truth, output_path, visualization_type="text"):
    """Wrapper function to create text, trajectory, or combined visualization."""
    if visualization_type == "text":
        create_text_visualization(frames, steering_preds, speed_preds, ground_truth, output_path)
    elif visualization_type == "trajectory":
        create_trajectory_visualization(frames, steering_preds, speed_preds, ground_truth, output_path)
    elif visualization_type == "combined":
        create_combined_visualization(frames, steering_preds, speed_preds, ground_truth, output_path)
    else:
        raise ValueError("visualization_type must be 'text', 'trajectory', or 'combined'")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained model and generate a visualization")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--data-path", type=str, default="data/comma2k19", help="Path to the evaluation dataset (default: data/comma2k19)")
    parser.add_argument("--output-path", type=str, required=True, help="Where to save the visualization")
    parser.add_argument("--index", type=int, default=0, help="Index of the segment to analyze from the dataset (default: 0)")
    parser.add_argument("--viz-type", type=str, choices=["text", "trajectory", "combined"], default="combined", 
                        help="Type of visualization: 'text' for text overlay, 'trajectory' for trajectory lines, 'combined' for both (default: text)")
    args = parser.parse_args()

    # Update the function to accept visualization type
    process_video_with_dataset(args.model_path, args.data_path, args.output_path, args.index, args.viz_type)
