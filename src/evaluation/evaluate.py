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

def process_video_with_dataset(model_path, data_path, output_path, video_path):
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
    processed_dataset = processor.preprocess_segment(base_dataset[0])
    frames, steering, speed = processed_dataset

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
            pred_speed, pred_steering, hidden_state = model(frame_pair, veh_states, hidden_state)
            
            # We only care about the T+100ms prediction (first value)
            t_100ms_steering = pred_steering.cpu().numpy()[0][0]  # First prediction only
            t_100ms_speed = pred_speed.cpu().numpy()[0][0]        # First prediction only
            
            # Store predictions and ground truth
            t_100ms_steering_preds.append(t_100ms_steering)
            t_100ms_speed_preds.append(t_100ms_speed)
            ground_truth.append((current_gt_steering, current_gt_speed))
    

    frames = []
    frame_reader = FrameReader(video_path)
    for i, frame in enumerate(frame_reader):
        frames.append(frame)
    
    # Create visualization video
    create_visualization(frames, t_100ms_steering_preds, t_100ms_speed_preds, ground_truth, output_path)
    
    return t_100ms_steering_preds, t_100ms_speed_preds, ground_truth

def mps_to_mph(speed_mps):
    """Convert speed from meters per second to miles per hour."""
    return speed_mps * 2.23694

def create_visualization(frames, steering_preds, speed_preds, ground_truth, output_path):
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
                                                                  desc="Creating visualization")):
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
        
        # Convert ground truth speeds to mph
        gt_speed_mph = mps_to_mph(gt[1])
        
        # Format steering angle display
        gt_steering = gt[0]
        if abs(gt_steering) < 1.0:
            steering_display = "Straight"
        else:
            direction = "Left" if gt_steering > 0 else "Right"
            steering_display = f"{direction} {abs(gt_steering):.1f}"
            
        # Add current ground truth overlay (in white)
        cv2.putText(frame, f"Speed: {gt_speed_mph:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Steering: {steering_display}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Convert predicted speed to mph
        pred_speed_mph = mps_to_mph(pred_speed)
        
        # Format predicted steering display
        if abs(pred_steering) < 1.0:
            pred_steering_display = "Straight"
        else:
            direction = "Left" if pred_steering > 0 else "Right"
            pred_steering_display = f"{direction} {abs(pred_steering):.1f}"
            
        # Add T+100ms prediction overlay (in green)
        cv2.putText(frame, f"Pred Speed: {pred_speed_mph:.1f}", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Pred Steering: {pred_steering_display}", (10, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    # Path to your trained model
    model_path = "/home/lordphone/my-av/models/best_model.pth"

    # Path to video (using raw string to handle special characters)
    video_path = r"/home/lordphone/my-av/data/evaluation/Chunk_1/b0c9d2329ad1606b|2018-07-27--06-03-57/9/video.hevc"
    
    
    # Path to data
    data_path = "/home/lordphone/my-av/data/evaluation/"
    
    # Path to save output video
    output_path = "/home/lordphone/my-av/data/evaluation/prediction.mp4"
    
    # Process the video with predictions and ground truth
    process_video_with_dataset(model_path, data_path, output_path, video_path)