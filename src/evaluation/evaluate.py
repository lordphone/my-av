# Evaluation script using ProcessedDataset infrastructure for proper ground truth integration

import torch
import numpy as np
import cv2
import os
import subprocess
import tempfile
from tqdm import tqdm

from src.models.model import Model
from src.data.comma2k19dataset import Comma2k19Dataset
from src.data.processed_dataset import ProcessedDataset

def draw_predictions(frame, steering_predictions, speed_predictions, current_steering=None, current_speed=None):
    """
    Draw predictions and ground truth on a frame.
    
    Args:
        frame: The frame to draw on
        steering_predictions: Steering prediction value (in degrees)
        speed_predictions: Speed prediction value (in m/s)
        current_steering: Current steering ground truth (in degrees)
        current_speed: Current speed ground truth (in m/s)
    
    Returns:
        The frame with predictions drawn
    """
    # Convert speed from m/s to mph
    speed_mph = speed_predictions[0] * 2.23694
    
    # Steering is already in degrees from the Comma2k19 dataset
    # Fix sign inversion: Invert the sign to match original data
    steering_degrees = -steering_predictions[0]  # Flip the sign to correct the inversion
    
    # Background rectangle for better visibility
    cv2.rectangle(frame, (5, 5), (350, 80), (0, 0, 0), -1)
    
    # Add speed text (prediction vs ground truth if available)
    if current_speed is not None:
        speed_gt_mph = current_speed * 2.23694
        cv2.putText(frame, f"Speed: {speed_mph:.1f} mph (GT: {speed_gt_mph:.1f})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f"Speed: {speed_mph:.1f} mph", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add steering value (prediction vs ground truth if available)
    if current_steering is not None:
        steering_gt_degrees = -current_steering  # Also fix sign for ground truth
        cv2.putText(frame, f"Steering: {steering_degrees:.1f}° (GT: {steering_gt_degrees:.1f}°)", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f"Steering: {steering_degrees:.1f}°", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

def process_video_with_dataset(model_path, video_path, output_path, window_size=20, frame_delay=2, img_size=(240, 320)):
    """
    Process a video file using ProcessedDataset infrastructure for proper ground truth integration.
    
    Args:
        model_path: Path to the trained model
        video_path: Path to the input video directory (segment directory, not video file)
        output_path: Path to save the output video
        window_size: Number of frames in each window
        frame_delay: Number of frames for T-100ms lookback  
        img_size: Size of the images (height, width)
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    try:
        from src.utils.inference_utils import load_model_for_inference
        model, norm_mean, norm_std = load_model_for_inference(model_path, Model, device)
        print(f"Model loaded successfully with normalization: mean={norm_mean}, std={norm_std}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create a dataset from the specific video path
    # Extract the dataset path (should be the base path containing all chunks)
    dataset_base_path = os.path.dirname(os.path.dirname(os.path.dirname(video_path)))
    print(f"Creating dataset from base path: {dataset_base_path}")
    
    # Create base dataset and processed dataset
    base_dataset = Comma2k19Dataset(dataset_base_path)
    processed_dataset = ProcessedDataset(
        base_dataset,
        window_size=window_size,
        target_length=1200,  # Standard segment length
        stride=20,  # Non-overlapping windows
        img_size=img_size,
        frame_delay=frame_delay,
        future_steps=5,
        future_step_size=2,
        fps=20
    )
    
    print(f"Dataset created with {len(processed_dataset)} windows")
    
    # Find the specific segment that matches our video path
    target_segment_idx = None
    for idx in range(len(base_dataset)):
        sample = base_dataset[idx]
        if sample['video_path'] == video_path:
            target_segment_idx = idx
            break
    
    if target_segment_idx is None:
        print(f"Could not find segment matching video path: {video_path}")
        return
    
    print(f"Found target segment at index: {target_segment_idx}")
    
    # Get all windows for this specific segment
    windows_per_segment = processed_dataset.windows_per_segment
    start_window_idx = target_segment_idx * windows_per_segment
    end_window_idx = start_window_idx + windows_per_segment
    
    print(f"Processing windows {start_window_idx} to {end_window_idx-1} (total: {windows_per_segment})")
    
    # Load the actual video file for frame extraction and output
    actual_video_path = video_path
    if video_path.endswith('.hevc'):
        print(f"Processing HEVC video file: {video_path}")
        # Create a temporary file  
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Convert HEVC to MP4 using ffmpeg
        safe_video_path = video_path.replace('|', '\|')
        print(f"Converting video with ffmpeg to {temp_path}")
        
        try:
            command = f"ffmpeg -y -i '{safe_video_path}' -c:v libx264 -preset ultrafast '{temp_path}'"
            subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Successfully converted to temporary MP4 file: {temp_path}")
            actual_video_path = temp_path
        except subprocess.CalledProcessError as e:
            print(f"Error converting video: {e}")
            print("Will attempt to open original file directly")
    
    # Open the video file for frame extraction
    cap = cv2.VideoCapture(actual_video_path)
    
    if not cap.isOpened():
        print(f"First attempt to open video failed, trying with FFMPEG backend: {actual_video_path}")
        cap = cv2.VideoCapture(actual_video_path, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print(f"Could not open video file: {actual_video_path}")
        return
    
    # Get video properties
    fps = 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Create output video writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each window using the dataset
    hidden_state = None
    frame_count = 0
    
    pbar = tqdm(total=windows_per_segment, desc="Processing windows")
    
    try:
        for window_idx in range(start_window_idx, min(end_window_idx, len(processed_dataset))):
            # Get the window data from ProcessedDataset
            window_data = processed_dataset[window_idx]
            
            frames_window = window_data['frames']  # [window_size, 6, H, W]
            steering_window = window_data['steering']  # [window_size]
            speed_window = window_data['speed']  # [window_size]
            current_steering = window_data['steering'][-1]  # Last value from steering sequence
            current_speed = window_data['speed'][-1]  # Last value from speed sequence
            
            # Process each frame in the window
            for frame_in_window in range(len(frames_window)):
                # Calculate the actual frame index in the video
                actual_frame_idx = (window_idx - start_window_idx) * 20 + frame_in_window + frame_delay
                
                # Read the corresponding frame from the video
                cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    print(f"Could not read frame {actual_frame_idx}")
                    continue
                
                # Get the preprocessed frame tensor for this specific frame
                frame_tensor = frames_window[frame_in_window:frame_in_window+1]  # [1, 6, H, W]
                
                # Create vehicle state tensor - use ground truth values from dataset
                current_frame_steering = steering_window[frame_in_window].item()
                current_frame_speed = speed_window[frame_in_window].item()
                veh_states = torch.tensor([[[current_frame_speed, current_frame_steering]]], 
                                        dtype=torch.float32).to(device)  # [1, 1, 2]
                
                # Add batch and sequence dimensions to frame tensor
                frames_tensor = frame_tensor.unsqueeze(0).to(device)  # [1, 1, 6, H, W]
                
                # Run inference
                with torch.no_grad():
                    steering_pred, speed_pred, hidden_state = model(frames_tensor, veh_states, hidden_state)
                
                # Extract predictions
                steering_predictions = steering_pred[0].cpu().numpy()
                speed_predictions = speed_pred[0].cpu().numpy()
                
                # Draw predictions and ground truth on the frame
                visualized_frame = draw_predictions(
                    frame.copy(), 
                    steering_predictions,
                    speed_predictions,
                    current_frame_steering,  # Ground truth steering
                    current_frame_speed      # Ground truth speed
                )
                
                # Write the frame to output video
                out.write(visualized_frame)
                frame_count += 1
            
            pbar.update(1)
    
    except Exception as e:
        print(f"Error during video processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Release resources
        if cap is not None:
            cap.release()
        out.release()
        pbar.close()
    
    print(f"Processed {frame_count} frames. Output saved to {output_path}")

if __name__ == "__main__":
    # Path to your trained model
    model_path = "/home/lordphone/my-av/models/best_model.pth"
    
    # Path to input video file
    video_path = "/home/lordphone/my-av/tests/data/Chunk_1/b0c9d2329ad1606b|2018-07-27--06-03-57/6/video.hevc"
    
    # Path to save output video
    output_path = "/home/lordphone/my-av/output/predictions_visualization_with_gt.mp4"
    
    # Process the video with predictions and ground truth
    process_video_with_dataset(model_path, video_path, output_path)