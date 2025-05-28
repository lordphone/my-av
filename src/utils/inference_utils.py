# inference_utils.py
# Utility functions for model inference

import torch

def load_model_for_inference(checkpoint_path, model_class, device='cuda'):
    """
    Load a trained model for inference, including normalization constants.
    
    Args:
        checkpoint_path: Path to the saved model checkpoint
        model_class: The model class to instantiate
        device: Device to load the model on
    
    Returns:
        Tuple of (model, normalization_mean, normalization_std)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if this is a best_model.pth (dict format) or state_dict format
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        normalization_mean = checkpoint.get('normalization_mean', [0.485, 0.456, 0.406] * 2)
        normalization_std = checkpoint.get('normalization_std', [0.229, 0.224, 0.225] * 2)
    else:
        # Old format - just state dict
        model_state_dict = checkpoint
        # Use ImageNet defaults for backward compatibility
        normalization_mean = [0.485, 0.456, 0.406] * 2  # For current + T-100ms frames
        normalization_std = [0.229, 0.224, 0.225] * 2
        print("Warning: Using default normalization constants (ImageNet). Consider retraining with updated checkpoint format.")
    
    # Initialize model and load weights
    model = model_class()
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    return model, normalization_mean, normalization_std

def prepare_frame_for_inference(frame, mean, std, img_size=(240, 320), device='cuda'):
    """
    Prepare a raw frame for model inference.
    Only needed if processing individual camera frames outside the dataset pipeline.
    
    Args:
        frame: Raw frame as numpy array (H, W, 3) with values in [0, 255]
        mean: Normalization means
        std: Normalization stds  
        img_size: Target image size (H, W)
        device: Device to move tensor to
    
    Returns:
        Normalized frame tensor ready for model input
    """
    from torchvision import transforms
    
    # Create the same transforms as used in training
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean[:3], std=std[:3])  # Only first 3 for single frame
    ])
    
    normalized_frame = transform(frame)
    return normalized_frame.unsqueeze(0).to(device)  # Add batch dimension

# Example usage:
"""
# Load model for inference
from src.models.model import Model
model, norm_mean, norm_std = load_model_for_inference('models/best_model.pth', Model)

# Now you have the exact normalization constants used during training
print(f"Normalization mean: {norm_mean}")
print(f"Normalization std: {norm_std}")
"""
