# evaluate.py
# Evalation script

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.models.model import Model
from src.data.comma2k19dataset import Comma2k19Dataset
from src.data.processed_dataset import ProcessedDataset
from torch.utils.data import DataLoader

def denormalize_image(image, mean, std):
    """Denormalize an image tensor."""
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    return (image * std) + mean

def evaluate(model_path, dataset_path, num_samples=1):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = Model()
    model.load_state_dict(torch.load(model_path))
    model.to(device)  # Move the model to the appropriate device
    model.eval()

    # Create the dataset
    base_dataset = Comma2k19Dataset(dataset_path)
    processed_dataset = ProcessedDataset(base_dataset, window_size=16)
    dataloader = DataLoader(processed_dataset, batch_size=1, shuffle=True)

    # Evaluate the model
    results = []

    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            if i >= num_samples:
                break

            frames = batch_data['frames'].to(device)
            steering = batch_data['steering'].numpy()
            speed = batch_data['speed'].numpy()

            # Forward pass
            steering_pred, speed_pred = model(frames)

            # Store results
            steering_pred = steering_pred.cpu().numpy() 
            speed_pred = speed_pred.cpu().numpy()

            # Store the predictions and ground truth
            results.append({
                'frame': batch_data['frames'][0].numpy(),
                'steering': steering[0, 0],
                'speed': speed[0, 0],
                'steering_pred': steering_pred[0, 0],
                'speed_pred': speed_pred[0, 0],
            })

        # Visualize results
        for i, result in enumerate(results):
            plt.figure(figsize=(12, 8))

            # Denormalize frames for visualization
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            denormalized_frame = denormalize_image(result['frame'][i], mean, std)

            # Plot frames
            ax = plt.subplot(2, 1, 1)
            ax.imshow(denormalized_frame.transpose(1, 2, 0))
            ax.set_title("Evaluation Frames")

            # Overlay trajectory cone
            # plot_trajectory_overlay(ax, denormalized_frame.transpose(1, 2, 0), result['steering_pred'], result['speed_pred'])

            # Add steering and speed plots
            plt.figtext(0.15, 0.8, f"Steering: {result['steering']:.2f} | Predicted Steering: {result['steering_pred']:.2f}", 
                        fontsize=12, color='white', bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.3'))
            plt.figtext(0.15, 0.75, f"Speed: {result['speed']:.2f} | Predicted Speed: {result['speed_pred']:.2f}", 
                        fontsize=12, color='white', bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.3'))
            plt.tight_layout()
            plt.savefig(f"evaluation_result_{i}.png")
            plt.close()
    print("Evaluation completed. Results saved as images.")
if __name__ == "__main__":
    evaluate("/home/lordphone/my-av/tests/data/best_model.pth", "/home/lordphone/my-av/data/raw/comma2k19", num_samples=10)