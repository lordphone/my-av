# model.py
# Defines the neural network architecture
# A 2D CNN model for processing video data, with a GRU for temporal processing

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class Model(nn.Module):
    def __init__(self, 
                 num_future_predictions=5, 
                 num_vehicle_state_features=2, # For current speed and steering
                 cnn_feature_dim_expected=512, 
                 rnn_hidden_size=256, 
                 rnn_num_layers=2):
        super(Model, self).__init__()

        # Load a pre-trained ResNet18 model
        self.cnn_backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Remove the final fully connected layer (classification layer)
        # The fully connected layer in ResNet18 is named 'fc'
        self.cnn_feature_dim = self.cnn_backbone.fc.in_features
        
        if self.cnn_feature_dim != cnn_feature_dim_expected:
            # This might happen if a different ResNet version is used or future torchvision changes.
            # For now, we'll proceed but a warning or handling might be good in a production system.
            print(f"Warning: ResNet18 fc.in_features is {self.cnn_feature_dim}, expected {cnn_feature_dim_expected}. Using actual value.")

        self.cnn_backbone.fc = nn.Identity() # Replace with an identity layer

        # Calculate the total number of features being fed into the GRU
        # Features from 2 frames (current & past) + current vehicle state features
        gru_input_size = (self.cnn_feature_dim * 2) + num_vehicle_state_features

        # GRU for temporal processing
        self.temporal_model = nn.GRU(
            input_size=gru_input_size,      # Adjusted input size
            hidden_size=rnn_hidden_size,    # Size of GRU's memory
            num_layers=rnn_num_layers,      # Number of stacked GRU layers
            batch_first=True,               # Input tensor format: (batch, seq_len=1, feature)
            dropout=0.2 if rnn_num_layers > 1 else 0 # Add dropout if using multiple GRU layers
        )
        
        # Fully connected layers to map GRU output to 5 future predictions
        self.steering_output_layer = nn.Linear(rnn_hidden_size, num_future_predictions)
        self.speed_output_layer = nn.Linear(rnn_hidden_size, num_future_predictions)

    def forward(self, current_frames, past_frames, current_vehicle_states, hidden_state=None):
        # current_frames: (batch_size, C, H, W)
        # past_frames: (batch_size, C, H, W)
        # current_vehicle_states: (batch_size, num_vehicle_state_features)
        # hidden_state: (rnn_num_layers, batch_size, rnn_hidden_size), optional
        
        batch_size = current_frames.shape[0]
        
        # Process current frames through CNN
        # features_current shape: (batch_size, self.cnn_feature_dim)
        features_current = self.cnn_backbone(current_frames)
        
        # Process past frames through CNN
        # features_past shape: (batch_size, self.cnn_feature_dim)
        features_past = self.cnn_backbone(past_frames)
        
        # Concatenate CNN features and current vehicle states
        # aggregated_features shape: (batch_size, cnn_feature_dim*2 + num_vehicle_state_features)
        aggregated_features = torch.cat((features_current, features_past, current_vehicle_states), dim=1)
        
        # Reshape for GRU: (batch_size, seq_len=1, features)
        rnn_in = aggregated_features.unsqueeze(1) 
        
        # Pass through GRU
        # gru_out shape: (batch_size, seq_len=1, rnn_hidden_size)
        # hidden_state shape: (rnn_num_layers, batch_size, rnn_hidden_size)
        gru_out, new_hidden_state = self.temporal_model(rnn_in, hidden_state)
        
        # Use the output from the GRU's single time step
        # gru_output_last_step shape: (batch_size, rnn_hidden_size)
        gru_output_last_step = gru_out[:, -1, :] # or gru_out.squeeze(1)
        
        # Pass through the final output layers
        # steering_predictions shape: (batch_size, num_future_predictions)
        # speed_predictions shape: (batch_size, num_future_predictions)
        steering_predictions = self.steering_output_layer(gru_output_last_step)
        speed_predictions = self.speed_output_layer(gru_output_last_step)
        
        return steering_predictions, speed_predictions, new_hidden_state

# Example usage (for testing the model structure)
if __name__ == '__main__':
    batch_size = 2
    img_c, img_h, img_w = 3, 160, 320 # Adjusted to match preprocessor typical output
    num_vehicle_state_features = 2 # speed, steering
    num_future_predictions = 5

    # Create dummy input tensors
    dummy_current_frames = torch.randn(batch_size, img_c, img_h, img_w)
    dummy_past_frames = torch.randn(batch_size, img_c, img_h, img_w)
    dummy_current_vehicle_states = torch.randn(batch_size, num_vehicle_state_features)
    
    # Model hyperparameters (can be tuned)
    cnn_features = 512 # Expected from ResNet18
    rnn_hidden = 256
    rnn_layers = 2

    model = Model(
        num_future_predictions=num_future_predictions,
        num_vehicle_state_features=num_vehicle_state_features,
        cnn_feature_dim_expected=cnn_features,
        rnn_hidden_size=rnn_hidden,
        rnn_num_layers=rnn_layers
    )
    
    # Test the forward pass
    # For the first pass, hidden_state can be None (or initialized to zeros)
    steering_preds, speed_preds, h_state = model(dummy_current_frames, dummy_past_frames, dummy_current_vehicle_states)
    
    print(f"Steering predictions shape: {steering_preds.shape}") # Expected: (batch_size, num_future_predictions)
    print(f"Speed predictions shape: {speed_preds.shape}")     # Expected: (batch_size, num_future_predictions)
    print(f"Hidden state shape: {h_state.shape}") # Expected: (rnn_layers, batch_size, rnn_hidden)
    print("\nSteering predictions (first item in batch):\n", steering_preds[0])
    print("\nSpeed predictions (first item in batch):\n", speed_preds[0])

    # Example of a subsequent pass, using the hidden state from the previous pass
    # This would typically involve new input frames and vehicle states
    dummy_current_frames_next = torch.randn(batch_size, img_c, img_h, img_w)
    dummy_past_frames_next = torch.randn(batch_size, img_c, img_h, img_w) # In a real scenario, this might be the previous 'current_frames'
    dummy_current_vehicle_states_next = torch.randn(batch_size, num_vehicle_state_features)

    steering_preds_next, speed_preds_next, h_state_next = model(
        dummy_current_frames_next, 
        dummy_past_frames_next, 
        dummy_current_vehicle_states_next, 
        h_state  # Pass the hidden state from the previous step
    )
    print(f"\n--- Next Timestep ---")
    print(f"Steering predictions shape (next): {steering_preds_next.shape}")
    print(f"Speed predictions shape (next): {speed_preds_next.shape}")
    print(f"Hidden state shape (next): {h_state_next.shape}")
