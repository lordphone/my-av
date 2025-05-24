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

        # Replace classifier with identity
        self.cnn_backbone.fc = nn.Identity()
        # Prepare first conv to accept stacked current+past (6-channel) inputs
        # and copy pretrained 3-channel weights to the new 6-channel layer
        orig_conv1 = models.resnet18(weights=ResNet18_Weights.DEFAULT).conv1.weight.data
        self.cnn_backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # copy pretrained weights to both halves of the 6-channel input
            self.cnn_backbone.conv1.weight[:, :3, :, :] = orig_conv1
            self.cnn_backbone.conv1.weight[:, 3:, :, :] = orig_conv1

        # Total features into GRU: one CNN output + vehicle state dims
        gru_input_size = self.cnn_feature_dim + num_vehicle_state_features

        # GRU for temporal processing (time-first: seq_len, batch, feature)
        self.temporal_model = nn.GRU(
            input_size=gru_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=0.2 if rnn_num_layers > 1 else 0
        )
        
        # Fully connected layers to map GRU output to 5 future predictions
        self.steering_output_layer = nn.Linear(rnn_hidden_size, num_future_predictions)
        self.speed_output_layer = nn.Linear(rnn_hidden_size, num_future_predictions)

    def forward(self, window_imgs, veh_states, hidden_state=None):
        # window_imgs: [batch_size, seq_len, 6, H, W]
        # veh_states:  [batch_size, seq_len, num_vehicle_state_features]
        batch_size, seq_len, C, H, W = window_imgs.shape
        # Flatten batch and time dims for CNN feature extraction
        flat_imgs = window_imgs.reshape(batch_size * seq_len, C, H, W)
        flat_feats = self.cnn_backbone(flat_imgs)                           # [batch*seq, cnn_feature_dim]
        feats = flat_feats.reshape(batch_size, seq_len, self.cnn_feature_dim)  # [batch, seq_len, cnn_feature_dim]
        # Combine CNN features with vehicle states
        seq_features = torch.cat([feats, veh_states], dim=2)               # [batch, seq_len, gru_input_size]

        # Run GRU over the full sequence (batch_first=True)
        gru_out, new_hidden_state = self.temporal_model(seq_features, hidden_state)
        # Take output from last time step
        final_feat = gru_out[:, -1, :]                           # [batch, rnn_hidden_size]

        # Map to future predictions
        steering = self.steering_output_layer(final_feat)
        speed = self.speed_output_layer(final_feat)
        return steering, speed, new_hidden_state

# Example usage (for testing the model structure)
if __name__ == '__main__':
    batch_size = 2
    seq_len = 5
    C, H, W = 3, 160, 320
    num_vehicle_state_features = 2  # speed, steering
    num_future_predictions = 5

    # Create dummy sequences of current and past frames in batch-first order
    dummy_current = torch.randn(batch_size, seq_len, C, H, W)
    dummy_past    = torch.randn(batch_size, seq_len, C, H, W)
    # Stack current and past frames along the channel axis → [B, T, 6, H, W]
    dummy_window_imgs = torch.cat([dummy_current, dummy_past], dim=2)
    # Create dummy vehicle state sequence [B, T, F]
    dummy_veh_states = torch.randn(batch_size, seq_len, num_vehicle_state_features)

    # Model hyperparameters
    cnn_features = 512
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
    steering, speed, _ = model(dummy_window_imgs, dummy_veh_states)
    print(f"Steering predictions shape: {steering.shape}")  # (batch_size, num_future_predictions)
    print(f"Speed predictions shape: {speed.shape}")      # (batch_size, num_future_predictions)
    print("\nSteering predictions (first item in batch):\n", steering[0])
    print("\nSpeed predictions (first item in batch):\n", speed[0])
