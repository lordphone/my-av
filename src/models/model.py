# model.py
# Defines the neural network architecture
# A 2D CNN model for processing video data, with a RNN for temporal processing

import torch
import torch.nn as nn
import torchvision.models as models

class Model(nn.Module):
    def __init__(self, num_steering_outputs=1, cnn_feature_dim_expected=512, rnn_hidden_size=256, rnn_num_layers=2):
        super(Model, self).__init__()

        # Load a pre-trained ResNet18 model
        self.cnn_backbone = models.resnet18(pretrained=True)

        # Remove the final fully connected layer (classification layer)
        # The fully connected layer in ResNet18 is named 'fc'
        self.cnn_feature_dim = self.cnn_backbone.fc.in_features
        
        if self.cnn_feature_dim != cnn_feature_dim_expected:
            # This might happen if a different ResNet version is used or future torchvision changes.
            # For now, we'll proceed but a warning or handling might be good in a production system.
            print(f"Warning: ResNet18 fc.in_features is {self.cnn_feature_dim}, expected {cnn_feature_dim_expected}. Using actual value.")

        self.cnn_backbone.fc = nn.Identity() # Replace with an identity layer

        # GRU for temporal processing
        self.temporal_model = nn.GRU(
            input_size=self.cnn_feature_dim,  # Features from CNN
            hidden_size=rnn_hidden_size,    # Size of GRU's memory
            num_layers=rnn_num_layers,      # Number of stacked GRU layers
            batch_first=True,               # Input tensor format: (batch, seq, feature)
            dropout=0.2 if rnn_num_layers > 1 else 0 # Add dropout if using multiple GRU layers
        )
        
        # Fully connected layer to map GRU output to steering angle prediction
        self.output_layer = nn.Linear(rnn_hidden_size, num_steering_outputs)


    def forward(self, x):
        # x is expected to be a batch of frame sequences: (batch_size, sequence_length, C, H, W)
        
        batch_size, seq_len, C, H, W = x.shape
        
        # Reshape to process frames independently by the CNN
        cnn_in = x.view(batch_size * seq_len, C, H, W)
        
        # Pass through CNN backbone
        # cnn_out_features shape: (batch_size * seq_len, self.cnn_feature_dim)
        cnn_out_features = self.cnn_backbone(cnn_in) 
        
        # Reshape back to sequence for GRU
        # rnn_in shape: (batch_size, seq_len, self.cnn_feature_dim)
        rnn_in = cnn_out_features.view(batch_size, seq_len, self.cnn_feature_dim)
        
        # Pass through GRU
        # gru_out shape: (batch_size, seq_len, rnn_hidden_size) - output for each time step
        # hidden_state shape: (rnn_num_layers, batch_size, rnn_hidden_size) - final hidden state
        gru_out, _ = self.temporal_model(rnn_in) # We don't need the final hidden state separately for this common setup
        
        # We'll use the output from the last time step of the GRU to make the prediction
        # This means "make a decision after seeing the whole relevant sequence"
        # last_time_step_features shape: (batch_size, rnn_hidden_size)
        last_time_step_features = gru_out[:, -1, :]
        
        # Pass through the final output layer
        # steering_prediction shape: (batch_size, num_steering_outputs)
        steering_prediction = self.output_layer(last_time_step_features)
        
        return steering_prediction

# Example usage (for testing the model structure)
if __name__ == '__main__':
    # Create a dummy input tensor: batch_size=2, sequence_length=10, C=3, H=224, W=224
    # The DataPreprocessor uses img_size=(160, 320). ResNet typically expects 224x224 or similar.
    # Ensure your preprocessor output matches what the CNN expects, or add adaptive pooling.
    dummy_input = torch.randn(2, 10, 3, 160, 320) # Adjusted to match preprocessor typical output
    
    # Model hyperparameters (can be tuned)
    num_outputs = 1 # e.g., 1 for steering angle
    cnn_features = 512 # Expected from ResNet18
    rnn_hidden = 256
    rnn_layers = 2

    model = Model(
        num_steering_outputs=num_outputs,
        cnn_feature_dim_expected=cnn_features,
        rnn_hidden_size=rnn_hidden,
        rnn_num_layers=rnn_layers
    )
    
    # Test the forward pass
    predictions = model(dummy_input)
    print("Output prediction shape:", predictions.shape) # Expected: (batch_size, num_outputs), e.g., (2, 1)
    print("A few dummy predictions:", predictions)
    