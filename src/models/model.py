# model.py
# Defines the neural network architecture
# A 3D CNN model for processing video data

import torch
import torch.nn as nn
import torchvision.models as models

class Model(nn.Module):
    def __init__(self, window_size=15):
        super(Model, self).__init__()
        
        # Use 3D convolutions for temporal processing
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            # More layers as needed
        )
        
        # Dynamically calculate the size after convolutions
        self._conv_output_size = None
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.get_conv_output_size((3, window_size, 160, 320)), 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Output heads
        self.steering_head = nn.Linear(512, 1) # Single output for current steering
        self.speed_head = nn.Linear(512, 1) # Single output for current speed

    def get_conv_output_size(self, input_shape):
        if self._conv_output_size is None:
            with torch.no_grad():
                dummy_input = torch.zeros(1, *input_shape)  # Batch size of 1
                output = self.conv3d(dummy_input)
                self._conv_output_size = int(torch.prod(torch.tensor(output.shape[1:])))
        return self._conv_output_size

    def forward(self, x):
        # x shape: [batch_size, sequence_length, channels, height, width]
        # Rearrange to [batch_size, channels, sequence_length, height, width]
        x = x.permute(0, 2, 1, 3, 4)
        
        # Process with 3D convolutions
        x = self.conv3d(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc(x)
        
        # Output heads
        steering = self.steering_head(x)
        speed = self.speed_head(x)
        
        return steering, speed