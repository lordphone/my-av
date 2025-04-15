# model.py
# Defines the neural network architecture

import torch
import torch.nn as nn
import torchvision.models as models

class Model(nn.Module):
    def __init__(self):
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
        
        # Calculate the size after convolutions
        # This would depend on your input size and architecture
        conv_output_size = 128 * 150 * 10 * 20  # Adjust these values based on your model
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Output heads
        self.steering_head = nn.Linear(512, 1)
        self.speed_head = nn.Linear(512, 1)
        
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