import torch
import torch.nn as nn
import torch.nn.init as init

class AminoAcidCNN(nn.Module):

    def __init__(self, input_channels: int, input_height: int, input_width: int) -> None:
        super(AminoAcidCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        # Calculate the output size after the first conv and pool layer
        self.output_height = (input_height - 3 + 2*1) // 1 + 1 # (H - F + 2P)/S + 1
        self.output_width = (input_width - 3 + 2*1) // 1 + 1 # (W - F + 2P)/S + 1
        self.output_height = self.output_height // 2 # Max pooling
        self.output_width = self.output_width // 2 # Max pooling
        
        # Add the second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        
        # Calculate the output size after the second conv and pool layer
        self.output_height = (self.output_height - 3 + 2*1) // 1 + 1 # (H - F + 2P)/S + 1
        self.output_width = (self.output_width - 3 + 2*1) // 1 + 1 # (W - F + 2P)/S + 1
        self.output_height = self.output_height // 2 # Max pooling
        self.output_width = self.output_width // 2 # Max pooling
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(32 * self.output_height * self.output_width, 512)
        self.bn1 = nn.BatchNorm1d(512) # BatchNorm layer normalizes the output of the previous layer and speeds up training
        self.fc2 = nn.Linear(512, 20) # Hidden layer
        self.softmax = nn.Softmax(dim=1)  # Softmax layer for probability distribution

        # Initialize the weights with Kaiming initialization
        init.kaiming_uniform_(self.conv1.weight)
        init.kaiming_uniform_(self.conv2.weight)
        init.kaiming_uniform_(self.fc1.weight)
        init.kaiming_uniform_(self.fc2.weight)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print(f"Input shape: {x.shape}")
        x = self.pool(torch.relu(self.conv1(x)))
        #print(x.shape)  # Should print torch.Size([1, 16, 2, 3])
        x = self.pool(torch.relu(self.conv2(x)))
        #print(x.shape)  # Check the shape after the second conv and pool layer
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        #print(f"Shape before BatchNorm1d: {x.shape}") 
        x = self.bn1(x)
        x = torch.relu(self.fc2(x))
        x = self.softmax(x)  # Apply softmax to the output
        return x


