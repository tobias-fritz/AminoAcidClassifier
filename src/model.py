import torch
import torch.nn as nn
import torch.nn.init as init

class AminoAcidCNN(nn.Module):
    """
    Convolutional Neural Network for classifying amino acids.
    """

    def __init__(self, input_channels: int, input_height: int, input_width: int) -> None:
        """
        Initialize the AminoAcidCNN model.

        Args:
            input_channels (int): Number of input channels.
            input_height (int): Height of the input images.
            input_width (int): Width of the input images.
        """
        super(AminoAcidCNN, self).__init__()

        # Add the first convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=16, kernel_size=(3, 3), padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        # Calculate the output size after the first conv and pool layer
        self.output_height = (input_height - 3 + 2 * 1) // 1 + 1  # (H - F + 2P)/S + 1
        self.output_width = (input_width - 3 + 2 * 1) // 1 + 1  # (W - F + 2P)/S + 1
        self.output_height = self.output_height // 2  # Max pooling
        self.output_width = self.output_width // 2  # Max pooling
        
        # Add the second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1
        )
        
        # Calculate the output size after the second conv and pool layer
        self.output_height = (self.output_height - 3 + 2 * 1) // 1 + 1  # (H - F + 2P)/S + 1
        self.output_width = (self.output_width - 3 + 2 * 1) // 1 + 1  # (W - F + 2P)/S + 1
        self.output_height = self.output_height // 2  # Max pooling
        self.output_width = self.output_width // 2  # Max pooling
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(32 * self.output_height * self.output_width, 512)
        self.bn1 = nn.BatchNorm1d(512)  # BatchNorm layer normalizes the output of the previous layer and speeds up training
        self.fc2 = nn.Linear(512, 20)  # Hidden layer
        self.softmax = nn.Softmax(dim=1)  # Softmax layer for probability distribution

        # Initialize the weights with Kaiming initialization
        init.kaiming_uniform_(self.conv1.weight)
        init.kaiming_uniform_(self.conv2.weight)
        init.kaiming_uniform_(self.fc1.weight)
        init.kaiming_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = torch.relu(self.fc2(x))
        x = self.softmax(x)
        return x
    
class AminoAcidCNN_coord_only(nn.Module):
    """
    Convolutional Neural Network for classifying amino acids.
    """

    def __init__(self, input_channels: int, input_height: int, input_width: int) -> None:
        """
        Initialize the AminoAcidCNN model.

        Args:
            input_channels (int): Number of input channels.
            input_height (int): Height of the input images.
            input_width (int): Width of the input images.
        """
        super(AminoAcidCNN_coord_only, self).__init__()

        # Add the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=(2, 2), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=1)
        
        # Add the second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Calculate the output size after the conv and pool layers
        self.output_height = input_height +1  # No pooling layers
        self.output_width = input_width +1  # No pooling layers
        
        # Ensure the output dimensions are valid
        assert self.output_height > 0 and self.output_width > 0, "Output dimensions are too small after conv and pool layers"
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(32 * self.output_height * self.output_width, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 20)
        self.softmax = nn.Softmax(dim=1)

        # Initialize the weights with Kaiming initialization
        init.kaiming_uniform_(self.conv1.weight)
        init.kaiming_uniform_(self.conv2.weight)
        init.kaiming_uniform_(self.fc1.weight)
        init.kaiming_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = torch.relu(self.bn2(self.conv2(x)))
        #print(f"Shape after conv layers: {x.shape}")  # Print the shape

        x = x.view(x.size(0), -1)
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.softmax(x)
        return x