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
    
# Only a transformer type model is implemented here
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        """
        Initialize the PositionalEncoding layer.

        Args:
            d_model (int): Dimension of the model.
            dropout (float): Dropout rate.
            max_len (int): Maximum length of the input.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PositionalEncoding layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the layer.
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
class AminoAcidTransformer(nn.Module):
    """
    Transformer model for classifying amino acids.
    """

    def __init__(self, input_dim: int, 
                 num_classes: int, 
                 num_heads: int, 
                 num_layers: int, 
                 hidden_dim: int, 
                 dropout: float) -> None:
        """
        Initialize the AminoAcidTransformer model.
        

        Args:
            input_dim (int): Dimension of the input.
            num_classes (int): Number of classes.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            hidden_dim (int): Hidden dimension of the model.
            dropout (float): Dropout rate.
        """
        super(AminoAcidTransformer, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Define the linear layer instead of embedding layer
        self.linear = nn.Linear(input_dim, hidden_dim)
        
        # Define the positional encoding layer
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)
        
        # Define the transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Define the output layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

        # Initialize the weights with Kaiming initialization
        init.kaiming_uniform_(self.fc.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = self.linear(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])
        x = self.softmax(x)
        return x
