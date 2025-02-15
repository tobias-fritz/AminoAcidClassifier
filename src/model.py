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
        print(x)
        x = self.linear(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])
        x = self.softmax(x)
        return x

# graph neural network
class GraphConvolution(nn.Module):
    def __init__(self, node_in_len: int, node_out_len: int):
        # Call constructor of base class
        super().__init__()

        # Create linear layer for node matrix
        self.conv_linear = nn.Linear(node_in_len, node_out_len)

        # Create activation function
        self.conv_activation = nn.LeakyReLU()

    def forward(self, node_mat, adj_mat):
        # Calculate number of neighbors
        n_neighbors = adj_mat.sum(dim=-1, keepdims=True)
        # Create identity tensor
        self.idx_mat = torch.eye(
            adj_mat.shape[-2], adj_mat.shape[-1], device=n_neighbors.device
        )
        # Add new (batch) dimension and expand
        idx_mat = self.idx_mat.unsqueeze(0).expand(*adj_mat.shape)
        # Get inverse degree matrix
        inv_degree_mat = torch.mul(idx_mat, 1 / n_neighbors)

        # Perform matrix multiplication: D^(-1)AN
        node_fea = torch.bmm(inv_degree_mat, adj_mat)
        node_fea = torch.bmm(node_fea, node_mat)

        # Perform linear transformation to node features 
        # (multiplication with W)
        node_fea = self.conv_linear(node_fea)

        # Apply activation
        node_fea = self.conv_activation(node_fea)

        return node_fea

class PoolingLayer(nn.Module):
    def __init__(self):
        # Call constructor of base class
        super().__init__()

    def forward(self, node_fea):
        # Pool the node matrix
        pooled_node_fea = node_fea.mean(dim=1)
        return pooled_node_fea

class GraphConvolutionalNetwork(nn.Module):
    def __init__(
        self,
        node_vec_len: int,
        node_fea_len: int,
        hidden_fea_len: int,
        n_conv: int,
        n_hidden: int,
        n_outputs: int,
        p_dropout: float = 0.0,
    ):
        # Call constructor of base class
        super().__init__()

        # Define layers
        # Initial transformation from node matrix to node features
        self.init_transform = nn.Linear(node_vec_len, node_fea_len)

        # Convolution layers
        self.conv_layers = nn.ModuleList(
            [
                GraphConvolution(
                    node_in_len=node_fea_len,
                    node_out_len=node_fea_len,
                )
                for i in range(n_conv)
            ]
        )

        # Pool convolution outputs
        self.pooling = PoolingLayer()
        pooled_node_fea_len = node_fea_len

        # Pooling activation
        self.pooling_activation = nn.LeakyReLU()

        # From pooled vector to hidden layers
        self.pooled_to_hidden = nn.Linear(pooled_node_fea_len, hidden_fea_len)

        # Hidden layer
        self.hidden_layer = nn.Linear(hidden_fea_len, hidden_fea_len)

        # Hidden layer activation function
        self.hidden_activation = nn.LeakyReLU()

        # Hidden layer dropout
        self.dropout = nn.Dropout(p=p_dropout)

        # If hidden layers more than 1, add more hidden layers
        self.n_hidden = n_hidden
        if self.n_hidden > 1:
            self.hidden_layers = nn.ModuleList(
                [self.hidden_layer for _ in range(n_hidden - 1)]
            )
            self.hidden_activation_layers = nn.ModuleList(
                [self.hidden_activation for _ in range(n_hidden - 1)]
            )
            self.hidden_dropout_layers = nn.ModuleList(
                [self.dropout for _ in range(n_hidden - 1)]
            )

        # Final layer going to the output
        self.hidden_to_output = nn.Linear(hidden_fea_len, n_outputs)

    def forward(self, node_mat, adj_mat):
        # Perform initial transform on node_mat
        node_fea = self.init_transform(node_mat)

        # Perform convolutions
        for conv in self.conv_layers:
            node_fea = conv(node_fea, adj_mat)

        # Perform pooling
        pooled_node_fea = self.pooling(node_fea)
        pooled_node_fea = self.pooling_activation(pooled_node_fea)

        # First hidden layer
        hidden_node_fea = self.pooled_to_hidden(pooled_node_fea)
        hidden_node_fea = self.hidden_activation(hidden_node_fea)
        hidden_node_fea = self.dropout(hidden_node_fea)

        # Subsequent hidden layers
        if self.n_hidden > 1:
            for i in range(self.n_hidden - 1):
                hidden_node_fea = self.hidden_layers[i](hidden_node_fea)
                hidden_node_fea = self.hidden_activation_layers[i](hidden_node_fea)
                hidden_node_fea = self.hidden_dropout_layers[i](hidden_node_fea)

        # Output
        out = self.hidden_to_output(hidden_node_fea)

        return out