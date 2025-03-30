import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

class WellLSTM(nn.Module):
    """
    LSTM model for well sensor time series prediction with attention mechanism.
    
    This model combines LSTM layers with self-attention to capture both temporal
    dependencies and long-range relationships in the sensor data.
    
    Attributes:
        input_size (int): Number of input features (number of sensors)
        hidden_size (int): Number of hidden units in LSTM layers
        num_layers (int): Number of LSTM layers
        dropout_rate (float): Dropout rate for regularization
        lstm (nn.LSTM): LSTM layers for temporal feature extraction
        attention_query (nn.Linear): Linear layer for attention query computation
        attention_key (nn.Linear): Linear layer for attention key computation
        attention_value (nn.Linear): Linear layer for attention value computation
        attention_scale (float): Scaling factor for attention scores
        fc1 (nn.Linear): First fully connected layer
        fc2 (nn.Linear): Second fully connected layer
        output_layer (nn.Linear): Final output layer
        layer_norm (nn.LayerNorm): Layer normalization for stability
        dropout (nn.Dropout): Dropout layer for regularization
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        """
        Initialize the WellLSTM model.
        
        Args:
            input_size: Number of input features (number of sensors)
            hidden_size: Number of hidden units in LSTM layers
            num_layers: Number of LSTM layers
            output_size: Number of output features (same as input_size for our case)
            dropout: Dropout rate for regularization (default: 0.2)
            
        Note:
            The model uses Xavier/Glorot initialization for LSTM weights and
            Kaiming/He initialization for ReLU-based layers.
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Attention mechanism
        self.attention_query = nn.Linear(hidden_size, hidden_size)
        self.attention_key = nn.Linear(hidden_size, hidden_size)
        self.attention_value = nn.Linear(hidden_size, hidden_size)
        self.attention_scale = math.sqrt(hidden_size)
        
        # Output layers with residual connections
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """
        Initialize model weights using appropriate initialization strategies.
        
        - LSTM weights: Xavier/Glorot initialization
        - ReLU-based layers: Kaiming/He initialization
        - Bias terms: Zero initialization
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM weights are concatenated as (input, forget, cell, output) gates
                    # Use Xavier/Glorot initialization for LSTM
                    nn.init.xavier_uniform_(param)
                else:
                    # Use Kaiming/He initialization for ReLU-based layers
                    if len(param.shape) >= 2:
                        nn.init.kaiming_normal_(param, nonlinearity='relu')
                    else:
                        # For 1D parameters (like some bias terms that got labeled as weights)
                        nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
               containing the sensor readings
            
        Returns:
            torch.Tensor: Predictions of shape (batch_size, output_size)
            
        Note:
            The forward pass consists of:
            1. LSTM processing
            2. Self-attention computation
            3. Layer normalization
            4. Residual feed-forward network
            5. Final prediction
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # Shape: (batch, seq_len, hidden_size)
        
        # Self-attention mechanism
        query = self.attention_query(lstm_out)  # Shape: (batch, seq_len, hidden_size)
        key = self.attention_key(lstm_out)  # Shape: (batch, seq_len, hidden_size)
        value = self.attention_value(lstm_out)  # Shape: (batch, seq_len, hidden_size)
        
        # Compute attention scores
        scores = torch.bmm(query, key.transpose(1, 2)) / self.attention_scale  # Shape: (batch, seq_len, seq_len)
        attention_weights = F.softmax(scores, dim=2)  # Shape: (batch, seq_len, seq_len)
        
        # Apply attention to get context vectors
        context = torch.bmm(attention_weights, value)  # Shape: (batch, seq_len, hidden_size)
        
        # Use the last context vector for prediction
        context = context[:, -1, :]  # Shape: (batch, hidden_size)
        
        # Apply layer normalization
        context = self.layer_norm(context)
        
        # Feed-forward network with residual connection
        residual = context
        out = self.fc1(context)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = out + residual  # Residual connection
        
        # Final prediction
        out = self.output_layer(out)  # Shape: (batch, output_size)
        return out 