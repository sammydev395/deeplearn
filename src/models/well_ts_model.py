import torch
import torch.nn as nn

class WellTimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        """
        input_size: number of features (pressure, temp, vibration, depth)
        hidden_size: size of LSTM hidden states
        num_layers: number of stacked LSTM layers
        """
        super(WellTimeSeriesModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, input_size)  # Predict next values for each feature
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Use only the last time step output for prediction
        last_time_step = lstm_out[:, -1, :]
        predictions = self.fc(last_time_step)
        return predictions 