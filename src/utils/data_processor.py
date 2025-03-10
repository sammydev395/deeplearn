import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class WellTimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length=24, prediction_horizon=1):
        """
        data: pandas DataFrame with columns for each sensor reading (pressure, temp, etc.)
        sequence_length: number of time steps to use for input sequence
        prediction_horizon: how many steps ahead to predict
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Normalize the data
        self.scaler_params = {}
        normalized_data = {}
        for column in data.columns:
            mean = data[column].mean()
            std = data[column].std()
            normalized_data[column] = (data[column] - mean) / std
            self.scaler_params[column] = {'mean': mean, 'std': std}
        
        self.data = pd.DataFrame(normalized_data)
        
        # Create sequences
        self.sequences = []
        for i in range(len(self.data) - sequence_length - prediction_horizon + 1):
            seq = self.data.iloc[i:(i + sequence_length)]
            target = self.data.iloc[i + sequence_length + prediction_horizon - 1:i + sequence_length + prediction_horizon]
            self.sequences.append((seq, target))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return (torch.FloatTensor(seq.values), 
                torch.FloatTensor(target.values.squeeze()))
    
    def inverse_transform(self, normalized_data, column):
        """Convert normalized values back to original scale"""
        mean = self.scaler_params[column]['mean']
        std = self.scaler_params[column]['std']
        return normalized_data * std + mean

def prepare_dataloader(data, batch_size=32, sequence_length=24, prediction_horizon=1):
    """Create DataLoader for training"""
    dataset = WellTimeSeriesDataset(data, sequence_length, prediction_horizon)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True) 