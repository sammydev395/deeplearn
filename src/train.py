import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from models.well_ts_model import WellTimeSeriesModel
from utils.data_processor import prepare_dataloader
import json
import os

def load_config(config_path):
    """Load tag alias configuration with limits"""
    with open(config_path, 'r') as f:
        return json.load(f)

def train_model(train_data, config_data, model_save_path='models/well_ts_model.pth',
                epochs=100, batch_size=32, sequence_length=24):
    # Determine input size from data columns
    input_size = len(train_data.columns)
    
    # Initialize model
    model = WellTimeSeriesModel(input_size=input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Prepare data
    train_loader = prepare_dataloader(train_data, batch_size=batch_size, 
                                    sequence_length=sequence_length)
    
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}')
    
    # Save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config_data
    }, model_save_path)
    
    return model

if __name__ == "__main__":
    # Example usage
    # Load your time series data
    # data = pd.read_csv('path_to_your_data.csv')
    # config = load_config('path_to_config.json')
    # train_model(data, config) 