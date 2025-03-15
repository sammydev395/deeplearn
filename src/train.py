import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime
from tqdm import tqdm
import os
import time

from models.well_ts_model import WellLSTM
from utils.data_processor import TagDataProcessor, prepare_dataloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def train_model(
    config_path: str,
    data_path: str,
    model_save_path: str,
    sequence_length: int = 24,
    prediction_horizon: int = 1,
    hidden_size: int = 128,
    num_layers: int = 2,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 100,
    patience: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    use_mixed_precision: bool = True
):
    """
    Train the LSTM model on well sensor data
    
    Args:
        config_path: Path to tag_config.json
        data_path: Path to tagdata.json
        model_save_path: Where to save the trained model
        sequence_length: Number of time steps to use for input
        prediction_horizon: How many steps ahead to predict
        hidden_size: Number of hidden units in LSTM layers
        num_layers: Number of LSTM layers
        learning_rate: Learning rate for optimization
        batch_size: Training batch size
        epochs: Number of training epochs
        patience: Number of epochs to wait for improvement before early stopping
        device: Device to train on (cuda/cpu)
        use_mixed_precision: Whether to use mixed precision training (faster on compatible GPUs)
    """
    # Log GPU information if available
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
        logging.info(f"Training on GPU: {gpu_name} with {gpu_memory:.2f} GB memory")
        
        # Set memory usage optimizations
        torch.backends.cudnn.benchmark = True
        
        # Empty GPU cache
        torch.cuda.empty_cache()
    else:
        logging.info("Training on CPU")
    
    # Setup mixed precision if requested and available
    scaler = None
    if use_mixed_precision and device == 'cuda' and torch.cuda.is_available():
        if torch.cuda.get_device_capability()[0] >= 7:  # Volta or newer architecture
            scaler = torch.cuda.amp.GradScaler()
            logging.info("Using mixed precision training")
        else:
            logging.info("Mixed precision requested but GPU architecture doesn't support it efficiently")
    
    # Load and preprocess data
    start_time = time.time()
    logging.info(f"Loading and preprocessing data from {data_path}")
    processor = TagDataProcessor(config_path)
    raw_data = processor.load_and_preprocess_data(data_path)
    normalized_data = processor.normalize_data(raw_data)
    logging.info(f"Data loaded and preprocessed in {time.time() - start_time:.2f} seconds")
    
    # Split into train/val
    train_size = int(0.8 * len(normalized_data))
    train_data = normalized_data.iloc[:train_size]
    val_data = normalized_data.iloc[train_size:]
    logging.info(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples")
    
    # Create dataloaders
    train_loader = prepare_dataloader(
        train_data,
        batch_size=batch_size,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon
    )
    val_loader = prepare_dataloader(
        val_data,
        batch_size=batch_size,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon
    )
    
    # Initialize model
    input_size = len(processor.numeric_columns)
    model = WellLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=input_size
    ).to(device)
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model has {total_params:,} parameters ({trainable_params:,} trainable)")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Create directory for model if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            # Mixed precision training
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]")
        
        with torch.no_grad():
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = model(data)
                        loss = criterion(output, target)
                else:
                    output = model(data)
                    loss = criterion(output, target)
                
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Log metrics
        logging.info(f'Epoch: {epoch+1}/{epochs}')
        logging.info(f'Training Loss: {train_loss:.6f}')
        logging.info(f'Validation Loss: {val_loss:.6f}')
        logging.info(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            
            model_info = {
                'state_dict': model.state_dict(),
                'config': {
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'sequence_length': sequence_length,
                    'prediction_horizon': prediction_horizon
                },
                'scaler_params': processor.scaler_params,
                'feature_columns': processor.numeric_columns,
                'training_date': datetime.now().isoformat(),
                'best_val_loss': best_val_loss,
                'epoch': epoch + 1
            }
            torch.save(model_info, model_save_path)
            logging.info(f'Saved best model with validation loss: {val_loss:.6f}')
        else:
            epochs_without_improvement += 1
            
        # Early stopping
        if epochs_without_improvement >= patience:
            logging.info(f'Early stopping after {patience} epochs without improvement')
            break
    
    # Final cleanup
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    logging.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    return model_info

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='Path to tag_config.json')
    parser.add_argument('--data_path', type=str, required=True, help='Path to tagdata.json')
    parser.add_argument('--model_save_path', type=str, required=True, help='Where to save the model')
    parser.add_argument('--sequence_length', type=int, default=24)
    parser.add_argument('--prediction_horizon', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--no_mixed_precision', action='store_true', help='Disable mixed precision training')
    
    args = parser.parse_args()
    train_model(
        config_path=args.config_path,
        data_path=args.data_path,
        model_save_path=args.model_save_path,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        use_mixed_precision=not args.no_mixed_precision
    ) 