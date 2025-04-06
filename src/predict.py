import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
from pathlib import Path
import json
import time
from typing import Optional

from models.well_ts_model import WellLSTM
from utils.data_processor import TagDataProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction.log'),
        logging.StreamHandler()
    ]
)

def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load a trained model from disk
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on
        
    Returns:
        model: Loaded model
        model_info: Model configuration and metadata
    """
    model_info = torch.load(model_path, map_location=device)
    
    # Extract model configuration
    config = model_info['config']
    
    # Create model with the same architecture
    model = WellLSTM(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=config['input_size']
    ).to(device)
    
    # Load the trained weights
    model.load_state_dict(model_info['state_dict'])
    
    return model, model_info

def predict(
    model_path: str,
    config_path: str,
    data_path: str,
    sequence_length: int = 24,  # Default sequence length
    num_predictions: int = 50,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Make predictions using the trained model
    
    Args:
        model_path: Path to the saved model
        config_path: Path to tag_config.json
        data_path: Path to tagdata.json
        sequence_length: Length of input sequence (if None, use model's default)
        num_predictions: Number of future steps to predict
        device: Device to run prediction on
        
    Returns:
        predictions_df: DataFrame of predictions
        actual_df: DataFrame of actual values (if available)
        feature_names: List of feature names
    """
    # Log GPU information if available
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
        logging.info(f"Running prediction on GPU: {gpu_name} with {gpu_memory:.2f} GB memory")
        
        # Empty GPU cache
        torch.cuda.empty_cache()
    else:
        logging.info("Running prediction on CPU")
    
    # Load model
    start_time = time.time()
    model, model_info = load_model(model_path, device)
    model.eval()  # Set to evaluation mode
    logging.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    # Get model configuration
    if sequence_length is None:
        sequence_length = model_info['config']['sequence_length']
    
    # Load data processor
    processor = TagDataProcessor(config_path)
    
    # Ensure numeric columns match model's expected input
    processor.numeric_columns = model_info['feature_columns']
    processor.scaler_params = model_info['scaler_params']
    
    # Load and preprocess data
    start_time = time.time()
    raw_data = processor.load_and_preprocess_data(data_path)
    normalized_data = processor.normalize_data(raw_data)
    logging.info(f"Data loaded and preprocessed in {time.time() - start_time:.2f} seconds")
    
    # Get the last sequence from the data
    last_sequence = normalized_data.iloc[-sequence_length:].values
    
    # Convert to tensor
    input_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)  # Add batch dimension
    
    # Make predictions
    predictions = []
    current_sequence = input_sequence.clone()
    
    with torch.no_grad():
        for _ in range(num_predictions):
            # Get prediction for next step
            output = model(current_sequence)
            
            # Add prediction to list
            predictions.append(output.cpu().numpy()[0])
            
            # Update sequence for next prediction (remove first step, add prediction as last step)
            new_sequence = torch.cat([
                current_sequence[:, 1:, :],
                output.unsqueeze(1)
            ], dim=1)
            current_sequence = new_sequence
    
    # Convert predictions to numpy array
    predictions = np.array(predictions)
    
    # Inverse normalize predictions
    original_scale_predictions = processor.inverse_normalize(
        predictions, 
        processor.numeric_columns
    )
    
    # Get the actual values for comparison (if available)
    actual_future_original = None
    if len(normalized_data) > sequence_length + num_predictions:
        actual_future = normalized_data.iloc[sequence_length:sequence_length+num_predictions]
        actual_future_original = processor.inverse_normalize(
            actual_future.values,
            processor.numeric_columns
        )
    
    return original_scale_predictions, actual_future_original, processor.numeric_columns

def plot_predictions(predictions_df, actual_df=None, feature_names=None, num_features_to_plot=4):
    """Plot predictions against actual values"""
    num_steps = len(predictions_df)
    steps = np.arange(num_steps)
    
    # Determine how many features to plot
    if feature_names is None:
        feature_names = predictions_df.columns.tolist()
    
    num_features = min(num_features_to_plot, len(feature_names))
    
    # Create plot
    fig, axes = plt.subplots(num_features, 1, figsize=(12, 3*num_features))
    if num_features == 1:
        axes = [axes]
    
    for i in range(num_features):
        ax = axes[i]
        feature = feature_names[i]
        
        # Plot predictions
        ax.plot(steps, predictions_df[feature].values, 'b-', label='Predicted')
        
        # Plot actual values if available
        if actual_df is not None and feature in actual_df.columns:
            ax.plot(steps, actual_df[feature].values[:num_steps], 'r-', label='Actual')
            
        ax.set_title(f"{feature}")
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    logging.info("Predictions plot saved to predictions.png")
    
    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--config_path', type=str, required=True, help='Path to tag_config.json')
    parser.add_argument('--data_path', type=str, required=True, help='Path to tagdata.json')
    parser.add_argument('--sequence_length', type=int, default=None, help='Length of input sequence')
    parser.add_argument('--num_predictions', type=int, default=50, help='Number of future steps to predict')
    
    args = parser.parse_args()
    
    # Make predictions
    predictions_df, actual_df, feature_names = predict(
        model_path=args.model_path,
        config_path=args.config_path,
        data_path=args.data_path,
        sequence_length=args.sequence_length,
        num_predictions=args.num_predictions
    )
    
    # Plot predictions
    plot_predictions(predictions_df, actual_df, feature_names)
    
    # Print some prediction statistics
    logging.info(f"Prediction summary:")
    for feature in feature_names[:4]:  # Show first 4 features
        values = predictions_df[feature].values
        logging.info(f"{feature}: Mean={values.mean():.2f}, Min={values.min():.2f}, Max={values.max():.2f}") 