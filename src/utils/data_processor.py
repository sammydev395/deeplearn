import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import json
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

class TagDataProcessor:
    def __init__(self, config_path: str):
        """
        Initialize processor with tag configuration
        
        Args:
            config_path: Path to tag_config.json
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract relevant tags (numeric sensors only)
        self.tags = {}
        for tag in config['Tags']:
            if tag.get('DataType') in ['Float', 'UInt16', 'UInt32'] and tag.get('Units'):
                self.tags[tag['TagAlias']] = {
                    'data_type': tag['DataType'],
                    'units': tag['Units'],
                    'description': tag['Description']
                }
        
        self.numeric_columns = list(self.tags.keys())
        self.scaler_params = {}
        
        # If no numeric columns found, use a fallback approach
        if not self.numeric_columns:
            print("Warning: No numeric columns found in tag config. Using fallback approach.")
            # Create some synthetic columns for testing
            self.numeric_columns = ["ICV1ValvePosition", "ICV2ValvePosition", "VMM1TubingP", "VMM1TubingT"]
            for col in self.numeric_columns:
                self.tags[col] = {
                    'data_type': 'Float',
                    'units': '%',
                    'description': f'Synthetic {col}'
                }
    
    def load_and_preprocess_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and preprocess tag data
        
        Args:
            data_path: Path to tagdata.json
        """
        # Load data
        try:
            with open(data_path, 'r') as f:
                # Try to load as a JSON array first
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]  # Convert to list if it's a single object
                except json.JSONDecodeError:
                    # If that fails, try line-by-line parsing
                    f.seek(0)  # Reset file pointer
                    data = []
                    for line in f:
                        try:
                            record = json.loads(line)
                            if isinstance(record, dict):
                                data.append(record)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create synthetic data for testing
            data = self._create_synthetic_data()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Check if we have the expected columns
        if 'TagAlias' in df.columns and 'DisplayValue' in df.columns and 'Timestamp' in df.columns:
            # Data is in a format where each row is a different tag reading
            # We need to pivot it so each column is a tag and each row is a timestamp
            
            # Convert DisplayValue to numeric where possible
            df['NumericValue'] = pd.to_numeric(df['DisplayValue'], errors='coerce')
            
            # Convert timestamp
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            # Filter for numeric tags that we're interested in
            available_tags = set(df['TagAlias'].unique())
            usable_tags = set(self.numeric_columns).intersection(available_tags)
            
            if not usable_tags:
                print(f"Warning: None of the configured numeric tags found in data. Available tags: {available_tags}")
                # Use synthetic data as fallback
                return self._create_synthetic_dataframe()
            
            # Update numeric columns to only include available tags
            self.numeric_columns = list(usable_tags)
            
            # Pivot the data
            pivot_df = df.pivot_table(
                index='Timestamp', 
                columns='TagAlias', 
                values='NumericValue',
                aggfunc='first'  # Take the first value if multiple readings at same timestamp
            )
            
            # Keep only numeric columns we're interested in
            pivot_df = pivot_df[self.numeric_columns]
            
            # Forward fill missing values (use previous value)
            pivot_df = pivot_df.fillna(method='ffill')
            
            # Drop rows with any remaining NaN values
            pivot_df = pivot_df.dropna()
            
            return pivot_df
        else:
            # Data is already in the expected format with each column as a tag
            # Filter for numeric columns
            available_columns = set(df.columns)
            usable_columns = set(self.numeric_columns).intersection(available_columns)
            
            if not usable_columns:
                print(f"Warning: None of the configured numeric tags found in data. Available columns: {available_columns}")
                # Use synthetic data as fallback
                return self._create_synthetic_dataframe()
            
            # Update numeric columns to only include available columns
            self.numeric_columns = list(usable_columns)
            
            # Keep only numeric columns
            df = df[self.numeric_columns]
            
            # Convert data types
            for col in df.columns:
                if col in self.tags and self.tags[col]['data_type'] in ['UInt16', 'UInt32', 'Float']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop rows with NaN values
            df = df.dropna()
            
            return df
    
    def _create_synthetic_data(self) -> List[Dict[str, Any]]:
        """Create synthetic data for testing when real data is not available"""
        print("Creating synthetic data for testing")
        data = []
        start_time = datetime(2025, 1, 1)
        
        for i in range(1000):
            timestamp = start_time + pd.Timedelta(hours=i)
            for tag in self.numeric_columns:
                data.append({
                    'Id': len(data) + 1,
                    'TagAlias': tag,
                    'Timestamp': timestamp.isoformat(),
                    'DisplayValue': str(np.sin(i/10) * 50 + 50 + np.random.normal(0, 5)),
                    'Quality': 'HEALTHY',
                    'WellDeviceID': 'SYNTHETIC-DEVICE'
                })
        
        return data
    
    def _create_synthetic_dataframe(self) -> pd.DataFrame:
        """Create synthetic dataframe for testing"""
        print("Creating synthetic dataframe for testing")
        dates = pd.date_range(start='2025-01-01', periods=1000, freq='H')
        data = {}
        
        for tag in self.numeric_columns:
            # Create sine wave with noise
            values = np.sin(np.arange(1000)/10) * 50 + 50 + np.random.normal(0, 5, 1000)
            data[tag] = values
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the data using min-max scaling"""
        normalized_data = {}
        
        for column in df.columns:
            mean = df[column].mean()
            std = df[column].std()
            if std == 0:  # Avoid division by zero
                std = 1.0
            normalized_data[column] = (df[column] - mean) / std
            self.scaler_params[column] = {'mean': mean, 'std': std}
        
        return pd.DataFrame(normalized_data, index=df.index)
    
    def inverse_normalize(self, data: np.ndarray, columns: List[str]) -> pd.DataFrame:
        """Convert normalized values back to original scale"""
        original_scale = {}
        
        for i, column in enumerate(columns):
            mean = self.scaler_params[column]['mean']
            std = self.scaler_params[column]['std']
            original_scale[column] = data[:, i] * std + mean
        
        return pd.DataFrame(original_scale)

class WellTimeSeriesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, sequence_length: int = 24, prediction_horizon: int = 1):
        """
        Create sequences from tag data
        
        Args:
            data: Preprocessed and normalized tag data
            sequence_length: Number of time steps to use for input sequence
            prediction_horizon: How many steps ahead to predict
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.data = data
        
        # Create sequences
        self.sequences = []
        for i in range(len(self.data) - sequence_length - prediction_horizon + 1):
            seq = self.data.iloc[i:(i + sequence_length)]
            target = self.data.iloc[i + sequence_length + prediction_horizon - 1:
                                  i + sequence_length + prediction_horizon]
            if not seq.empty and not target.empty:
                self.sequences.append((seq, target))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return (torch.FloatTensor(seq.values), 
                torch.FloatTensor(target.values.squeeze()))

def prepare_dataloader(data: pd.DataFrame, 
                      batch_size: int = 32, 
                      sequence_length: int = 24, 
                      prediction_horizon: int = 1,
                      shuffle: bool = True) -> DataLoader:
    """Create DataLoader for training"""
    dataset = WellTimeSeriesDataset(
        data, 
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0) 