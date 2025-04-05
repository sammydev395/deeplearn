import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import json
from typing import Dict, List, Optional, Union, Any, TypedDict
from datetime import datetime

class TagConfig(TypedDict):
    """Type definition for tag configuration"""
    data_type: str  # One of 'Float', 'UInt16', 'UInt32'
    units: str      # Unit of measurement (e.g., '%', 'PSI', '°C')
    description: str # Human-readable description of the tag

class TagDataProcessor:
    def __init__(self, config_path: str):
        """
        Initialize processor with tag configuration
        
        Args:
            config_path: Path to tag_config.json containing tag definitions
            
        Attributes:
            tags (Dict[str, TagConfig]): Dictionary mapping tag aliases to their configurations
            numeric_columns (List[str]): List of tag aliases that contain numeric data
            scaler_params (Dict[str, Dict[str, float]]): Parameters for data normalization
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract relevant tags (numeric sensors only)
        self.tags: Dict[str, TagConfig] = {}
        for tag in config['Tags']:
            if tag.get('DataType') in ['Float', 'UInt16', 'UInt32']:
                self.tags[tag['TagAlias']] = {
                    'data_type': tag['DataType'],
                    'units': tag.get('Units', ''),
                    'description': tag['Description']
                }
        
        self.numeric_columns = list(self.tags.keys())
        self.scaler_params: Dict[str, Dict[str, float]] = {}
        
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
        Load and preprocess tag data from a CSV file
        
        Args:
            data_path: Path to tagdata.csv containing the sensor readings
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame with numeric columns and timestamps as index
        """
        print(f"\n=== Data Loading Debug Information ===")
        print(f"Loading data from: {data_path}")
        
        # Load data from CSV
        try:
            df = pd.read_csv(data_path)
            print(f"\nRaw CSV file information:")
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"\nFirst few rows of raw data:")
            print(df.head())
            print(f"\nUnique TagAlias values in raw data: {df['TagAlias'].unique()}")
            print(f"Number of unique TagAlias values: {len(df['TagAlias'].unique())}")
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()  # Return an empty DataFrame instead of synthetic data

        # Check if we have the expected columns
        if 'TagAlias' in df.columns and 'DisplayValue' in df.columns and 'Timestamp' in df.columns:
            print("\nConverting DisplayValue to numeric...")
            # Convert DisplayValue to numeric where possible
            df['NumericValue'] = pd.to_numeric(df['DisplayValue'], errors='coerce')
            print(f"Number of non-numeric values: {df['NumericValue'].isna().sum()}")
            
            print("\nConverting timestamps...")
            # Convert timestamp
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            # Filter for numeric tags that we're interested in
            available_tags = set(df['TagAlias'].unique())
            print(f"\nAvailable tags in data: {available_tags}")
            print(f"Processor configured tags: {set(self.numeric_columns)}")
            
            usable_tags = available_tags  # Use all available tags
            #usable_tags = set(self.numeric_columns).intersection(available_tags)
            
            if not usable_tags:
                print(f"Warning: None of the configured numeric tags found in data. Available tags: {available_tags}")
                return pd.DataFrame()  # Return an empty DataFrame instead of synthetic data
            
            # Update numeric columns to only include available tags
            self.numeric_columns = list(usable_tags)
            print(f"\nUsing tags: {self.numeric_columns}")
            
            print("\nPivoting data...")
            # Pivot the data
            pivot_df = df.pivot_table(
                index='Timestamp', 
                columns='TagAlias', 
                values='NumericValue',
                aggfunc='first'
            )
            
            print(f"\nPivoted DataFrame shape: {pivot_df.shape}")
            print(f"Pivoted DataFrame columns: {pivot_df.columns.tolist()}")
            
            # Keep only numeric columns we're interested in
            pivot_df = pivot_df[self.numeric_columns]
            
            print("\nForward filling missing values...")
            # Forward fill missing values
            pivot_df = pivot_df.ffill()
            
            print("\nDropping rows with NaN values...")
            # Drop rows with any remaining NaN values
            pivot_df = pivot_df.dropna()
            print(f"Final DataFrame shape after dropping NaN: {pivot_df.shape}")
            
            print("\n=== End of Data Loading Debug Information ===\n")
            return pivot_df
        else:
            print("CSV file does not contain the expected columns.")
            print(f"Available columns: {df.columns.tolist()}")
            return pd.DataFrame()  # Return an empty DataFrame instead of synthetic data
    
    def _create_synthetic_data(self) -> List[Dict[str, Any]]:
        """
        Create synthetic data for testing when real data is not available
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing synthetic sensor readings
                Each dictionary has keys: 'Id', 'TagAlias', 'Timestamp', 'DisplayValue', 'Quality', 'WellDeviceID'
        """
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
        """
        Create synthetic dataframe for testing
        
        Returns:
            pd.DataFrame: DataFrame with synthetic sensor readings and timestamps as index
        """
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
        """
        Normalize the data using z-score standardization
        
        Args:
            df: Input DataFrame with numeric columns
            
        Returns:
            pd.DataFrame: Normalized DataFrame with same structure as input
            
        Note:
            Stores normalization parameters (mean, std) in self.scaler_params for later use
        """
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
        """
        Convert normalized values back to original scale
        
        Args:
            data: Numpy array of normalized values
            columns: List of column names corresponding to the data
            
        Returns:
            pd.DataFrame: DataFrame with values converted back to original scale
        """
        original_scale = {}
        
        for i, column in enumerate(columns):
            mean = self.scaler_params[column]['mean']
            std = self.scaler_params[column]['std']
            original_scale[column] = data[:, i] * std + mean
        
        return pd.DataFrame(original_scale)

class WellTimeSeriesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, sequence_length: int = 24, prediction_horizon: int = 1):
        """
        Create sequences from tag data for time series prediction
        
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
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset"""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence and its target from the dataset
        
        Args:
            idx: Index of the sequence to retrieve
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Input sequence and target tensors
        """
        seq, target = self.sequences[idx]
        return (torch.FloatTensor(seq.values), 
                torch.FloatTensor(target.values.squeeze()))

def prepare_dataloader(data: pd.DataFrame, 
                      batch_size: int = 32, 
                      sequence_length: int = 24, 
                      prediction_horizon: int = 1,
                      shuffle: bool = True) -> DataLoader:
    """
    Create DataLoader for training time series data
    
    Args:
        data: Preprocessed and normalized tag data
        batch_size: Number of sequences per batch
        sequence_length: Number of time steps to use for input sequence
        prediction_horizon: How many steps ahead to predict
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader: PyTorch DataLoader configured for the time series data
    """
    dataset = WellTimeSeriesDataset(
        data, 
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0) 