"""
Utility functions for visualizing well sensor data and analysis results.
This module provides functions for creating various plots and visualizations
to help understand the patterns and anomalies in well sensor data.
"""

from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

try:
    from IPython.display import display
    IN_NOTEBOOK = True
except ImportError:
    IN_NOTEBOOK = False
    def display(x):
        print(x)

def plot_sensor_analysis(data: pd.DataFrame, tag_alias: str) -> None:
    """Comprehensive visualization for a specific tag alias.
    
    Args:
        data: Input data containing sensor readings
        tag_alias: Tag alias to analyze
    """
    # Filter data for specific tag
    tag_data = data[data['tagalias'] == tag_alias]
    
    # Create figure with grid layout
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2)
    
    # 1. Time series plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(tag_data.index, tag_data['value'], 'b-')
    ax1.set_title(f'Time Series - {tag_alias}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    
    # 2. Distribution plot
    ax2 = fig.add_subplot(gs[1, 0])
    sns.histplot(data=pd.DataFrame({'value': tag_data['value']}), x='value', kde=True, ax=ax2)
    ax2.set_title(f'Distribution - {tag_alias}')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Count')
    
    # 3. Box plot
    ax3 = fig.add_subplot(gs[1, 1])
    sns.boxplot(y=tag_data['value'], ax=ax3)
    ax3.set_title(f'Box Plot - {tag_alias}')
    
    # 4. Rolling statistics
    ax4 = fig.add_subplot(gs[2, :])
    rolling_mean = tag_data['value'].rolling(window=24).mean()
    rolling_std = tag_data['value'].rolling(window=24).std()
    
    ax4.plot(tag_data.index, tag_data['value'], 'b-', alpha=0.5, label='Original')
    ax4.plot(tag_data.index, rolling_mean, 'r-', label='24-hour Moving Average')
    ax4.fill_between(tag_data.index,
                    rolling_mean - 2*rolling_std,
                    rolling_mean + 2*rolling_std,
                    alpha=0.2, label='±2 STD')
    
    ax4.set_title(f'Rolling Statistics - {tag_alias}')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Value')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

def analyze_data_balance(data: pd.DataFrame) -> None:
    """Analyze and visualize data balance across different dimensions of the dataset.
    
    This function creates a comprehensive set of visualizations to understand:
    1. Distribution of data across different tags
    2. Temporal distribution (hourly, daily, monthly)
    3. Value ranges for each tag
    
    Args:
        data: DataFrame containing the sensor data with columns:
            - timestamp: Time of the measurement
            - tagalias: Name of the sensor/tag
            - value: Measured value
    """
    # Extract time-based features for analysis
    # Convert timestamp to datetime and extract hour, day, and month components
    data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
    data['day'] = pd.to_datetime(data['timestamp']).dt.day
    data['month'] = pd.to_datetime(data['timestamp']).dt.month
    
    # Create a figure with a 3x2 grid layout for multiple plots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2)
    
    # 1. Tag Distribution Plot
    # Shows how many samples we have for each tag/sensor
    ax1 = fig.add_subplot(gs[0, :])
    sns.countplot(data=data, x='tagalias', ax=ax1)
    ax1.set_title('Distribution of Tags')
    ax1.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability
    
    # 2. Hourly Distribution Plot
    # Shows how data is distributed across different hours of the day
    ax2 = fig.add_subplot(gs[1, 0])
    sns.countplot(data=data, x='hour', ax=ax2)
    ax2.set_title('Hourly Distribution')
    
    # 3. Daily Distribution Plot
    # Shows how data is distributed across different days of the month
    ax3 = fig.add_subplot(gs[1, 1])
    sns.countplot(data=data, x='day', ax=ax3)
    ax3.set_title('Daily Distribution')
    
    # 4. Monthly Distribution Plot
    # Shows how data is distributed across different months
    ax4 = fig.add_subplot(gs[2, 0])
    sns.countplot(data=data, x='month', ax=ax4)
    ax4.set_title('Monthly Distribution')
    
    # 5. Value Range Distribution Plot
    # Shows the distribution of values for each tag using box plots
    ax5 = fig.add_subplot(gs[2, 1])
    sns.boxplot(data=data, x='tagalias', y='value', ax=ax5)
    ax5.set_title('Value Ranges by Tag')
    ax5.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    
    # Print detailed balance metrics
    
    # 1. Tag Distribution Metrics
    print("\nData Balance Metrics:")
    print("\nTag Distribution:")
    if IN_NOTEBOOK:
        display(data['tagalias'].value_counts(normalize=True))  # Show proportions
    else:
        print(data['tagalias'].value_counts(normalize=True))
    
    # 2. Hourly Coverage Metrics
    print("\nHourly Coverage:")
    if IN_NOTEBOOK:
        display(data['hour'].value_counts(normalize=True))  # Show proportions
    else:
        print(data['hour'].value_counts(normalize=True))
    
    # 3. Daily Coverage Metrics
    print("\nDaily Coverage:")
    if IN_NOTEBOOK:
        display(data['day'].value_counts(normalize=True))  # Show proportions
    else:
        print(data['day'].value_counts(normalize=True)) 