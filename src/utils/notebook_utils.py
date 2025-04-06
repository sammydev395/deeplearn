"""
Utility functions for well time series analysis notebook.
"""

from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

def detect_outliers(
    data: pd.DataFrame,
    columns: List[str],
    methods: List[str] = ['zscore', 'iqr', 'isolation_forest']
) -> Dict[str, np.ndarray]:
    """Detect outliers using multiple methods.
    
    Args:
        data: Input DataFrame containing sensor readings
        columns: List of column names to check for outliers
        methods: List of outlier detection methods to use. Available methods:
            - 'zscore': Uses standard deviation from mean
            - 'iqr': Uses interquartile range
            - 'isolation_forest': Uses Isolation Forest algorithm
    
    Returns:
        Dictionary mapping method names to arrays of outlier indices
    """
    results: Dict[str, np.ndarray] = {}
    
    if 'zscore' in methods:
        # Z-score method (assumes normal distribution)
        z_scores = np.abs(stats.zscore(data[columns]))
        results['zscore'] = np.where(z_scores > 3)[0]
    
    if 'iqr' in methods:
        # IQR method
        Q1 = data[columns].quantile(0.25)
        Q3 = data[columns].quantile(0.75)
        IQR = Q3 - Q1
        results['iqr'] = data[
            ((data[columns] < (Q1 - 1.5 * IQR)) | 
             (data[columns] > (Q3 + 1.5 * IQR))).any(axis=1)
        ].index.to_numpy()
    
    if 'isolation_forest' in methods:
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        results['isolation_forest'] = np.where(
            iso_forest.fit_predict(data[columns]) == -1
        )[0]
    
    return results

def compare_outlier_methods(data: pd.DataFrame, column_name: str) -> None:
    """Compare different outlier detection methods visually.
    
    Args:
        data: DataFrame containing the sensor data
        column_name: Name of the column to analyze
    """
    plt.figure(figsize=(15, 10))
    
    # Original Data
    plt.subplot(2, 2, 1)
    plt.scatter(range(len(data)), data[column_name], c='blue', alpha=0.5)
    plt.title('Original Data')
    plt.xlabel('Index')
    plt.ylabel(column_name)
    
    # Z-Score Method
    z_scores = np.abs(stats.zscore(data[column_name]))
    outliers_z = z_scores > 3
    
    plt.subplot(2, 2, 2)
    plt.scatter(range(len(data)), data[column_name], c='blue', alpha=0.5)
    plt.scatter(np.where(outliers_z)[0], data[column_name][outliers_z], 
                c='red', alpha=0.7, label='Outliers')
    plt.title('Z-Score Method')
    plt.xlabel('Index')
    plt.ylabel(column_name)
    plt.legend()
    
    # IQR Method
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    outliers_iqr = (data[column_name] < (Q1 - 1.5 * IQR)) | (data[column_name] > (Q3 + 1.5 * IQR))
    
    plt.subplot(2, 2, 3)
    plt.scatter(range(len(data)), data[column_name], c='blue', alpha=0.5)
    plt.scatter(np.where(outliers_iqr)[0], data[column_name][outliers_iqr], 
                c='red', alpha=0.7, label='Outliers')
    plt.title('IQR Method')
    plt.xlabel('Index')
    plt.ylabel(column_name)
    plt.legend()
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outliers_if = iso_forest.fit_predict(data[[column_name]]) == -1
    
    plt.subplot(2, 2, 4)
    plt.scatter(range(len(data)), data[column_name], c='blue', alpha=0.5)
    plt.scatter(np.where(outliers_if)[0], data[column_name][outliers_if], 
                c='red', alpha=0.7, label='Outliers')
    plt.title('Isolation Forest')
    plt.xlabel('Index')
    plt.ylabel(column_name)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nOutlier Detection Statistics for {column_name}:")
    print(f"Z-Score outliers: {sum(outliers_z)} ({sum(outliers_z)/len(data)*100:.2f}%)")
    print(f"IQR outliers: {sum(outliers_iqr)} ({sum(outliers_iqr)/len(data)*100:.2f}%)")
    print(f"Isolation Forest outliers: {sum(outliers_if)} ({sum(outliers_if)/len(data)*100:.2f}%)")

def detect_time_series_outliers(
    data: pd.DataFrame,
    tag_alias: str,
    value_column: str = 'value',
    timestamp_column: str = 'timestamp',
    window_size: int = 24,
    threshold: float = 3.0
) -> Dict[str, pd.Series]:
    """Detect outliers in time series data using multiple methods.
    
    Args:
        data: DataFrame containing time series data
        tag_alias: Name of the tag/sensor to analyze
        value_column: Name of the column containing sensor values
        timestamp_column: Name of the column containing timestamps
        window_size: Size of the rolling window for statistics
        threshold: Number of standard deviations for outlier detection
    
    Returns:
        Dictionary containing:
        - 'rolling_zscore': Boolean series indicating outliers from rolling z-score method
        - 'seasonal_decomposition': Boolean series indicating outliers from seasonal decomposition
        - 'ma_std': Boolean series indicating outliers from moving average method
    """
    # Filter data for specific tag
    tag_data = data[data['tagalias'] == tag_alias].copy()
    tag_data[timestamp_column] = pd.to_datetime(tag_data[timestamp_column])
    tag_data = tag_data.set_index(timestamp_column)
    tag_data = tag_data.sort_index()
    
    results = {}
    
    # 1. Rolling Z-Score Method
    def rolling_zscore(series: pd.Series, window: int = 24) -> pd.Series:
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        z_scores = np.abs((series - rolling_mean) / rolling_std)
        return pd.Series(z_scores > 3, index=series.index)
    
    results['rolling_zscore'] = rolling_zscore(tag_data[value_column])
    
    # 2. Seasonal Decomposition Method
    # Resample data to ensure regular time intervals (hourly)
    resampled_data = tag_data[value_column].resample('H').mean().interpolate()
    
    # Perform seasonal decomposition
    try:
        decomposition = seasonal_decompose(
            resampled_data,
            period=24,  # 24 hours for daily seasonality
            extrapolate_trend=True
        )
        
        # Calculate residuals
        residuals = decomposition.resid
        residuals_std = np.std(residuals)
        results['seasonal_decomposition'] = np.abs(residuals) > 2 * residuals_std
    except Exception as e:
        print(f"Could not perform seasonal decomposition for {tag_alias}: {str(e)}")
        results['seasonal_decomposition'] = pd.Series(False, index=resampled_data.index)
    
    # 3. Moving Average with Standard Deviation Method
    def ma_std_outliers(series: pd.Series, window: int = 24, threshold: float = 2.0) -> pd.Series:
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        lower_bound = rolling_mean - threshold * rolling_std
        upper_bound = rolling_mean + threshold * rolling_std
        return (series < lower_bound) | (series > upper_bound)
    
    results['ma_std'] = ma_std_outliers(tag_data[value_column])
    
    # Visualize results
    plt.figure(figsize=(15, 12))
    
    # Original Time Series with Rolling Z-Score Outliers
    plt.subplot(3, 1, 1)
    plt.plot(tag_data.index, tag_data[value_column], 'b-', label='Original', alpha=0.5)
    plt.scatter(tag_data.index[results['rolling_zscore']], 
                tag_data[value_column][results['rolling_zscore']], 
                c='red', label='Rolling Z-Score Outliers')
    plt.title(f'Rolling Z-Score Outliers - {tag_alias}')
    plt.legend()
    
    # Time Series with Seasonal Decomposition Outliers
    plt.subplot(3, 1, 2)
    plt.plot(resampled_data.index, resampled_data, 'b-', label='Original', alpha=0.5)
    plt.scatter(resampled_data.index[results['seasonal_decomposition']], 
                resampled_data[results['seasonal_decomposition']], 
                c='red', label='Seasonal Decomposition Outliers')
    plt.title('Seasonal Decomposition Outliers')
    plt.legend()
    
    # Time Series with Moving Average Outliers
    plt.subplot(3, 1, 3)
    plt.plot(tag_data.index, tag_data[value_column], 'b-', label='Original', alpha=0.5)
    plt.scatter(tag_data.index[results['ma_std']], 
                tag_data[value_column][results['ma_std']], 
                c='red', label='MA-STD Outliers')
    plt.title('Moving Average with STD Outliers')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nOutlier Detection Summary for {tag_alias}:")
    print(f"Rolling Z-Score Outliers: {sum(results['rolling_zscore'])} "
          f"({sum(results['rolling_zscore'])/len(tag_data)*100:.2f}%)")
    print(f"Seasonal Decomposition Outliers: {sum(results['seasonal_decomposition'])} "
          f"({sum(results['seasonal_decomposition'])/len(resampled_data)*100:.2f}%)")
    print(f"Moving Average-STD Outliers: {sum(results['ma_std'])} "
          f"({sum(results['ma_std'])/len(tag_data)*100:.2f}%)")
    
    return results

def analyze_seasonal_patterns(
    data: pd.DataFrame,
    tag_alias: str,
    value_column: str = 'value',
    timestamp_column: str = 'timestamp'
) -> None:
    """Analyze and visualize seasonal patterns in the time series data.
    
    Args:
        data: DataFrame containing time series data
        tag_alias: Name of the tag/sensor to analyze
        value_column: Name of the column containing sensor values
        timestamp_column: Name of the column containing timestamps
    """
    # Filter data for specific tag
    tag_data = data[data['tagalias'] == tag_alias].copy()
    tag_data[timestamp_column] = pd.to_datetime(tag_data[timestamp_column])
    tag_data = tag_data.set_index(timestamp_column)
    tag_data = tag_data.sort_index()
    
    # Resample data to hourly intervals
    hourly_data = tag_data[value_column].resample('H').mean().interpolate()
    
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(
        hourly_data,
        period=24,  # 24 hours for daily seasonality
        extrapolate_trend=True
    )
    
    # Plot decomposition
    plt.figure(figsize=(15, 12))
    
    plt.subplot(4, 1, 1)
    plt.plot(hourly_data.index, hourly_data)
    plt.title(f'Original Time Series - {tag_alias}')
    
    plt.subplot(4, 1, 2)
    plt.plot(decomposition.trend)
    plt.title('Trend')
    
    plt.subplot(4, 1, 3)
    plt.plot(decomposition.seasonal)
    plt.title('Seasonal')
    
    plt.subplot(4, 1, 4)
    plt.plot(decomposition.resid)
    plt.title('Residuals')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and plot hourly patterns
    hourly_patterns = hourly_data.groupby(hourly_data.index.hour).mean()
    hourly_std = hourly_data.groupby(hourly_data.index.hour).std()
    
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_patterns.index, hourly_patterns.values, 'b-', label='Mean')
    plt.fill_between(hourly_patterns.index,
                    hourly_patterns.values - hourly_std.values,
                    hourly_patterns.values + hourly_std.values,
                    alpha=0.2, label='±1 STD')
    plt.title(f'Hourly Patterns - {tag_alias}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show() 