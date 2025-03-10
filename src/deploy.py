import torch
import pandas as pd
import json
from models.well_ts_model import WellTimeSeriesModel
from utils.data_processor import WellTimeSeriesDataset, prepare_dataloader
import numpy as np
from datetime import datetime, timedelta
import logging
from collections import deque
from typing import Dict, List
import torch.optim as optim
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertLevel:
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"

class WellPredictor:
    def __init__(self, model_path, config_path):
        """
        Initialize the predictor with a trained model and configuration
        
        Args:
            model_path: Path to the saved model file
            config_path: Path to the tag configuration file
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load the model
        self.model_path = model_path
        checkpoint = torch.load(model_path, map_location=self.device)
        input_size = len(self.config.keys())  # Number of sensors
        self.model = WellTimeSeriesModel(input_size=input_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
        
        # Initialize sequence buffer and historical data
        self.sequence_length = 24  # Same as training
        self.data_buffer = []
        self.historical_data = deque(maxlen=10000)  # Store last 10000 readings for retraining
        self.alert_history = deque(maxlen=1000)     # Store last 1000 alerts
        
        # Initialize trend analysis
        self.trend_window = 6  # Hours for trend analysis
        self.trend_thresholds = {
            'rapid_increase': 0.15,  # 15% increase
            'rapid_decrease': -0.15  # 15% decrease
        }
    
    def analyze_trend(self, sensor: str, recent_values: List[float]) -> Dict:
        """Analyze trend for a specific sensor"""
        if len(recent_values) < 2:
            return {"trend": "insufficient_data"}
            
        start_value = recent_values[0]
        end_value = recent_values[-1]
        percent_change = (end_value - start_value) / start_value if start_value != 0 else 0
        
        # Calculate rate of change
        time_diff = len(recent_values)  # in hours
        rate_of_change = percent_change / time_diff if time_diff > 0 else 0
        
        if rate_of_change > self.trend_thresholds['rapid_increase']:
            trend = "rapid_increase"
        elif rate_of_change < self.trend_thresholds['rapid_decrease']:
            trend = "rapid_decrease"
        else:
            trend = "stable"
            
        return {
            "trend": trend,
            "percent_change": percent_change,
            "rate_of_change": rate_of_change
        }
    
    def check_alerts(self, predictions, current_values) -> List[Dict]:
        """Enhanced alert checking with multiple severity levels and trend analysis"""
        alerts = []
        current_time = datetime.now()
        
        for sensor, config in self.config.items():
            idx = list(self.config.keys()).index(sensor)
            pred_value = predictions[idx]
            current_value = current_values[idx]
            
            # Get recent values for trend analysis
            recent_values = [data[idx] for data in self.data_buffer[-self.trend_window:]]
            trend_info = self.analyze_trend(sensor, recent_values)
            
            # Determine alert level
            alert_level = AlertLevel.NORMAL
            if pred_value > config['alert_threshold']:
                alert_level = AlertLevel.WARNING
            if pred_value > config['alert_threshold'] * 1.2:  # 20% above threshold
                alert_level = AlertLevel.CRITICAL
            
            # Check for anomalies using statistical analysis
            if len(recent_values) > 0:
                mean = np.mean(recent_values)
                std = np.std(recent_values)
                z_score = (pred_value - mean) / std if std > 0 else 0
                
                # Add alert if significant deviation or concerning trend
                if (alert_level != AlertLevel.NORMAL or 
                    abs(z_score) > 3 or  # More than 3 standard deviations
                    trend_info['trend'] in ['rapid_increase', 'rapid_decrease']):
                    
                    alert = {
                        'sensor': sensor,
                        'predicted_value': float(pred_value),
                        'current_value': float(current_value),
                        'threshold': config['alert_threshold'],
                        'alert_level': alert_level,
                        'timestamp': current_time.isoformat(),
                        'z_score': float(z_score),
                        'trend': trend_info,
                        'unit': config['unit']
                    }
                    alerts.append(alert)
                    self.alert_history.append(alert)
        
        return alerts
    
    def should_retrain(self) -> bool:
        """Determine if model should be retrained based on performance"""
        if len(self.historical_data) < 1000:  # Need minimum amount of new data
            return False
            
        # Check prediction accuracy
        recent_predictions = [alert['predicted_value'] for alert in self.alert_history]
        recent_actuals = [alert['current_value'] for alert in self.alert_history]
        
        if len(recent_predictions) > 100:  # Need minimum predictions to evaluate
            mse = np.mean(np.square(np.array(recent_predictions) - np.array(recent_actuals)))
            if mse > 0.2:  # Threshold for retraining
                return True
        
        return False
    
    def retrain(self, learning_rate=0.001, epochs=10):
        """Retrain the model with accumulated historical data"""
        if len(self.historical_data) < self.sequence_length:
            return {"status": "error", "message": "Insufficient data for retraining"}
            
        logger.info("Starting model retraining...")
        
        # Prepare data for training
        data_df = pd.DataFrame(self.historical_data, columns=self.config.keys())
        train_loader = prepare_dataloader(data_df, batch_size=32, sequence_length=self.sequence_length)
        
        # Setup training
        self.model.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        total_loss = 0
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_epoch_loss:.6f}")
            total_loss += avg_epoch_loss
        
        # Save retrained model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config
        }, self.model_path)
        
        self.model.eval()
        return {
            "status": "success",
            "average_loss": total_loss / epochs,
            "epochs": epochs,
            "data_points": len(self.historical_data)
        }
    
    def predict(self, current_readings):
        """Enhanced prediction with historical data storage"""
        if not all(sensor in current_readings for sensor in self.config.keys()):
            raise ValueError(f"Missing sensor readings. Required: {list(self.config.keys())}")
        
        ordered_values = [current_readings[sensor] for sensor in self.config.keys()]
        self.data_buffer.append(ordered_values)
        self.historical_data.append(ordered_values)
        
        if len(self.data_buffer) > self.sequence_length:
            self.data_buffer.pop(0)
        
        if len(self.data_buffer) < self.sequence_length:
            return None
        
        input_sequence = torch.FloatTensor(self.data_buffer).unsqueeze(0)
        input_sequence = input_sequence.to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_sequence)
            prediction = prediction.cpu().numpy().squeeze()
        
        alerts = self.check_alerts(prediction, ordered_values)
        
        # Check if retraining is needed
        needs_retraining = self.should_retrain()
        
        results = {
            'predictions': {
                sensor: float(prediction[i])
                for i, sensor in enumerate(self.config.keys())
            },
            'alerts': alerts,
            'timestamp': datetime.now().isoformat(),
            'needs_retraining': needs_retraining
        }
        
        return results

if __name__ == "__main__":
    # Example usage
    model_path = "../models/well_ts_model.pth"
    config_path = "../data/tag_config.json"
    
    try:
        predictor = WellPredictor(model_path, config_path)
        
        # Example readings
        readings = {
            "pressure": 3000,
            "temperature": 150,
            "vibration": 60,
            "depth": 8000
        }
        
        # Make prediction
        result = predictor.predict(readings)
        if result:
            print("\nPredictions:")
            for sensor, value in result['predictions'].items():
                print(f"{sensor}: {value:.2f}")
            
            if result['alerts']:
                print("\nAlerts:")
                for alert in result['alerts']:
                    print(f"⚠️ {alert['sensor']}: Predicted value {alert['predicted_value']:.2f} "
                          f"exceeds threshold {alert['threshold']}")
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}") 