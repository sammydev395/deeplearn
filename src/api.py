from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List
from deploy import WellPredictor, AlertLevel
import logging
import uvicorn
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Well Sensor Predictor API")

# Initialize predictor
try:
    predictor = WellPredictor(
        model_path="models/well_ts_model.pth",
        config_path="data/tag_config.json"
    )
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    predictor = None

class SensorReadings(BaseModel):
    readings: Dict[str, float]

class RetrainingConfig(BaseModel):
    learning_rate: Optional[float] = 0.001
    epochs: Optional[int] = 10

@app.post("/predict")
async def predict(data: SensorReadings):
    """
    Make predictions based on current sensor readings
    
    Example request body:
    {
        "readings": {
            "pressure": 3000,
            "temperature": 150,
            "vibration": 60,
            "depth": 8000
        }
    }
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        result = predictor.predict(data.readings)
        if result is None:
            return {
                "status": "waiting",
                "message": f"Collecting initial sequence data. {len(predictor.data_buffer)}/{predictor.sequence_length} readings collected."
            }
        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts")
async def get_alerts(
    severity: Optional[str] = None,
    sensor: Optional[str] = None,
    hours: Optional[int] = 24
):
    """
    Get historical alerts with optional filtering
    
    Parameters:
    - severity: Filter by alert level (normal, warning, critical)
    - sensor: Filter by specific sensor
    - hours: Number of hours of history to return (default 24)
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    alerts = []
    for alert in predictor.alert_history:
        alert_time = datetime.fromisoformat(alert['timestamp'])
        if alert_time < cutoff_time:
            continue
            
        if severity and alert['alert_level'] != severity:
            continue
            
        if sensor and alert['sensor'] != sensor:
            continue
            
        alerts.append(alert)
    
    return {
        "alerts": alerts,
        "total_count": len(alerts),
        "filter_criteria": {
            "severity": severity,
            "sensor": sensor,
            "hours": hours
        }
    }

@app.get("/trends")
async def get_trends(hours: Optional[int] = 6):
    """
    Get trend analysis for all sensors over specified time period
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    trends = {}
    for sensor in predictor.config.keys():
        recent_values = [
            data[list(predictor.config.keys()).index(sensor)] 
            for data in predictor.data_buffer[-hours:]
        ]
        trends[sensor] = predictor.analyze_trend(sensor, recent_values)
    
    return {
        "trends": trends,
        "analysis_period_hours": hours
    }

@app.post("/retrain")
async def retrain_model(config: RetrainingConfig):
    """
    Trigger model retraining with optional parameters
    """
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        result = predictor.retrain(
            learning_rate=config.learning_rate,
            epochs=config.epochs
        )
        return result
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check if the service is healthy and model is loaded"""
    if predictor is None:
        return {
            "status": "unhealthy",
            "model_loaded": False
        }
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "data_points_collected": len(predictor.historical_data),
        "alerts_in_history": len(predictor.alert_history),
        "needs_retraining": predictor.should_retrain()
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 