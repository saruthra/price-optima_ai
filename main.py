import os
import io
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Set timezone to IST (UTC+5:30)
import pytz
ist = pytz.timezone('Asia/Kolkata')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("price_prediction_api")

# ===== Config =====
MODEL_PATH = os.getenv("PRICE_MODEL_PATH", "price_model_compatible.pkl")
PORT = int(os.getenv("PORT", "8000"))

# Policy parameters
STABILITY_PCT = 0.15
MIN_GM_PCT = 0.12
COMP_CAP = {"Economy": 1.05, "Premium": 1.08}
COMP_FLOOR = {"Economy": 0.90, "Premium": 0.88}
TIME_NUDGE = {"Morning": 0.03, "Afternoon": 0.0, "Evening": 0.04, "Night": 0.01}

# ===== App =====
app = FastAPI(
    title="Price Prediction API",
    version="1.0.0",
    description="API for predicting prices using trained Gradient Boosting model"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Load model =====
try:
    model = joblib.load(MODEL_PATH)
    logger.info("✅ Loaded price prediction model from %s", MODEL_PATH)
    
    if hasattr(model, 'feature_names_in_'):
        FEATURE_NAMES = model.feature_names_in_.tolist()
        logger.info("✅ Model expects features: %s", FEATURE_NAMES)
    else:
        FEATURE_NAMES = None
        logger.info("⚠️  No feature names found in model")
        
except Exception as e:
    logger.error("❌ Could not load model: %s", e)
    model = None

class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Input features for prediction")

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence: float = Field(..., description="Prediction confidence score")
    status: str
    message: str

class BatchPredictionResponse(BaseModel):
    predictions: list[Dict]
    total_processed: int
    errors: list[str] = []

# ====== Helper Functions ======
def gm_pct(price, cost):
    if price <= 0:
        return 0.0
    return (price - cost) / price

def inv_nudge(ratio):
    if ratio < 0.8:
        return 0.03
    if ratio > 1.2:
        return -0.03
    return 0.0

def row_price_bounds(row: dict):
    """Calculates price bounds based on a raw features dictionary"""
    base = float(row.get("baseline_price", row["Historical_Cost_of_Ride"]))
    cost = float(row["Historical_Cost_of_Ride"])
    veh = row.get("Vehicle_Type", "Economy")
    comp = float(row.get("competitor_price", base))

    lo = base * (1 - STABILITY_PCT)
    hi = base * (1 + STABILITY_PCT)

    base_gm = gm_pct(base, cost)
    min_gm = max(MIN_GM_PCT, base_gm)
    lo_gm = cost / max(1 - min_gm, 1e-6)

    cap = COMP_CAP.get(veh, 1.06)
    floor = COMP_FLOOR.get(veh, 0.90)
    lo_cmp = comp * floor
    hi_cmp = comp * cap

    lower = max(lo, lo_gm, lo_cmp)
    upper = min(hi, hi_cmp)

    if upper < lower:
        upper = lower

    logger.info(f"Price bounds: lower={lower}, upper={upper}")
    return lower, upper

#
# ===== THIS IS THE FULLY CORRECTED FUNCTION =====
#
def validate_features(input_data: dict) -> pd.DataFrame:
    """Validate and prepare features for prediction"""
    global FEATURE_NAMES
    if FEATURE_NAMES is None:
        logger.error("FEATURE_NAMES not loaded from model.")
        raise RuntimeError("Model feature names not loaded")

    try:
        df = pd.DataFrame([input_data])
        logger.info("Raw input features: %s", input_data)
        
        # Compute derived features
        # These ARE expected by the model (based on logs)
        df['Rider_Driver_Ratio'] = df['Number_of_Riders'] / (df['Number_of_Drivers'] + 1e-6)
        df['Driver_to_Rider_Ratio'] = df['Number_of_Drivers'] / (df['Number_of_Riders'] + 1e-6)
        df['Cost_per_Min'] = df['Historical_Cost_of_Ride'] / (df['Expected_Ride_Duration'] + 1e-6)
        df['Supply_Tightness'] = df['Number_of_Riders'] - df['Number_of_Drivers']
        df['Inventory_Health_Index'] = df['Driver_to_Rider_Ratio'] * 100
        df['baseline_price'] = df['Historical_Cost_of_Ride']
        df['price'] = df['baseline_price']  # Initial for base prediction
        
        #
        # ===== ALL ONE-HOT ENCODING AND DROPS ARE REMOVED =====
        # The model is a pipeline and expects raw strings.
        #
        
        # Add any other features the model expects but are not in the input
        for feature in FEATURE_NAMES:
            if feature not in df.columns:
                 df[feature] = 0.0  # Add as 0 or np.nan, model pipeline should handle
        
        # Reorder/select columns to *exactly* match what the model expects
        # This will pass string columns (like 'Vehicle_Type') as-is.
        try:
            df = df[FEATURE_NAMES]
        except KeyError as e:
            missing_cols = set(FEATURE_NAMES) - set(df.columns)
            logger.error("Mismatch: Code is missing features the model expects: %s", missing_cols)
            raise ValueError(f"Feature preparation failed. Missing expected model features: {missing_cols}.")

        logger.info("Processed features for model: %s", df.columns.tolist())
        return df
        
    except Exception as e:
        logger.error("Feature validation failed: %s", e)
        raise ValueError(f"Feature validation failed: {e}")


def make_prediction(features_df: pd.DataFrame, raw_features: dict) -> tuple[float, float]:
    """Make prediction using the loaded model and return optimal price with confidence"""
    if model is None:
        raise RuntimeError("Model not loaded")
    
    try:
        # Use processed_row for model features, raw_features for rules
        processed_row = features_df.iloc[0]
        
        # Use raw_features for original values
        cost = float(raw_features["Historical_Cost_of_Ride"])
        base_p = model.predict(features_df)[0]  # Initial with price = baseline_price
        
        # Pass the raw_features dict to row_price_bounds
        lo, hi = row_price_bounds(raw_features)
        
        # Use raw_features for string lookups
        t_n = TIME_NUDGE.get(raw_features.get("Time_of_Booking", "Afternoon"), 0.0)
        # Use processed_row for numeric features
        i_n = inv_nudge(processed_row["Driver_to_Rider_Ratio"]) 
        base = processed_row["baseline_price"]
        center = np.clip((lo + hi) / 2, lo, hi)
        
        n_grid = 11
        grid = np.linspace(lo, hi, n_grid)
        
        best_price = center
        best_rev = -1.0
        
        for p in grid:
            if gm_pct(p, cost) < MIN_GM_PCT:
                continue
            
            pred_df = features_df.copy()
            pred_df["price"] = p
            
            # Ensure columns are in the correct order for the model
            if FEATURE_NAMES:
                pred_df = pred_df[FEATURE_NAMES]
                
            p_now = model.predict(pred_df)[0]
            
            if p_now < base_p:
                continue
            
            rev = p * p_now
            if rev > best_rev:
                best_price = p
                best_rev = rev
        
        best_price = min(max(best_price, lo), hi)
        logger.info(f"Final price after bounds: {best_price}, bounds: [{lo}, {hi}]")
        confidence = 0.9
        
        return round(best_price, 2), confidence
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise RuntimeError(f"Prediction failed: {e}")

# ====== Endpoints ======
@app.get("/")
async def root():
    current_time = datetime.now(ist).strftime("%I:%M %p IST on %B %d, %Y")
    return {
        "message": "Price Prediction API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "current_time": current_time,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "docs": "/docs",
            "test-predict": "/test-predict"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "service": "price_prediction_api"
    }

@app.get("/model-info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    info = {
        "model_type": type(model).__name__,
        "features_expected": FEATURE_NAMES,
    }
    return info

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        features_df = validate_features(request.features)
        predicted_price, confidence = make_prediction(features_df, request.features)
        
        return PredictionResponse(
            predicted_price=predicted_price,
            confidence=round(confidence * 100, 2),
            status="success",
            message="Prediction completed successfully"
        )
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(
            status_code=400, 
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(file: UploadFile = File(...)):
    errors = []
    predictions = []
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        for index, row in df.iterrows():
            try:
                features = row.to_dict()
                features_df = validate_features(features)
                predicted_price, confidence = make_prediction(features_df, features)
                
                predictions.append({
                    "row_index": index,
                    "features": features,
                    "predicted_price": predicted_price,
                    "confidence": round(confidence * 100, 2)
                })
            except Exception as e:
                error_msg = f"Row {index}: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            errors=errors
        )
    except Exception as e:
        logger.error("Batch prediction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.post("/test-predict")
async def test_predict():
    """Test prediction with sample data from dashboard"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Test with sample data
        test_data = {
            "Number_of_Riders": 70,
            "Number_of_Drivers": 20,
            "Historical_Cost_of_Ride": 300.00,
            "competitor_price": 350.00,
            "Location_Category": "Rural",
            "Time_of_Booking": "Morning",
            "Vehicle_Type": "Economy",
            "Customer_Loyalty_Status": "Silver",
            "Expected_Ride_Duration": 120,
            # These are NOT in the model, but we pass them anyway.
            # validate_features will correctly ignore them.
            "Number_of_Past_Rides": 30,
            "Average_Ratings": 4.2
        }
        
        features_df = validate_features(test_data)
        predicted_price, confidence = make_prediction(features_df, test_data)
        
        return {
            "predicted_price": predicted_price,
            "confidence": round(confidence * 100, 2),
            "status": "success",
            "test_data": test_data
        }
    except Exception as e:
        logger.error("Test prediction failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Test prediction failed: {str(e)}")

# Error handler
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc)
    return {"detail": "Internal server error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)