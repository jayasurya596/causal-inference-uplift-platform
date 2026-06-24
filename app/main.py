import os
import sys
from pathlib import Path
from typing import Literal
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure the project root is in the path so python can resolve app.models
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import wrappers to register them in sys.modules so joblib can deserialize the model
from app.models import SLearner, TLearner, XLearner

app = FastAPI(
    title="Causal Inference & Uplift Scoring API",
    description="Stateless microservice to score customers for discount eligibility based on heterogeneous treatment effects.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants for business logic (drawn from config.py)
TREATMENT_COST = 50.0  # ₹ cost of sending discount coupon
MARGIN_RATE = 0.30     # 30% margin on purchases

# Model path resolution
MODEL_PATH = BASE_DIR / "outputs" / "models" / "uplift_model.joblib"
model = None

@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        print(f"⚠️ Model not found at {MODEL_PATH}! Will try to load in-memory or fail gracefully.")
    else:
        try:
            model = joblib.load(MODEL_PATH)
            print(f"✅ Successfully loaded uplift model from {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")

class CustomerProfile(BaseModel):
    recency: float = Field(
        ..., ge=1, le=365,
        description="Days since last purchase. Range: 1 to 365 days."
    )
    frequency: int = Field(
        ..., ge=0, le=100,
        description="Purchase frequency (number of past purchases). Range: 0 to 100."
    )
    monetary: float = Field(
        ..., ge=1.0, le=50000.0,
        description="Average order value (₹). Range: ₹1.0 to ₹50,000.0."
    )
    tenure_months: float = Field(
        ..., ge=1.0, le=120.0,
        description="Customer tenure in months. Range: 1 to 120 months."
    )
    channel: Literal["app", "web", "store"] = Field(
        ...,
        description="Primary shopping channel preference."
    )
    age_group: Literal["18-25", "26-35", "36-50", "51+"] = Field(
        ...,
        description="Customer age bracket."
    )

class ScoringResponse(BaseModel):
    uplift_score: float
    expected_net_profit: float
    recommendation: Literal["Target", "Do Not Target"]
    confidence: Literal["High", "Moderate", "Low"]
    explanation: str

@app.get("/health")
def health():
    if model is None:
        return {"status": "unhealthy", "error": "Model not loaded in memory"}
    return {"status": "healthy", "model_class": model.__class__.__name__}

@app.post("/score_uplift", response_model=ScoringResponse)
def score_uplift(profile: CustomerProfile):
    global model
    if model is None:
        # Attempt to reload model once
        if MODEL_PATH.exists():
            try:
                model = joblib.load(MODEL_PATH)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Model files present but failed loading: {str(e)}")
        else:
            raise HTTPException(status_code=503, detail="Uplift scoring service is unavailable (Model not loaded)")

    # 1. Transform raw input dictionary to one-row pandas DataFrame matching the model's feature structure
    # Predefined feature order from training:
    # recency, frequency, monetary, tenure_months,
    # channel_app, channel_store, channel_web,
    # age_group_18-25, age_group_26-35, age_group_36-50, age_group_51+
    
    input_data = {
        "recency": [profile.recency],
        "frequency": [profile.frequency],
        "monetary": [profile.monetary],
        "tenure_months": [profile.tenure_months],
        "channel_app": [1.0 if profile.channel == "app" else 0.0],
        "channel_store": [1.0 if profile.channel == "store" else 0.0],
        "channel_web": [1.0 if profile.channel == "web" else 0.0],
        "age_group_18-25": [1.0 if profile.age_group == "18-25" else 0.0],
        "age_group_26-35": [1.0 if profile.age_group == "26-35" else 0.0],
        "age_group_36-50": [1.0 if profile.age_group == "36-50" else 0.0],
        "age_group_51+": [1.0 if profile.age_group == "51+" else 0.0]
    }
    
    df_features = pd.DataFrame(input_data)
    
    try:
        # 2. Score uplift (Expected incremental order value change τ(X) in ₹)
        predicted_uplift = float(model.predict(df_features)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # 3. Calculate expected net profit from treating this user
    # Net Profit = Uplift * Margin - Treatment Cost
    expected_net_profit = (predicted_uplift * MARGIN_RATE) - TREATMENT_COST
    
    # 4. Determine recommendation
    # Target only if net profit is positive (meaning discount makes money)
    recommendation = "Target" if expected_net_profit > 0 else "Do Not Target"

    # 5. Assess prediction confidence based on covariate overlap boundaries
    # (Synthetic boundaries based on typical overlap region in training data)
    is_in_bounds = (
        1 <= profile.recency <= 365 and
        0 <= profile.frequency <= 15 and
        10.0 <= profile.monetary <= 15000.0 and
        1 <= profile.tenure_months <= 60
    )
    
    if is_in_bounds:
        confidence = "High"
    elif profile.monetary > 25000.0 or profile.frequency > 20:
        # Outliers / Extreme values (unreliable support)
        confidence = "Low"
    else:
        confidence = "Moderate"

    # 6. Construct business explanation
    if recommendation == "Target":
        explanation = (
            f"This customer is expected to spend ₹{predicted_uplift:.2f} more if given the discount. "
            f"At a {MARGIN_RATE*100:.0f}% margin, this translates to ₹{predicted_uplift * MARGIN_RATE:.2f} in incremental margin. "
            f"Accounting for the ₹{TREATMENT_COST:.2f} cost, targeting this user generates an expected net profit of +₹{expected_net_profit:.2f}. "
            f"Recommend targeting."
        )
    else:
        explanation = (
            f"This customer has an expected incremental spend of ₹{predicted_uplift:.2f} (incremental margin of ₹{predicted_uplift * MARGIN_RATE:.2f}). "
            f"Since the incremental margin does not cover the ₹{TREATMENT_COST:.2f} discount cost, "
            f"targeting would result in a net loss of -₹{abs(expected_net_profit):.2f}. "
            f"Recommend withholding the discount (this is a 'Sure Thing' or a 'Lost Cause')."
        )

    return ScoringResponse(
        uplift_score=round(predicted_uplift, 2),
        expected_net_profit=round(expected_net_profit, 2),
        recommendation=recommendation,
        confidence=confidence,
        explanation=explanation
    )
