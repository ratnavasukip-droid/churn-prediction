from fastapi import FastAPI, Body
import pandas as pd
import numpy as np

# import your helpers and loaded artifacts
from src.features import preprocess
from src.predict import _model as MODEL
from src.predict import _encoder as ENCODER
from src.predict import _feature_names as FEATURE_NAMES

app = FastAPI(title="Churn API")

@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/predict")
def predict(records: list[dict] = Body(...)):
    """
    records: list of row dicts; we keep this endpoint very lenient to avoid 422s.
    """
    df = pd.DataFrame(records)
    # normalize potential numpy/object types early
    for c in df.columns:
        # replace NaNs with None for safety
        df[c] = df[c].where(pd.notna(df[c]), None)
    X, _, _ = preprocess(df, fit_encoder=False, encoder=ENCODER)
    proba = MODEL.predict_proba(X)[:, 1].astype(float).tolist()
    return {"predictions": proba}

@app.post("/explain")
def explain(record: dict = Body(...)):
    """
    record: a single row dict; also lenient.
    """
    df = pd.DataFrame([record])
    for c in df.columns:
        df[c] = df[c].where(pd.notna(df[c]), None)
    X, _, feature_names = preprocess(df, fit_encoder=False, encoder=ENCODER)

    # SHAP is heavy; simple per-feature contribution could be added later.
    # For now, we can return the same prob + feature names placeholder impacts (zeros)
    prob = float(MODEL.predict_proba(X)[0, 1])
    # if you previously added SHAP, you can reintroduce it; keeping API simple for stability
    shap_values = [0.0] * len(feature_names)
    return {
        "prob": prob,
        "feature_names": feature_names,
        "shap_values": shap_values,
    }
