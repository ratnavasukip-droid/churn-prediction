from fastapi import FastAPI, Body
import pandas as pd

from src.features import preprocess
from src.predict import load_artifacts

app = FastAPI(title="Churn API")

@app.post("/predict")
def predict(records: list[dict] = Body(...)):
    """
    Accepts: {"records": [ {...row1...}, {...row2...}, ... ]}
    Returns: {"predictions": [float, ...]}
    """
    model, encoder, _ = load_artifacts()
    # Get list from Body(...) directly or {"records": ...}
    rows = records if isinstance(records, list) else records.get("records", [])
    df = pd.DataFrame(rows)
    # normalize None/NaN
    for c in df.columns:
        df[c] = df[c].where(pd.notna(df[c]), None)
    X, _, _ = preprocess(df, fit_encoder=False, encoder=encoder)
    proba = model.predict_proba(X)[:, 1].astype(float).tolist()
    return {"predictions": proba}

@app.post("/explain")
def explain(record: dict = Body(...)):
    """
    Accepts: {"record": {...single row...}}
    Returns: simple prob + placeholder shap_values list (zeros)
    """
    model, encoder, feature_names = load_artifacts()
    row = record if isinstance(record, dict) else record.get("record", {})
    df = pd.DataFrame([row])
    for c in df.columns:
        df[c] = df[c].where(pd.notna(df[c]), None)
    X, _, feature_names = preprocess(df, fit_encoder=False, encoder=encoder)

    prob = float(model.predict_proba(X)[0, 1])
    shap_values = [0.0] * len(feature_names)  # keep simple for stability
    return {
        "prob": prob,
        "feature_names": feature_names,
        "shap_values": shap_values,
    }

