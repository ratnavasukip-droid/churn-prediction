from fastapi import FastAPI, Body
import pandas as pd
from src.features import preprocess
from src.predict import load_artifacts  # lazy load model/encoder/feature_names

app = FastAPI(title="Churn API")

@app.post("/predict")
def predict(payload: dict | list = Body(...)):
    """
    Accepts either:
      - {"records": [ {...}, {...} ]}  (preferred)
      - [ {...}, {...} ]               (backward compatible)
    Returns: {"predictions": [float, ...]}
    """
    # Normalize input shape
    if isinstance(payload, dict) and "records" in payload:
        records = payload["records"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("Body must be {'records': [...]} or a bare list of records")

    df = pd.DataFrame(records)
    df = df.where(pd.notna(df), None)  # replace NaN with None

    model, encoder, _ = load_artifacts()
    X, _, _ = preprocess(df, fit_encoder=False, encoder=encoder)
    proba = model.predict_proba(X)[:, 1].astype(float).tolist()
    return {"predictions": proba}

@app.post("/explain")
def explain(record: dict = Body(...)):
    """
    Accepts: {"record": {...}}  OR a bare record dict
    Returns: {"prob": float, "feature_names": [...], "shap_values": [...]}
    (SHAP values are placeholder zeros for now; we can enable real SHAP later.)
    """
    # Normalize input shape
    if isinstance(record, dict) and "record" in record:
        row = record["record"]
    else:
        row = record

    df = pd.DataFrame([row])
    df = df.where(pd.notna(df), None)

    model, encoder, feature_names = load_artifacts()
    X, _, feature_names = preprocess(df, fit_encoder=False, encoder=encoder)
    prob = float(model.predict_proba(X)[0, 1])

    # placeholder impacts (we’ll wire true SHAP after everything is stable)
    shap_values = [0.0] * len(feature_names)

    return {
        "prob": prob,
        "feature_names": feature_names,
        "shap_values": shap_values,
    }

