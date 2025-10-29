from typing import Any
from fastapi import FastAPI, Body
import pandas as pd
from src.predict import load_artifacts
from src.features import preprocess

app = FastAPI(title="Churn API")

@app.post("/predict")
def predict(body: Any = Body(...)):
    """
    Accepts either:
      - {"records": [ {row}, {row}, ... ]}
      - [ {row}, {row}, ... ]
      - or a single {row} (we'll wrap into a list)
    """
    # Normalize incoming payload to a list[dict]
    records = None
    if isinstance(body, dict):
        # common wrapper keys
        for key in ("records", "data", "rows"):
            if key in body:
                records = body[key]
                break
        # if no wrapper and looks like a single row, wrap it
        if records is None:
            records = [body]
    elif isinstance(body, list):
        records = body
    else:
        records = []

    if not isinstance(records, list):
        records = [records]

    df = pd.DataFrame(records)
    # clean NAs to avoid type issues
    for c in df.columns:
        df[c] = df[c].where(pd.notna(df[c]), None)

    model, encoder, _ = load_artifacts()
    X, _, _ = preprocess(df, fit_encoder=False, encoder=encoder)
    proba = model.predict_proba(X)[:, 1]
    return {"predictions": [float(p) for p in proba]}

