# api/app.py
from fastapi import FastAPI, Body, HTTPException
import pandas as pd
from src.predict import load_artifacts
from src.features import preprocess

app = FastAPI(title="Churn API")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(payload: dict | list = Body(...)):
    try:
        # Accept {"records":[...]} or bare list
        records = payload.get("records") if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise HTTPException(status_code=400, detail="Bad payload: expected {'records': [...]} or list of dicts")

        df = pd.DataFrame(records)
        for c in df.columns:
            df[c] = df[c].where(pd.notna(df[c]), None)

        model, encoder, _ = load_artifacts()
        X, _, _ = preprocess(df, fit_encoder=False, encoder=encoder)
        proba = model.predict_proba(X)[:, 1].astype(float).tolist()
        return {"predictions": proba}

    except Exception as e:
        # Return message so Streamlit shows the real reason
        raise HTTPException(status_code=500, detail=f"/predict failed: {type(e).__name__}: {e}")

