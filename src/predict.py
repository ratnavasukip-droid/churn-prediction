import joblib
import json
from pathlib import Path
import pandas as pd
from src.features import preprocess

MODEL_PATH = Path("src/xgb_model.joblib")
ENC_PATH = Path("src/encoder.joblib")
FEAT_PATH = Path("src/feature_names.json")

_model = None
_encoder = None
_feature_names = None

def load_artifacts():
    global _model, _encoder, _feature_names
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _encoder = joblib.load(ENC_PATH)
        with FEAT_PATH.open() as f:
            _feature_names = json.load(f)
    return _model, _encoder, _feature_names

def predict_df(df: pd.DataFrame):
    model, encoder, _ = load_artifacts()
    X, _, _ = preprocess(df, fit_encoder=False, encoder=encoder)
    return model.predict_proba(X)[:, 1]

