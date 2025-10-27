import joblib
import os
from src.features import preprocess
import xgboost as xgb
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgb_model.joblib")

class ChurnModel:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
    def predict(self, raw_df):
        X, _ = preprocess(raw_df, fit_encoder=False)
        try:
            preds = self.model.predict(xgb.DMatrix(X))
        except Exception:
            preds = self.model.predict_proba(X)[:,1]
        return preds
