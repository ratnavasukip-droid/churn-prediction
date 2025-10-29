# src/train.py
import json, joblib, pandas as pd
from src.features import preprocess
from src.model import build_model

def main():
    df = pd.read_csv("data/synthetic_churn.csv")
    X, encoder, feature_names = preprocess(df, fit_encoder=True, encoder=None)

    model = build_model()
    model.fit(X, df["churn"] if "churn" in df.columns else (X.shape[0] * [0]))  # adapt if you have target column

    joblib.dump(model, "src/xgb_model.joblib")
    joblib.dump(encoder, "src/encoder.joblib")
    with open("src/feature_names.json", "w") as f:
        json.dump(feature_names, f)

    print("Saved model, encoder, and feature_names.")

if __name__ == "__main__":
    main()

