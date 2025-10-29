# src/features.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

CATEGORICAL = ["contract", "internet", "payment_method"]  # adjust to your raw cols
NUMERIC = ["tenure", "monthly_charges", "tech_support"]   # adjust to your raw cols

def _build_preprocessor():
    cat = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))  # <-- key
    ])
    num = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median"))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num, NUMERIC),
            ("cat", cat, CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre

def preprocess(df: pd.DataFrame, fit_encoder: bool, encoder=None):
    df = df.copy()

    # Coerce expected columns to exist (missing → NaN)
    for col in NUMERIC + CATEGORICAL:
        if col not in df.columns:
            df[col] = pd.NA

    pre = encoder if encoder is not None else _build_preprocessor()
    if fit_encoder:
        X = pre.fit_transform(df)
        feature_names = pre.get_feature_names_out().tolist()
        return X, pre, feature_names
    else:
        X = pre.transform(df)
        feature_names = pre.get_feature_names_out().tolist()
        return X, None, feature_names

