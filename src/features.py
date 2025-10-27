import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

ENCODER_PATH = os.path.join(os.path.dirname(__file__), "encoder.joblib")

def load_data(path):
    return pd.read_csv(path)

def preprocess(df, fit_encoder=False):
    df = df.copy()
    df['high_charge'] = (df['monthly_charges'] > 100).astype(int)
    df['short_tenure'] = (df['tenure'] < 6).astype(int)
    cat_cols = ['contract','internet','payment_method']
    num_cols = ['tenure','monthly_charges','num_products','high_charge','short_tenure','tech_support']
    if fit_encoder:
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        enc.fit(df[cat_cols])
        joblib.dump(enc, ENCODER_PATH)
    else:
        enc = joblib.load(ENCODER_PATH)
    cat_ohe = pd.DataFrame(enc.transform(df[cat_cols]), columns=enc.get_feature_names_out(cat_cols))
    X = pd.concat([df[num_cols].reset_index(drop=True), cat_ohe.reset_index(drop=True)], axis=1)
    y = df['churn'] if 'churn' in df.columns else None
    return X, y
