import pandas as pd
from src.model import ChurnModel

def predict_from_csv(path):
    df = pd.read_csv(path)
    model = ChurnModel()
    preds = model.predict(df)
    df['churn_prob'] = preds
    return df

if __name__ == "__main__":
    out = predict_from_csv("data/synthetic_churn.csv")
    print(out.head())
