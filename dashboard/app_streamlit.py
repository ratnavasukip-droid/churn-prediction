import streamlit as st
import pandas as pd
import requests

st.title("Customer Churn Explorer")

uploaded = st.file_uploader("Upload CSV with customer fields", type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview", df.head())
    if st.button("Run predictions (local API expected at :8000)"):
        rows = []
        for _, r in df.iterrows():
            payload = {
                "tenure": int(r.get("tenure", 12)),
                "monthly_charges": float(r.get("monthly_charges", 70.0)),
                "contract": r.get("contract", "month-to-month"),
                "internet": r.get("internet", "dsl"),
                "tech_support": int(r.get("tech_support", 0)),
                "payment_method": r.get("payment_method", "electronic"),
                "num_products": int(r.get("num_products", 1))
            }
            try:
                res = requests.post("http://localhost:8000/predict", json=payload, timeout=5)
                prob = res.json().get("churn_prob")
            except Exception:
                prob = None
            rows.append({**payload, "churn_prob": prob})
        st.write(pd.DataFrame(rows))
else:
    st.info("Upload a CSV or run data/generate_synthetic.py to create a sample dataset.")
