import streamlit as st
import pandas as pd
import requests
import numpy as np

API_URL = "https://churn-prediction-1-73fit.onrender.com"

st.title("Customer Churn Explorer")

uploaded_file = st.file_uploader("Upload CSV with customer fields", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview")
    st.dataframe(df.head())

    if st.button("Run Predictions"):
        try:
            resp = requests.post(f"{API_URL}/predict",
                                json={"records": df.to_dict(orient="records")},
                                timeout=60)
            resp.raise_for_status()
            preds = pd.DataFrame(resp.json()["predictions"], columns=["churn_probability"])
            st.dataframe(pd.concat([df, preds], axis=1))
        except Exception as e:
            st.error(f"API error: {e}")

    st.write("---")
    st.subheader("🔎 Explain a single prediction (SHAP)")
    row_idx = st.number_input("Select row index:", 0, len(df)-1)

    if st.button("Explain this row"):
        row = df.iloc[int(row_idx)].to_dict()
        try:
            resp = requests.post(f"{API_URL}/explain",
                                json={"record": row},
                                timeout=60)
            resp.raise_for_status()
            res = resp.json()

            st.write(f"Predicted churn probability: {res['prob']:.3f}")

            feature_names = res["feature_names"]
            shap_values = res["shap_values"]

            idx = np.argsort(np.abs(shap_values))[::-1][:10]
            st.bar_chart(pd.DataFrame({
                "impact": np.array(shap_values)[idx]
            }, index=np.array(feature_names)[idx]))

        except Exception as e:
            st.error(f"Explain error: {e}")
