import numpy as np
import pandas as pd
import os
os.makedirs(os.path.dirname(__file__), exist_ok=True)

def generate(n=20000, seed=42):
    np.random.seed(seed)
    customer_id = np.arange(1, n+1)
    tenure = np.random.exponential(scale=12, size=n).astype(int) + 1
    monthly_charges = np.round(np.random.normal(70, 20, size=n).clip(10, 200), 2)
    contract = np.random.choice(['month-to-month', 'one-year', 'two-year'], size=n, p=[0.5,0.3,0.2])
    internet = np.random.choice(['dsl','fiber-optic','none'], size=n, p=[0.4,0.4,0.2])
    tech_support = np.random.choice([0,1], size=n, p=[0.8,0.2])
    payment_method = np.random.choice(['electronic','mailed','bank'], size=n)
    num_products = np.random.poisson(1.2, size=n) + 1
    churn_prob = (
        0.3*(contract == 'month-to-month').astype(float) +
        0.2*(internet == 'fiber-optic').astype(float) +
        0.15*(monthly_charges/150) +
        0.1*(tenure < 6).astype(float) -
        0.2*tech_support
    )
    churn_prob = 1/(1+np.exp(- (churn_prob*3 - 1)))
    churn = (np.random.rand(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        'customer_id': customer_id,
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'contract': contract,
        'internet': internet,
        'tech_support': tech_support,
        'payment_method': payment_method,
        'num_products': num_products,
        'churn': churn
    })
    return df

if __name__ == "__main__":
    df = generate(20000)
    df.to_csv("data/synthetic_churn.csv", index=False)
    print("Wrote data/synthetic_churn.csv")
