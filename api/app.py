from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title="Churn Prediction API")

class Customer(BaseModel):
    tenure: int
    monthly_charges: float
    contract: str
    internet: str
    tech_support: int
    payment_method: str
    num_products: int

@app.on_event("startup")
def load_model():
    global model
    from src.model import ChurnModel
    model = ChurnModel()

@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.dict()])
    probs = model.predict(df)
    return {"churn_prob": float(probs[0])}
