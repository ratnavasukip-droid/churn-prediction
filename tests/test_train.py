import os
from src.train import main

def test_training_runs(tmp_path):
    from data.generate_synthetic import generate
    df = generate(n=100)
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    main(str(p), exp_name="test_churn")
    assert os.path.exists("src/xgb_model.joblib")
