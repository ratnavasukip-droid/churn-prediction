import mlflow
import mlflow.sklearn
import xgboost as xgb
import pandas as pd
import joblib
from src.features import load_data, preprocess
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgb_model.joblib")

def main(data_path="data/synthetic_churn.csv", exp_name="churn_experiment"):
    df = load_data(data_path)
    X, y = preprocess(df, fit_encoder=True)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 42,
        'max_depth': 6,
        'eta': 0.1
    }
    mlflow.set_experiment(exp_name)
    with mlflow.start_run():
        bst = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtest, 'eval')], early_stopping_rounds=20, verbose_eval=False)
        preds = bst.predict(dtest)
        from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
        auc = roc_auc_score(y_test, preds)
        pred_labels = (preds > 0.5).astype(int)
        acc = accuracy_score(y_test, pred_labels)
        prec = precision_score(y_test, pred_labels)
        rec = recall_score(y_test, pred_labels)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        joblib.dump(bst, MODEL_PATH)
        mlflow.sklearn.log_model(bst, "xgb_model")
    print(f"Trained model: AUC={auc:.4f}, acc={acc:.4f}")

if __name__ == "__main__":
    main()
