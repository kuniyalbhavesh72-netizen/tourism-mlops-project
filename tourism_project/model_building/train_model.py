import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from huggingface_hub import login, HfApi, hf_hub_download

login(token=os.environ.get("HF_TOKEN"))

HF_USERNAME = "Bhavesh-Hug23"
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness-dataset"
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-wellness-model"

train_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="train.csv", repo_type="dataset")
test_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="test.csv", repo_type="dataset")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train, y_train = train_df.drop(columns=["ProdTaken"]), train_df["ProdTaken"]
X_test, y_test = test_df.drop(columns=["ProdTaken"]), test_df["ProdTaken"]

mlflow.set_experiment("tourism-wellness-package")

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
}

best_name, best_model, best_f1 = None, None, -1

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc, prec = accuracy_score(y_test, preds), precision_score(y_test, preds)
        rec, f1 = recall_score(y_test, preds), f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        mlflow.log_param("model_type", name)
        mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc})
        mlflow.sklearn.log_model(model, name)

        print(f"{name}: accuracy={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f} roc_auc={auc:.4f}")

        if f1 > best_f1:
            best_name, best_model, best_f1 = name, model, f1

print(f"\nBest model: {best_name} (F1={best_f1:.4f})")

os.makedirs("tourism_project/model_building", exist_ok=True)
joblib.dump(best_model, "tourism_project/model_building/best_model.joblib")
joblib.dump(list(X_train.columns), "tourism_project/model_building/feature_columns.joblib")

api = HfApi()
api.create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False, exist_ok=True)
api.upload_file(path_or_fileobj="tourism_project/model_building/best_model.joblib", path_in_repo="best_model.joblib", repo_id=MODEL_REPO_ID, repo_type="model")
api.upload_file(path_or_fileobj="tourism_project/model_building/feature_columns.joblib", path_in_repo="feature_columns.joblib", repo_id=MODEL_REPO_ID, repo_type="model")
print(f"Best model registered to https://huggingface.co/{MODEL_REPO_ID}")
