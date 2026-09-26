import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import login, HfApi

login(token=os.environ.get("HF_TOKEN"))

HF_USERNAME = "Bhavesh-Hug23"
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness-dataset"

df = pd.read_csv("tourism_project/data/tourism.csv")

# Drop identifier / index columns that carry no predictive signal
df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns])

numeric_cols = [c for c in df.select_dtypes(include=["int64", "float64"]).columns if c != "ProdTaken"]
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Impute missing values: median for numeric, mode for categorical
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# One-hot encode categorical features
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["ProdTaken"]
)

os.makedirs("tourism_project/data", exist_ok=True)
train_df.to_csv("tourism_project/data/train.csv", index=False)
test_df.to_csv("tourism_project/data/test.csv", index=False)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

api = HfApi()
api.create_repo(repo_id=DATASET_REPO_ID, repo_type="dataset", private=False, exist_ok=True)
api.upload_file(path_or_fileobj="tourism_project/data/train.csv", path_in_repo="train.csv", repo_id=DATASET_REPO_ID, repo_type="dataset")
api.upload_file(path_or_fileobj="tourism_project/data/test.csv", path_in_repo="test.csv", repo_id=DATASET_REPO_ID, repo_type="dataset")
print("train.csv and test.csv uploaded to the Hugging Face dataset repo.")
