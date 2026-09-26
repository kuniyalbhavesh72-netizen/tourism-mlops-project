import os
from huggingface_hub import login, HfApi

login(token=os.environ.get("HF_TOKEN"))

HF_USERNAME = "Bhavesh-Hug23"
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness-dataset"

api = HfApi()
api.create_repo(repo_id=DATASET_REPO_ID, repo_type="dataset", private=False, exist_ok=True)
api.upload_file(
    path_or_fileobj="tourism_project/data/tourism.csv",
    path_in_repo="tourism.csv",
    repo_id=DATASET_REPO_ID,
    repo_type="dataset"
)
print("Raw tourism.csv registered to the Hugging Face dataset repo.")
