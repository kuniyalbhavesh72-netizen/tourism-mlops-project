import os
from huggingface_hub import login, HfApi

login(token=os.environ.get("HF_TOKEN"))

HF_USERNAME = "Bhavesh-Hug23"
SPACE_REPO_ID = f"{HF_USERNAME}/tourism-wellness-app"

api = HfApi()
api.create_repo(repo_id=SPACE_REPO_ID, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
api.upload_folder(folder_path="tourism_project/deployment", repo_id=SPACE_REPO_ID, repo_type="space")
print(f"App pushed to https://huggingface.co/spaces/{SPACE_REPO_ID}")
