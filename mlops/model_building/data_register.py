from huggingface_hub import HfApi, create_repo
import os

create_repo(
    "bank-customer-churn",  # Your dataset repo name
    repo_type="dataset",  # Specify this is a dataset
    private=False,  # Set to True if it should be private
)

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="data",
    repo_id="praneeth232/bank-customer-churn",
    repo_type="dataset",
)
