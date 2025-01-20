from huggingface_hub import login, create_repo, upload_folder

# Replace 'your_access_token' with your actual Hugging Face token
login(token="<your_access_token>")

repo_name = "dhrumeen/small_summarization_model"
create_repo(repo_name, exist_ok=True)  #  exist_ok=True to avoid errors if it already exists

local_model_path = "./trained_model"
upload_folder(
    repo_id=repo_name,
    folder_path=local_model_path,
    commit_message="Initial model upload"
)

