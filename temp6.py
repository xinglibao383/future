import kagglehub

# Download latest version
path = kagglehub.dataset_download("anonymous20251/xrfv2dataset")

print("Path to dataset files:", path)