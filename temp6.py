# The following code will only execute
# successfully when compression is complete

import kagglehub

# Download latest version
path = kagglehub.dataset_download("yrrr666/fake-review")

print("Path to dataset files:", path)