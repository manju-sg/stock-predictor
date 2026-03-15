import kagglehub
import os

path = kagglehub.dataset_download("nelgiriyewithana/world-stock-prices-daily-updating")
print("Dataset downloaded to:", path)
print("Files in dataset:", os.listdir(path))
