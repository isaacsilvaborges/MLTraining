import kagglehub
import shutil
import os

path = kagglehub.dataset_download("samuelotiattakorah/agriculture-crop-yield")
destination = "./datasets"
shutil.copytree(path, destination, dirs_exist_ok=True)
print("Dataset saved on: ", destination)