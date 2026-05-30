import kagglehub
import zipfile
import os

# 指定目标路径
target_dir = "/Data/user_3505_9/StyleGaussian-main/datasets/wikiart"

# 下载数据集
print("Downloading dataset...")
path = kagglehub.dataset_download("ipythonx/wikiart-gangogh-creating-art-gan", force_download=True)

# 解压到目标路径
if path.endswith(".zip"):
    print("Extracting dataset...")
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
else:
    print("Dataset is not a zip file, please check the downloaded content.")

print("Path to dataset files:", target_dir)