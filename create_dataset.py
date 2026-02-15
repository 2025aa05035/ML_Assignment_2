import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Get project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create data folder
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Load dataset
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

# Save CSV
csv_path = os.path.join(DATA_DIR, "breast_cancer.csv")
df.to_csv(csv_path, index=False)

print("Dataset created at:", csv_path)
