import pandas as pd

file_path = "data/raw/sales.csv"

df = pd.read_csv(file_path)

# Retain only the original historical observations
original_df = df.iloc[:5000].copy()

original_df.to_csv(file_path, index=False)

print("Restored rows:", len(original_df))