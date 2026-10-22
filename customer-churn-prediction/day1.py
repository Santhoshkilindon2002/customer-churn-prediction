
import pandas as pd
import numpy as np

# Load the customer churn dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

print("Dataset loaded successfully!")
print("Number of rows and columns:", df.shape)

print("\nFirst five customer records:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()
