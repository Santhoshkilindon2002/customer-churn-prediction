
import pandas as pd

# Load the original dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing TotalCharges with 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Check the results
print("Missing values after cleaning:")
print(df.isnull().sum().sum())

print("\nTotalCharges data type:")
print(df["TotalCharges"].dtype)

print("\nDataset shape:")
print(df.shape)

# Create a cleaned-data folder
import os
os.makedirs("data/processed", exist_ok=True)

# Save the cleaned dataset
df.to_csv("data/processed/cleaned_churn.csv", index=False)

print("\nCleaned dataset saved successfully!")
