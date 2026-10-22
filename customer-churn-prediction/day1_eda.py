
import pandas as pd

# Load dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# 1. Check missing values
print("Missing values in each column:")
print(df.isnull().sum())

# 2. Check duplicate records
print("\nDuplicate rows:")
print(df.duplicated().sum())

# 3. Check customer churn counts
print("\nChurn counts:")
print(df["Churn"].value_counts())

# 4. Check customer churn percentages
print("\nChurn percentages:")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))

# 5. Check numerical statistics
print("\nNumerical statistics:")
print(df[["tenure", "MonthlyCharges"]].describe())
