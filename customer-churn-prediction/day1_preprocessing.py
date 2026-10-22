
import pandas as pd

# Load the original dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

# Check TotalCharges before conversion
print("TotalCharges data type before conversion:")
print(df["TotalCharges"].dtype)

# Convert TotalCharges to numeric
# Invalid or blank text values become NaN
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Check how many missing values exist after conversion
print("\nMissing TotalCharges after conversion:")
print(df["TotalCharges"].isnull().sum())

# Display rows with missing TotalCharges
print("\nRows with missing TotalCharges:")
print(df[df["TotalCharges"].isnull()][
    ["customerID", "tenure", "TotalCharges", "Churn"]
])

# Display the corrected data type
print("\nTotalCharges data type after conversion:")
print(df["TotalCharges"].dtype)