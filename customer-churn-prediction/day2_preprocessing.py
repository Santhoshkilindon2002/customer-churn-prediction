
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 1. Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_churn.csv")

# 2. Remove customer ID
df = df.drop(columns=["customerID"])

# 3. Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# 4. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Identify categorical and numerical columns
categorical_columns = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()

# 6. Create numerical preprocessing pipeline
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 7. Create categorical preprocessing pipeline
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 8. Combine both pipelines
preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# 9. Learn preprocessing rules from training data only
X_train_processed = preprocessor.fit_transform(X_train)

# 10. Apply the same rules to test data
X_test_processed = preprocessor.transform(X_test)

# 11. Create output folder
os.makedirs("data/processed", exist_ok=True)

# 12. Save processed datasets
joblib.dump(preprocessor, "data/processed/preprocessor.joblib")

# Save transformed features as CSV files
feature_names = preprocessor.get_feature_names_out()

X_train_df = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)
X_test_df = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)

X_train_df.to_csv("data/processed/X_train_processed.csv", index=False)
X_test_df.to_csv("data/processed/X_test_processed.csv", index=False)

y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Preprocessing completed successfully!")
print("Training data shape:", X_train_processed.shape)
print("Testing data shape:", X_test_processed.shape)
print("Target training shape:", y_train.shape)
print("Target testing shape:", y_test.shape)
print("Preprocessor saved successfully!")
