
import joblib
import pandas as pd
import os

from sklearn.linear_model import LogisticRegression

# Load training data only
X_train = pd.read_csv(
    "data/processed/X_train_processed.csv"
)

y_train = pd.read_csv(
    "data/processed/y_train.csv"
).squeeze("columns")

# Create and train final baseline model
best_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

best_model.fit(X_train, y_train)

# Save the model
os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/best_model.joblib"
)

print("Final baseline model trained and saved!")
print("Model path: models/best_model.joblib")
