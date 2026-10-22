
import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 1. Define folder paths
data_dir = "data/processed"
models_dir = "models"

os.makedirs(models_dir, exist_ok=True)

# 2. Load the processed data
X_train = pd.read_csv(
    os.path.join(data_dir, "X_train_processed.csv")
)
X_test = pd.read_csv(
    os.path.join(data_dir, "X_test_processed.csv")
)
y_train = pd.read_csv(
    os.path.join(data_dir, "y_train.csv")
).squeeze("columns")
y_test = pd.read_csv(
    os.path.join(data_dir, "y_test.csv")
).squeeze("columns")

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

# 3. Create the five ML models
models = {
    "logistic_regression": LogisticRegression(
        max_iter=2000, random_state=42
    ),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "decision_tree": DecisionTreeClassifier(
        random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),
    "svm": SVC(
        probability=True,
        random_state=42
    )
}

# 4. Train and save each model
for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    model_path = os.path.join(
        models_dir, f"{name}.joblib"
    )
    joblib.dump(model, model_path)

    print(f"Model saved: {model_path}")

print("\nAll five models trained successfully!")
