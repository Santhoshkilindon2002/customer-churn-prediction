
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

# Load training data only
X_train = pd.read_csv(
    "data/processed/X_train_processed.csv"
)

y_train = pd.read_csv(
    "data/processed/y_train.csv"
).squeeze("columns")

# Create Logistic Regression model
model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

# 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)

print("Logistic Regression - 5-Fold Cross-Validation")

for metric in scoring:
    scores = results[f"test_{metric}"]
    print(
        f"{metric.upper()}: "
        f"{scores.mean():.4f} "
        f"+/- {scores.std():.4f}"
    )
