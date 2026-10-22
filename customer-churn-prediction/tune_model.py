
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

X_train = pd.read_csv(
    "data/processed/X_train_processed.csv"
)

y_train = pd.read_csv(
    "data/processed/y_train.csv"
).squeeze("columns")

model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

param_grid = {
    "n_estimators": [200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

print("\nBest Parameters:")
print(search.best_params_)

print(
    f"\nBest CV Accuracy: "
    f"{search.best_score_ * 100:.2f}%"
)

joblib.dump(
    search.best_estimator_,
    "models/tuned_random_forest.joblib"
)

print("\nTuned model saved successfully!")
