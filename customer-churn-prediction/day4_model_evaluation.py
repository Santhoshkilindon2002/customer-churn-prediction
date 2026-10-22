
import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# 1. Load test data
data_dir = "data/processed"
models_dir = "models"

X_test = pd.read_csv(
    os.path.join(data_dir, "X_test_processed.csv")
)

y_test = pd.read_csv(
    os.path.join(data_dir, "y_test.csv")
).squeeze("columns")

# 2. List of trained models
model_names = [
    "logistic_regression",
    "knn",
    "decision_tree",
    "random_forest",
    "svm"
]

# 3. Evaluate every saved model
results = []

for name in model_names:
    print(f"\nEvaluating {name}...")

    model = joblib.load(
        os.path.join(models_dir, f"{name}.joblib")
    )

    # Predict using test data
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    # ROC-AUC using predicted probabilities
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    roc_auc = roc_auc_score(y_test, y_score)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

# 4. Save comparison results
results_df = pd.DataFrame(results)
results_df.to_csv(
    "model_evaluation_results.csv",
    index=False
)

print("\nModel comparison:")
print(
    results_df.sort_values(
        by="F1 Score", ascending=False
    ).to_string(index=False)
)

print("\nResults saved to model_evaluation_results.csv")
