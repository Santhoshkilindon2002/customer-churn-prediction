
import joblib
import pandas as pd

# Load saved preprocessor and trained model
preprocessor = joblib.load("data/processed/preprocessor.joblib")
model = joblib.load("models/best_model.joblib")

# Sample new customer details
customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}])

processed_customer = preprocessor.transform(customer)

if hasattr(processed_customer, "toarray"):
    processed_customer = processed_customer.toarray()

processed_customer = pd.DataFrame(
    processed_customer,
    columns=model.feature_names_in_
)

prediction = model.predict(processed_customer)[0]
probability = model.predict_proba(processed_customer)[0][1]
print("\n--- Customer Churn Prediction ---")

if prediction == 1:
    print("Prediction: Customer may leave (Churn: Yes)")
else:
    print("Prediction: Customer is likely to stay (Churn: No)")

print(f"Predicted churn probability: {probability:.2%}")
