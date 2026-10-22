
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Load saved files
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(
        BASE_DIR / "data" / "processed" / "preprocessor.joblib"
    )
    model = joblib.load(
        BASE_DIR / "models" / "best_model.joblib"
    )
    return preprocessor, model

# Header
st.title("📊 Customer Churn Prediction")
st.write(
    "Predict whether a telecom customer is likely to leave "
    "the service using Machine Learning."
)

st.divider()

# Customer input form
st.subheader("Customer Information")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", 0, 100, 12)

    with col2:
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox(
            "Multiple Lines",
            ["No phone service", "No", "Yes"]
        )
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )

    with col3:
        support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    col4, col5 = st.columns(2)

    with col4:
        monthly = st.number_input(
            "Monthly Charges", min_value=0.0, value=70.0
        )

    with col5:
        total = st.number_input(
            "Total Charges", min_value=0.0, value=840.0
        )

    submitted = st.form_submit_button(
        "🔍 Predict Churn", use_container_width=True
    )

if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    try:
        processed = preprocessor.transform(customer)

        if hasattr(processed, "toarray"):
            processed = processed.toarray()

        # Preserve the feature order used during model training
        processed = pd.DataFrame(
            processed,
            columns=model.feature_names_in_
        )

        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0][1]

        st.divider()
        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if prediction == 1:
                st.error("⚠️ Customer may leave (Churn: Yes)")
            else:
                st.success("✅ Customer is likely to stay (Churn: No)")

        with result_col2:
            st.metric(
                "Predicted Churn Probability",
                f"{probability:.2%}"
            )
            st.progress(float(probability))

        st.caption(
            "This is a machine-learning estimate, not a guarantee. "
            "Use sample data for demonstrations."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Check that the saved preprocessor and model use "
            "the same feature order and columns."
        )
