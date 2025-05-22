import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# 🎯 Wczytaj model, label encodery i scaler
model = joblib.load("xgb.pkl")
encoder_dict = joblib.load("le.pkl")   # dict: {col_name: LabelEncoder}
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Customer Churn Prediction")
st.title("📉 Customer Churn Prediction")

# 📥 Interfejs użytkownika
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)

phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# 🧾 Tworzymy DataFrame z inputem użytkownika
input_data = pd.DataFrame([{
    'gender': gender,
    'SeniorCitizen': senior,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}])

# 🧠 Label Encoding
for col in input_data.columns:
    if col in encoder_dict:
        input_data[col] = encoder_dict[col].transform(input_data[col])

# 📏 Skalowanie zmiennych numerycznych
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
input_data[num_cols] = scaler.transform(input_data[num_cols])

# 🔮 Predykcja
if st.button("Predict"):
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    st.markdown(f"### 🧾 Prediction: {'✅ Churn' if pred == 1 else '❌ No Churn'}")
    st.markdown(f"**Probability of churn: {proba:.2%}**")

    # 🎨 SHAP Explanation
    with st.spinner("Calculating SHAP values..."):
        explainer = shap.Explainer(model)
        shap_values = explainer(input_data)
        st.subheader("SHAP Explanation (Waterfall)")
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(plt.gcf())
