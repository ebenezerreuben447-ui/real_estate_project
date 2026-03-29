import streamlit as st
import requests

# Backend API URL
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction App")

st.write("Enter house details to predict price")

# Input fields
GrLivArea = st.number_input("Living Area (sq ft)", min_value=100.0)
BedroomAbvGr = st.number_input("Bedrooms Above Ground", min_value=1)
Neighborhood = st.text_input("Neighborhood")
YearBuilt = st.number_input("Year Built", min_value=1900)
YearRemodAdd = st.number_input("Year Remodeled", min_value=1900)
YrSold = st.number_input("Year Sold", min_value=2000)
TotalBsmtSF = st.number_input("Basement Area (sq ft)", min_value=0.0)

# Button
if st.button("Predict Price"):
    input_data = {
        "GrLivArea": GrLivArea,
        "BedroomAbvGr": BedroomAbvGr,
        "Neighborhood": Neighborhood,
        "YearBuilt": YearBuilt,
        "YearRemodAdd": YearRemodAdd,
        "YrSold": YrSold,
        "TotalBsmtSF": TotalBsmtSF
    }

    try:
        response = requests.post(API_URL, json=input_data)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Price: ${result['Predicted House Price']:,.2f}")
        else:
            st.error("Error from API")

    except Exception as e:
        st.error(f"Failed to connect: {e}")