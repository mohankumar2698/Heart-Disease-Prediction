import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# 1. Page Configuration
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

# 2. Cached Asset Loading
@st.cache_resource
def load_deep_learning_assets():
    # Keras models use load_model; Scalers still use joblib
    model = load_model("models/heart_model.h5") # or .keras / .pkl if saved as weights
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

try:
    model, scaler = load_deep_learning_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# 3. UI Layout
st.title("❤️ Heart Disease Prediction Dashboard")
# ... (your input fields go here) ...

# 4. Neural Network Prediction Logic
if st.button("Predict"):
    # Construct input array matching your training features
    data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang]])
    
    # Scale features
    scaled_data = scaler.transform(data)
    
    # Neural network inference
    probability = model.predict(scaled_data)[0][0]
    
    if probability > 0.5:
        st.error(f"High Risk ({probability*100:.2f}%)")
    else:
        st.success(f"Low Risk ({(1-probability)*100:.2f}%)")
