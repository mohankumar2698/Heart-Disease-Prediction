import streamlit as st
import numpy as np
from joblib import load
from pathlib import Path

# 1. Get the directory containing app.py (src/dashboard/)
current_dir = Path(__file__).parent

# 2. Point to where your models folder actually is. 
# Adjust the "../" depending on where your models folder sits relative to app.py
model_path = current_dir / "../../models/heart_model.pkl" 
scaler_path = current_dir / "../../models/scaler.pkl"

# 3. Load using the clean absolute path resolution
model = load(model_path.resolve())
scaler = load(scaler_path.resolve())

st.title("❤️ Heart Disease Prediction Dashboard")

# Fixed: Added quotes to labels
age = st.number_input("Age", 1, 120, 45)
sex = st.selectbox("Sex", [0, 1])
cp = st.number_input("Chest Pain Type", 0, 3)

# Fixed: Added quotes to labels and minimum value
trestbps = st.number_input("Resting Blood Pressure", 50, 300, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)

fbs = st.selectbox("Fasting Blood Sugar", [0, 1])

# Fixed: Added quotes to labels
restecg = st.number_input("Rest ECG", 0, 2)
thalach = st.number_input("Max Heart Rate", 60, 220, 150)

exang = st.selectbox("Exercise Angina", [0, 1])

if st.button("Predict"):
    # Fixed: Wrapped inputs in float() to prevent scikit-learn type errors
    data = np.array([[float(age), float(sex), float(cp), float(trestbps), 
                      float(chol), float(fbs), float(restecg), float(thalach), float(exang)]])
    
    data = scaler.transform(data)
    prediction = model.predict(data)
    probability = model.predict_proba(data)[0][1]
    
    if prediction[0] == 1:
        # Fixed: F-string syntax requires closing quotes correctly inside the block
        st.error(f"High Risk ({probability * 100:.2f}%)")
    else:
        st.success(f"Low Risk ({(1 - probability) * 100:.2f}%)")
