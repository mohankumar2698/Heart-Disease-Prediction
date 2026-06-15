import streamlit as st
import numpy as np
import streamlit as st
from joblib import model = joblib.load('heart_model.pkl')
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)

model = load("models/heart_model.pkl")
scaler = load("models/scaler.pkl")

st.title("❤️ Heart Disease Prediction Dashboard")

age = st.number_input("Age", 1, 120, 45)
sex = st.selectbox("Sex", [0,1])
cp = st.number_input("Chest Pain Type", 0, 3)
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar", [0,1])
restecg = st.number_input("Rest ECG", 0, 2)
thalach = st.number_input("Max Heart Rate")
exang = st.selectbox("Exercise Angina", [0,1])

if st.button("Predict"):

    data = np.array([
        [
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang
        ]
    ])

    data = scaler.transform(data)

    prediction = model.predict(data)

    probability = model.predict_proba(data)[0][1]

    if prediction[0] == 1:
        st.error(
            f"High Risk ({probability*100:.2f}%)"
        )
    else:
        st.success(
            f"Low Risk ({(1-probability)*100:.2f}%)"
        )
