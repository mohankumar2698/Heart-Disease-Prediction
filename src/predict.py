from joblib import load
import numpy as np

model = load("../models/heart_model.pkl")
scaler = load("../models/scaler.pkl")

sample = np.array([
    [63,1,3,145,233,1,0,150,0]
])

sample = scaler.transform(sample)

prediction = model.predict(sample)

if prediction[0] == 1:
    print("Heart Disease Risk Detected")
else:
    print("No Heart Disease Risk")
