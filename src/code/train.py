from sklearn.svm import SVC
from joblib import dump
from preprocess import preprocess_data

X_train, X_test, y_train, y_test, scaler = preprocess_data(
    "../data/heart.csv"
)

model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True
)

model.fit(X_train, y_train)

dump(model, "../models/heart_model.pkl")
dump(scaler, "../models/scaler.pkl")

print("Model saved successfully")
