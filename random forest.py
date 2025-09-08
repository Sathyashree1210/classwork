from sklearn.ensemble import RandomForestClassifier

# Sample data: [Glucose, BloodPressure, BMI, Age]
X = [
    [130, 70, 32.0, 45],
    [85, 66, 26.6, 31],
    [90, 80, 30.5, 25],
    [120, 60, 35.0, 40],
    [150, 72, 33.6, 50],
    [88, 64, 28.1, 22],
    [140, 75, 34.2, 55],
    [100, 68, 29.0, 30]
]

# Labels: 1 = Diabetic, 0 = Not Diabetic
y = [1, 0, 0, 1, 1, 0, 1, 0]

# Create and train model
model = RandomForestClassifier()
model.fit(X, y)

# New patient data to predict
new_patient = [[110, 65, 31.5, 35]]
prediction = model.predict(new_patient)

print("Diabetes Prediction (1=Yes, 0=No):", prediction[0])
