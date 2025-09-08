# logistic_regression_real_world.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_data():
    # Load dataset from a CSV file
    try:
        data = pd.read_csv("Social_Network_Ads.csv")
    except FileNotFoundError:
        print("ERROR: Dataset not found. Please make sure 'Social_Network_Ads.csv' is in the same folder.")
        exit()

    return data

def preprocess_data(data):
    # Select features and target
    X = data[["Age", "EstimatedSalary"]]
    y = data["Purchased"]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # Scale features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def train_model(X_train, y_train):
    # Initialize and train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("=== Accuracy Score ===")
    print(f"{accuracy_score(y_test, y_pred)*100:.2f}%")

def visualize(data):
    # Plot data distribution
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x="Age", y="EstimatedSalary", hue="Purchased", palette="Set1")
    plt.title("Age vs Salary with Purchase Outcome")
    plt.xlabel("Age")
    plt.ylabel("Estimated Salary")
    plt.grid(True)
    plt.show()

def main():
    print("==== Logistic Regression: Predicting Purchase Behavior ====\n")
    data = load_data()
    print("First 5 rows of the dataset:\n")
    print(data.head())
    visualize(data)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
