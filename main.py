import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("iris_data.csv")

# Input features
X = df[
    [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
]

# Target
y = df["Species"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = DecisionTreeClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nAccuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Iris Classification - Confusion Matrix")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")

plt.xticks(
    range(3),
    ["Setosa", "Versicolor", "Virginica"]
)

plt.yticks(
    range(3),
    ["Setosa", "Versicolor", "Virginica"]
)

# Show numbers inside the matrix
for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.show()
import joblib
joblib.dump(model, "iris_model.pkl")
print("\nModel saved successfully as iris_model.pkl")
# User input prediction
print("\n--- Iris Flower Prediction ---")

sepal_length = float(input("Enter Sepal Length (cm): "))
sepal_width = float(input("Enter Sepal Width (cm): "))
petal_length = float(input("Enter Petal Length (cm): "))
petal_width = float(input("Enter Petal Width (cm): "))

new_flower = pd.DataFrame([{
    "SepalLengthCm": sepal_length,
    "SepalWidthCm": sepal_width,
    "PetalLengthCm": petal_length,
    "PetalWidthCm": petal_width
}])

prediction = model.predict(new_flower)

print("\nPredicted Species:", prediction[0])
# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import joblib
joblib.dump(model, "iris_model.pkl")
print("\nModel saved successfully as iris_model.pkl")