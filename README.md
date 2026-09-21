# Iris Flower Classification 🌸

## Project Overview

This project is developed as part of the CodeAlpha Data Science Internship.

The objective of this project is to build a Machine Learning classification model that can identify the species of an Iris flower based on its measurements.

The model classifies flowers into three species:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

## Dataset

The project uses the Iris dataset containing measurements of Iris flowers.

The input features are:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The dataset contains 150 flower records.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- VS Code

## Machine Learning Algorithm

A Decision Tree Classifier is used for classification.

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

Training samples: 120

Testing samples: 30

## Model Performance

The trained model achieved:

**Accuracy: 100%**

The confusion matrix shows that all 30 test samples were classified correctly.

## Project Features

- Load and analyze the Iris dataset
- Split data into training and testing sets
- Train a Decision Tree classification model
- Evaluate model accuracy
- Generate classification report
- Generate confusion matrix
- Predict the species of a new Iris flower
- Save the trained model as `iris_model.pkl`

## Example Prediction

Input:

- Sepal Length: 5.1 cm
- Sepal Width: 3.5 cm
- Petal Length: 1.4 cm
- Petal Width: 0.2 cm

Prediction:

**Iris-setosa**

## Project Structure

```text
CodeAlpha_Iris_Classification/
│
├── main.py
├── iris_data.csv
├── iris_model.pkl
├── README.md
└── confusion_matrix.png