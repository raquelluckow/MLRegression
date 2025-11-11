import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

dataset = load_diabetes()

X = dataset.data
y = dataset.target

# Split the data into train and test (67/33)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=14)

# Linear Regression model
LR_model = LinearRegression()

# Train the model
LR_model.fit(X_train, y_train)

# Predict the target values
y_pred = LR_model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2:.2f}")