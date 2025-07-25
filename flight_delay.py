# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 16:09:12 2025

@author: DELL
"""

import pandas as pd

import pandas as pd

# Load the CSV file
import pandas as pd

# Use raw string to avoid issues with backslashes
df = pd.read_csv(r"C:\Users\DELL\Desktop\I_flight_delays.csv")

# Now check the data
df.head()

# Show first few rows
print(df.head())
print(df.shape)       # Rows and columns
print(df.columns)     # Column names

print(df.isnull().sum())
df.info()
print(df.columns)
df.info()
print(df.isnull().sum())
for i in df.columns:
    if df[i].dtype== 'object':
        unique_val = df[i].unique()
        mapping={label: idx for idx,label in enumerate(unique_val)}
        df[i] = df[i].map(mapping)
print(df.head())

for idx, col in enumerate(df.columns):
    print(f"{idx} : {col}")
y = df.iloc[:, 6]          # Column at index 6 is the target/output
X = df.drop(df.columns[6], axis=1)  # Drop the same column from input
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

