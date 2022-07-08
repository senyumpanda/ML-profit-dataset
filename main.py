# Import Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Dataset
df = pd.read_csv('Dataset 3/50_Startups.csv')

# Preprocessing Data - Menyaring nilai tidak 0
for i in df.columns:
    df = df[df[i] != 0]

# Visualiasi Beberapa Kolom - Mencari Pola Linear Regression
plt.scatter(df["R&D Spend"], df["Profit"], color="blue")
plt.show()

# Data pada dataset dijadikan array
X = np.array(df["R&D Spend"]).reshape(-1,1)
y = np.array(df["Profit"]).reshape(-1,1)

# Training dan Testing Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

# Model Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Visualisasi Linear Regression
plt.scatter(df["R&D Spend"], df["Profit"], color="blue")
plt.plot(X_test, model.predict(X_test), color="red",linewidth=3)
plt.show()