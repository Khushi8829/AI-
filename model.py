import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("dataset.csv")
X = data[["size"]]
y = data["price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict and accuracy
predictions = model.predict(X)
accuracy = r2_score(y, predictions)

print("Model Accuracy:", accuracy)
