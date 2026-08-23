import pandas as pd
from sklearn.metrics import accuracy_score

# 1. Truth data load
df = pd.read_csv("dataset_truth.csv")

# 2. Prediction data load
preds = pd.read_csv("houses.csv")

# 3. Extraxt ground truth
y_true = df["Hogwarts House"]

# 4. Extract predictions
y_pred = preds["Hogwarts House"]

# 5. Accuracy
acc = accuracy_score(y_true, y_pred)

print("Accuracy:", acc)