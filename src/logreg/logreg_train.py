import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
from src.utils.utils import load, standardize, fillna_median, extract_features_labels, train_test_split, apply_standardize
from sklearn.metrics import accuracy_score
from model import train_one_vs_all, predict

def add_bias(X):
	"""
	Add a bias term to each feature vector.

	The bias term is a constant value inserted at the beginning of each row.
	It allows the model to learn an intercept term.

	Args:
		X (list[list[float]]): Feature matrix.

	Returns:
		list[list[float]]: Feature matrix with a leading bias term.
	"""
	return [[1] + row for row in X]

def main():
	"""
	Train and evaluate a logistic regression model.

	The program:
	1. Loads the dataset and extracts features/labels
	2. Splits data into training and validation sets (80/20)
	3. Preprocesses training data (fill missing values, standardize, add bias)
	4. Applies the same preprocessing to validation data
	5. Trains a one-vs-all logistic regression model
	6. Evaluates the model using accuracy on the validation set
	7. Retrains the model on the full dataset
	8. Saves the model weights and normalization parameters

	Usage:
		python logreg_train.py dataset_train.csv

	Output:
		Saves 'weights.json' containing:
			- theta (model weights)
			- means and stds (for standardization)
	"""
	if len(sys.argv) != 2:
		print("Usage: python logreg_train.py dataset_train.csv")
		return

	data_file = sys.argv[1]

	data = load(data_file)
	if data is None:
		return

	# 1. Extract
	X, y = extract_features_labels(data)

	# 2. Split
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

	# 3. Train Preprocessing
	X_train = fillna_median(X_train)
	X_train, means, stds = standardize(X_train)
	X_train = add_bias(X_train)

	# 4. Validation preprocessing
	X_val = fillna_median(X_val)
	X_val = apply_standardize(X_val, means, stds)
	X_val = add_bias(X_val)

	# 5. Train
	all_theta = train_one_vs_all(
		X_train,
		y_train,
		num_classes=4,
		lr=0.01,
		iterations=2000
	)

	# 6. Evaluate
	y_pred = predict(X_val, all_theta)
	acc = accuracy_score(y_val, y_pred)
	print(f"Validation accuracy: {acc:.4f}")

	# 7. Final train on whole dataset
	X_full = fillna_median(X)
	X_full, means, stds = standardize(X_full)
	X_full = add_bias(X_full)

	all_theta = train_one_vs_all(
		X_full,
		y,
		num_classes=4,
		lr=0.01,
		iterations=2000
	)

	# 8. Save
	data_to_save = {
		"theta": all_theta,
		"means": means,
		"stds": stds
	}

	with open("weights.json", "w") as f:
		json.dump(data_to_save, f)

	print("Training complete! Weights saved to weights.json")


if __name__ == "__main__":
	main()