import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
from src.utils.utils import load, fillna_median, apply_standardize, extract_features
from model import predict


def add_bias(X):
	"""
	Add a bias term to each feature vector.

	The bias term is a constant value inserted at the beginning of each row.
	It allows the model to learn an intercept.

	Args:
		X (list[list[float]]): Feature matrix.

	Returns:
		list[list[float]]: Feature matrix with a leading bias term.
	"""
	return [[1] + row for row in X]


def main():
	"""
	Main entry point for model prediction.

	The program:
	1. Loads the test dataset
	2. Extracts the feature matrix
	3. Replaces missing values with column medians
	4. Applies the training standardization parameters
	5. Adds a bias term
	6. Loads the trained weights
	7. Predicts the Hogwarts house for each row
	8. Writes the results to houses.csv

	Usage:
		python logreg_predict.py dataset_test.csv weights.json
	"""
	if len(sys.argv) != 3:
		print("Usage: python logreg_predict.py dataset_test.csv weights.json")
		return

	data_file = sys.argv[1]
	weights_file = sys.argv[2]

	data = load(data_file)
	if data is None:
		return

	X = extract_features(data)
	X = fillna_median(X)

	with open(weights_file, "r") as f:
		model_data = json.load(f)

	all_theta = model_data["theta"]
	means = model_data["means"]
	stds = model_data["stds"]

	X = apply_standardize(X, means, stds)
	X = add_bias(X)

	preds = predict(X, all_theta)

	houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

	with open("houses.csv", "w") as f:
		f.write("Index,Hogwarts House\n")
		for i, p in enumerate(preds):
			f.write(f"{i},{houses[p]}\n")

	print("Prediction complete!")


if __name__ == "__main__":
	main()