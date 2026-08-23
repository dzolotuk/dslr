import math

def sigmoid(z):
	"""
	Compute the sigmoid function.

	The sigmoid maps any real value into the range (0, 1),
	which allows interpreting the output as a probability.

	Args:
		z (float): Linear combination of inputs and weights.

	Returns:
		float: Value between 0 and 1.
	"""
	return 1 / (1 + math.exp(-z))



def predict_proba(x, theta):
	"""
	Compute the probability that input x belongs to class 1.

	This is done by applying the sigmoid function to the linear
	combination of input features and model weights.

	Args:
		x (list[float]): Feature vector (including bias term).
		theta (list[float]): Model weights.

	Returns:
		float: Probability estimate.
	"""
	z = 0
	for i in range(len(theta)):
		z += theta[i] * x[i]
	return sigmoid(z)


def compute_loss(X, y, theta):
	"""
	Compute the logistic regression loss (cross-entropy).

	The loss measures how well the model predictions match
	the true labels. Lower is better.

	Args:
		X (list[list[float]]): Dataset features.
		y (list[int]): Binary labels (0 or 1).
		theta (list[float]): Model weights.

	Returns:
		float: Average loss over the dataset.
	"""
	m = len(X)
	loss = 0

	for i in range(m):
		h = predict_proba(X[i], theta)

		# Avoid log(0)
		h = max(min(h, 1 - 1e-15), 1e-15)

		loss += y[i] * math.log(h) + (1 - y[i]) * math.log(1 - h)

	return -loss / m

def compute_gradient(X, y, theta):
	"""
	Compute the gradient of the loss function.

	The gradient indicates how each weight should be adjusted
	to reduce the loss.

	Args:
		X (list[list[float]]): Dataset features.
		y (list[int]): Binary labels.
		theta (list[float]): Model weights.

	Returns:
		list[float]: Gradient vector.
	"""
	m = len(X)
	n = len(theta)
	grad = [0] * n

	for i in range(m):
		h = predict_proba(X[i], theta)
		error = h - y[i]

		for j in range(n):
			grad[j] += error * X[i][j]

	for j in range(n):
		grad[j] /= m

	return grad

def gradient_descent(X, y, theta, lr, iterations):
	"""
	Train the model using gradient descent.

	At each iteration, weights are updated in the direction
	that reduces the loss.

	Args:
		X (list[list[float]]): Training features.
		y (list[int]): Binary labels.
		theta (list[float]): Initial weights.
		lr (float): Learning rate.
		iterations (int): Number of iterations.

	Returns:
		list[float]: Trained weights.
	"""
	for _ in range(iterations):
		grad = compute_gradient(X, y, theta)

		for j in range(len(theta)):
			theta[j] -= lr * grad[j]

	return theta

def train_one_vs_all(X, y, num_classes, lr, iterations):
	"""
	Train multiple binary classifiers using One-vs-All strategy.

	Each class is trained as a separate binary classification
	problem (class vs all others).

	Args:
		X (list[list[float]]): Training features.
		y (list[int]): Class labels.
		num_classes (int): Number of distinct classes.
		lr (float): Learning rate.
		iterations (int): Number of iterations.

	Returns:
		list[list[float]]: List of weight vectors (one per class).
	"""
	all_theta = []

	for c in range(num_classes):
		theta = [0] * len(X[0])

		y_binary = [1 if label == c else 0 for label in y]

		theta = gradient_descent(X, y_binary, theta, lr, iterations)
		all_theta.append(theta)

	return all_theta

def predict(X, all_theta):
	"""
	Predict class labels for given inputs.

	Each input is evaluated against all classifiers, and
	the class with the highest probability is selected.

	Args:
		X (list[list[float]]): Feature matrix.
		all_theta (list[list[float]]): Trained weights.

	Returns:
		list[int]: Predicted class indices.
	"""
	predictions = []

	for x in X:
		probas = []

		for theta in all_theta:
			probas.append(predict_proba(x, theta))

		predictions.append(probas.index(max(probas)))

	return predictions