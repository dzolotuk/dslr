import os
import random

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is not installed.")
    pd = None


def load(path: str) -> pd.DataFrame | None:  # type: ignore
    """
    Load a CSV dataset from the given path.

    The function checks that the path is valid,
    that the file exists and that it has a .csv extension.
    If an error occurs, the function returns None.

    Args:
        path (str): Path to the CSV file.

    Returns:
        The loaded dataset if successful, otherwise None.
    """
    if pd is None:
        print("Error: pandas is required.")
        return None

    if not isinstance(path, str):
        print("Error: path must be a string.")
        return None

    if not os.path.isfile(path):
        print(f"Error: file '{path}' does not exist.")
        return None

    if not path.lower().endswith(".csv"):
        print("Error: bad format, file is not a CSV.")
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        print("Error: unable to read the file as a CSV dataset.")
        return None

    return df


def our_len(values):
    """Return the length of the data"""
    our_len = 0
    for _ in values:
        our_len += 1
    return our_len


def mean(lst):
    """Calculate and return the average of a list of numbers"""
    total = 0
    for x in lst:
        total += x
    return total / our_len(lst)


def std(lst, m):
    """Calculate the variance and return the standard deviation of a list"""
    total = 0
    for x in lst:
        total += (x - m) ** 2
    return (total / our_len(lst)) ** 0.5


def ft_sort(vals):
    """Sort a list without using sorted()"""
    n = our_len(vals)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if vals[j] < vals[min_idx]:
                min_idx = j
        vals[i], vals[min_idx] = vals[min_idx], vals[i]
    return vals


def fillna_median(X):
	"""
	Replace missing values (NaN) in a 2D list
	with the median of their respective columns.

	This function
	1. extracts all non-NaN values from the column
	2. sorts the values and calculates the median
	3. replaces any NaN values in that column with the calculated median

	Parameters:
	X (list of list): A matrix where each inner list represents a row
	                  and missing values are represented as NaN (float('nan'))

	Returns:
	the modified 2D list with NaN values replaced by column medians

	Note:
	Uses the property that NaN != NaN to identify missing values
	"""
	for j in range(our_len(X[0])):
		col = [row[j] for row in X if row[j] == row[j]]
		sorted_col = ft_sort(col)
		n = our_len(sorted_col)

		if n == 0:
			med = 0
		else:
			sorted_col = ft_sort(col)
			if n % 2 == 1:
				med = sorted_col[n // 2]
			else:
				med = (sorted_col[n // 2 - 1] + sorted_col[n // 2]) / 2

		for i in range(our_len(X)):
			if X[i][j] != X[i][j]:
				X[i][j] = med
	return X


def standardize(X):
	"""
	Standardize the dataset using z-score normalization.

	This function computes the mean and standard deviation
	for each feature and scales the dataset accordingly.

	Args:
		X (list[list[float]]): Input feature matrix.

	Returns:
		tuple:
			- X_scaled (list[list[float]]): Standardized data
			- means (list[float]): Mean of each feature
			- stds (list[float]): Standard deviation of each feature
	"""
	n_features = our_len(X[0])
	m = our_len(X)

	means = [0] * n_features
	stds = [0] * n_features

	for j in range(n_features):
		col = [X[i][j] for i in range(m)]
		m_val = mean(col)
		s_val = std(col, m_val)

		means[j] = m_val
		stds[j] = s_val if s_val != 0 else 1


	X_scaled = []
	for i in range(m):
		row = []
		for j in range(n_features):
			val = (X[i][j] - means[j]) / stds[j]
			row.append(val)
		X_scaled.append(row)

	return X_scaled, means, stds


def train_test_split(X, y, test_size=0.2):
    """
    Splits features and labels into training and validation sets.

    Args:
        X (list): Input features.
        y (list): Target labels.
        test_size (float): Proportion of the dataset to include
                           in the validation split.

    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    idx = list(range(our_len(X)))
    random.shuffle(idx)

    split = int(our_len(X) * (1 - test_size))

    train_idx = idx[:split]
    val_idx = idx[split:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]

    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]

    return X_train, X_val, y_train, y_val

def apply_standardize(X, means, stds):
	"""
	Apply standardization using precomputed mean and std.

	This function is used during prediction to ensure that
	the test data is transformed in the same way as training data.

	Args:
		X (list[list[float]]): Input feature matrix.
		means (list[float]): Mean values from training data.
		stds (list[float]): Standard deviations from training data.

	Returns:
		list[list[float]]: Standardized data.
	"""
	if len(means) != len(stds):
		raise ValueError("means and stds must have the same length.")

	if len(X) == 0:
		return X

	if len(X[0]) != len(means):
		raise ValueError(
			f"Feature mismatch: X has {len(X[0])} features, "
			f"but means/stds have {len(means)}."
		)
	
	means_len = our_len(means)
	x_len = our_len(X)

	X_scaled = []

	for i in range(x_len):
		row = []
		for j in range(means_len):
			if stds[j] != 0:
				val = (X[i][j] - means[j]) / stds[j]
			else:
				val = 0
			row.append(val)
		X_scaled.append(row)

	return X_scaled

def extract_features_labels(df):
	"""
	Prepare training data from the raw dataset.

	Steps:
	1. Remove irrelevant columns
	2. Rename columns for consistency
	3. Normalize column names (lowercase)
	4. Select relevant features in fixed order
	5. Split into X (features) and y (labels)

	Args:
		df (pandas.DataFrame): Raw training dataset

	Returns:
		tuple:
			- X (list[list[float]]): Feature matrix
			- y (list[int]): Class labels (encoded as integers)
	"""
	houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

	df = df.drop(columns=[
		'Index', 'First Name', 'Last Name', 'Birthday',
		'Best Hand', 'Astronomy', 'Ancient Runes',
		'Care of Magical Creatures', 'Charms', 'Flying'
	])

	df = df.rename(columns={
		'Hogwarts House': 'house',
		'Defense Against the Dark Arts': 'defense_ada',
		'Muggle Studies': 'muggles',
		'History of Magic': 'history'
	})

	df.columns = [col.lower() for col in df.columns]

	df = df[
		['house', 'arithmancy', 'herbology', 'defense_ada', 'divination',
		 'muggles', 'history', 'transfiguration', 'potions']
	]

	y = df['house'].tolist()
	X = df.drop(columns=['house']).values.tolist()

	y = [houses.index(h) for h in y]

	return X, y

def extract_features(df):
	"""
	Prepare feature matrix from the test dataset.

	Steps:
	1. Remove irrelevant columns
	2. Rename columns for consistency
	3. Normalize column names (lowercase)
	4. Select relevant features in fixed order

	Args:
		df (pandas.DataFrame): Raw test dataset

	Returns:
		list[list[float]]: Feature matrix
	"""

	df = df.drop(columns=[
		'Index', 'First Name', 'Last Name', 'Birthday',
		'Best Hand', 'Astronomy', 'Ancient Runes',
		'Care of Magical Creatures', 'Charms', 'Flying'
	])

	df = df.rename(columns={
		'Defense Against the Dark Arts': 'defense_ada',
		'Muggle Studies': 'muggles',
		'History of Magic': 'history'
	})

	df.columns = [col.lower() for col in df.columns]

	df = df[
		['arithmancy', 'herbology', 'defense_ada', 'divination',
		 'muggles', 'history', 'transfiguration', 'potions']
	]

	return df.values.tolist()