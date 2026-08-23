#!/usr/bin/python3
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from src.utils.utils import load, our_len
except ImportError:
    print("An error occurred importing functions from utils.py")
    load = None
    our_len = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is not installed.")
    plt = None

try:
    import seaborn as sns
except ImportError:
    print("Error: seaborn is not installed.")
    sns = None


def main():
    """
    Load the Hogwarts student dataset and generate a scatterplot
    based on the two most correlated numerical features.

    Steps performed:
    1. Load the training dataset
    2. Select numerical features
    3. Compute correlation matrix
    4. Find the most correlated feature pair (manual iteration)
    5. Plot a scatterplot colored by Hogwarts house

    Returns:
        None:
        Displays a scatterplot and prints correlation results.
    """

    if load is None or our_len is None or plt is None:
        print("Missing required dependencies.")
        return

    dftr = load("dataset_train.csv")
    if dftr is None:
        print("Error loading dataset_train.csv")
        return

    features = [
        'Arithmancy', 'Astronomy', 'Herbology',
        'Defense Against the Dark Arts', 'Divination',
        'Muggle Studies', 'Ancient Runes', 'History of Magic',
        'Transfiguration', 'Potions', 'Charms', 'Flying'
    ]

    corr = dftr[features].corr()

    best_pair = None
    best_abs_corr = 0
    best_corr_value = 0

    for i in range(our_len(features)):
        for j in range(i + 1, our_len(features)):
            f1 = features[i]
            f2 = features[j]

            value = corr.loc[f1, f2]

            if abs(value) > best_abs_corr:
                best_abs_corr = abs(value)
                best_corr_value = value
                best_pair = (f1, f2)

    similar_features = f"{best_pair[0]} and {best_pair[1]}"
    correlation = f"(corr = {best_corr_value:.0f})"

    print(f"The most similar features are: {similar_features} {correlation}")

    plt.figure()

    for house in dftr['Hogwarts House'].unique():
        subset = dftr[dftr['Hogwarts House'] == house]
        plt.scatter(
            subset[best_pair[0]],
            subset[best_pair[1]],
            alpha=0.075,
            label=house
        )

    plt.xlabel(best_pair[0])
    plt.ylabel(best_pair[1])
    plt.title("Scatterplot of the similar features")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
