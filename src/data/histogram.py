#!/usr/bin/python3
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from src.utils.utils import load
except ImportError:
    print("An error occurred importing load from utils.py")
    load = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is not installed.")
    plt = None


def main():
    """
    Load the Hogwarts student dataset and generate histogram
    visualizations for each numerical feature grouped by house.

    Steps performed:
    1. Load the training dataset from a CSV file
    2. Remove irrelevant columns
    3. Iterate over selected features
    4. Plot histograms per feature, separated by Hogwarts house

    Returns:
        None:
        Displays matplotlib histograms.
    """

    if load is None or plt is None:
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
        'Transfiguration', 'Potions', 'Care of Magical Creatures',
        'Charms', 'Flying'
    ]

    houses = dftr['Hogwarts House'].unique()

    for feature in features:
        plt.figure(figsize=(10, 10))

        for house in houses:
            subset = dftr[dftr['Hogwarts House'] == house]

            plt.hist(
                subset[feature].dropna(),
                bins=30,
                alpha=0.5,
                label=house
            )

        plt.title(feature)
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.legend()

        plt.show()


if __name__ == "__main__":
    main()
