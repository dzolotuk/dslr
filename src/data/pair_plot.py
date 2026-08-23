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

try:
    import seaborn as sns
except ImportError:
    print("Error: seaborn is not installed.")
    sns = None


def main():
    """
    Load the Hogwarts student dataset and generate a pairplot
    to visualize relationships between selected numerical features.

    This script:
    1. Loads the training dataset from a CSV file
    2. Renames selected columns for consistency
    3. Selects numerical features for analysis
    4. Generates a Seaborn pairplot colored by Hogwarts house

    Returns:
        None:
        Displays the pairplot.
    """

    if load is None or sns is None or plt is None:
        print("Missing required dependencies.")
        return

    dftr = load("dataset_train.csv")
    if dftr is None:
        print("Error loading dataset_train.csv")
        return

    dftr = dftr.rename(columns={
        'Hogwarts House': 'House',
        'Defense Against the Dark Arts': 'Defense',
        'Muggle Studies': 'Muggles',
        'Ancient Runes': 'Runes',
        'History of Magic': 'History',
        'Transfiguration': 'Transfig'
    })

    features = [
        'Arithmancy', 'Herbology', 'Defense', 'Muggles', 'Runes',
        'History', 'Transfig', 'Potions', 'Charms', 'Flying'
    ]

    sns.pairplot(dftr, vars=features, hue='House')
    plt.show()


if __name__ == "__main__":
    main()
