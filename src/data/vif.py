#!/usr/bin/python3

import os
try:
    import pandas as pd
except ImportError:
    print("Error: pandas is not installed.")
    pd = None

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError:
    print("Error: statsmodels is not installed.")
    variance_inflation_factor = None

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("Error: scikit_learn is not installed.")
    StandardScaler = None

try:
    import scipy.stats as stats
except ImportError:
    print("Error: scipy.stats is not installed.")
    stats = None


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


def compute_vif(X):
    """
    Compute the Variance Inflation Factor (VIF) for each feature in X.

    Returns a DataFrame with feature names and their corresponding VIF values.
    """
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
    return vif_data


def select(X, threshold=10):
    """
    Iteratively remove features with VIF above the given threshold.

    Returns the reduced dataset and the final VIF values.
    """
    X = X.copy()

    while True:
        vif = compute_vif(X)
        max_vif = vif["VIF"].max()

        if max_vif < threshold:
            break

        feature_to_drop = vif.loc[vif["VIF"].idxmax(), "feature"]
        print(f"Dropping '{feature_to_drop}' with VIF = {max_vif:.2f}")
        X = X.drop(columns=[feature_to_drop])

    return X, vif


def main():
    """
    Main function to load, preprocess, and analyze the dataset.

    Steps:
    1. Load and clean the dataset.
    2. Analyze the distribution of 'Best Hand'
       and its relationship with 'Hogwarts House'.
    3. Preprocess the data for feature selection.
    4. Compute and reduce VIF to select the most relevant features.
    """
    dftr = load("dataset_train.csv")
    if dftr is None:
        print("Error loading dataset_train.csv")
        return

    dftr = dftr.drop(columns=['Index', 'First Name', 'Last Name', 'Birthday'])

    # Focus on "Best Hand"
    #####################

    print(dftr['Best Hand'].value_counts(), "\n")
    cross_tab = pd.crosstab(dftr['Hogwarts House'], dftr['Best Hand'])

    chi2, p_value, _, expected = stats.chi2_contingency(cross_tab)
    print("Chi2:", chi2)
    print("p-value:", p_value)

    # Data Preprocessing
    ####################

    dftr = dftr.rename(columns={
        'Hogwarts House': 'house',
        'Defense Against the Dark Arts': 'defense_ada',
        'Muggle Studies': 'muggles',
        'Ancient Runes': 'runes',
        'History of Magic': 'history',
        'Care of Magical Creatures': 'care_mc'
    })

    dftr.columns = [col.lower() for col in dftr.columns]

    dftr = dftr[['house', 'arithmancy', 'herbology', 'defense_ada',
                 'divination', 'muggles', 'runes', 'history',
                 'transfiguration', 'potions', 'charms', 'flying']]

    X = dftr.drop(columns=['house'])

    X = X.fillna(X.median(numeric_only=True))

    # Variance Inflation Factor (VIF)
    #################################

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    print("\nInitial VIF:")
    print(compute_vif(X_scaled))

    X_selected_scaled, final_vif = select(X_scaled)

    print("\nFinal features:")
    print(X_selected_scaled.columns.tolist())

    print("\nFinal VIF:")
    print(final_vif)


if __name__ == "__main__":
    main()
