#!/usr/bin/python3

import csv
import sys

def read_categorical_csv(filename):
    """
    Read a CSV file and extract non-numeric columns.

    Each column is scanned and values that cannot be converted to float
    are considered categorical. Empty strings and 'nan' values are ignored.

    Args:
        filename (str): Path to the CSV file.

    Returns:
        dict: Dictionary where keys are column names and values are lists
        of categorical (non-numeric) values.
    """
    categorical_data = {}

    try:
        with open(filename, newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return {}

            for col in reader.fieldnames:
                categorical_data[col] = []

            for row in reader:
                for col, val in row.items():
                    val = val.strip()
                    if val == "" or val.lower() == "nan":
                        continue
                    try:
                        float(val)
                    except ValueError:
                        categorical_data[col].append(val)

        categorical_data = {
            col: vals for col, vals in categorical_data.items() if vals
        }

    except Exception as e:
        print(f"Unexpected ERROR: {e}")
        return {}

    return categorical_data


def count(vals):
    """Return number of elements in a list."""
    n = 0
    for _ in vals:
        n += 1
    return n


def unique_count(vals):
    """Return number of unique values without using set()."""
    uniques = []
    for val in vals:
        found = False
        for u in uniques:
            if u == val:
                found = True
                break
        if not found:
            uniques.append(val)
    return count(uniques)


def mode_and_freq(vals):
    """Return most frequent value (mode) and its frequency."""
    max_count = 0
    mode_val = None

    for i in range(count(vals)):
        current = vals[i]
        current_count = 0

        for j in range(count(vals)):
            if vals[j] == current:
                current_count += 1

        if current_count > max_count:
            max_count = current_count
            mode_val = current

    return mode_val, max_count


def main():
    """
    Compute and display descriptive statistics for non-numeric columns
    in a CSV dataset.

    For each non-numeric column, the following metrics are computed:
        - Count  : number of valid entries
        - Unique : number of distinct values
        - Top    : most frequent value
        - Freq   : frequency of the most common value

    The results are displayed in a formatted table.

    Usage:
        python describe_bonus.py dataset.csv

    Args:
        None (uses sys.argv[1] as input file)

    Exits:
        1 if incorrect usage
        0 if successful or no categorical data found
    """

    if len(sys.argv) != 2:
        print("Usage: python describe_bonus.py dataset.csv")
        sys.exit(1)

    filename = sys.argv[1]
    cat_data = read_categorical_csv(filename)

    if not cat_data:
        print("No non-numeric data found.")
        sys.exit(0)

    cat_stats = {}

    for col, vals in cat_data.items():
        top, freq = mode_and_freq(vals)
        cat_stats[col] = {
            'Count': count(vals),
            'Unique': unique_count(vals),
            'Top': top,
            'Freq': freq
        }

    col_width = 15
    cols = list(cat_stats.keys())

    for i in range(0, len(cols), 5):
        packet = cols[i:i+5]

        header = " ".join(f"{col:>{col_width}}" for col in packet)
        print("\n" + " " * 11 + header)

        for stat in ['Count', 'Unique', 'Top', 'Freq']:
            values = " ".join(
                f"{str(cat_stats[col][stat]):>{col_width}}"
                for col in packet
            )
            print(f"{stat:<11}{values}")


if __name__ == "__main__":
    main()
