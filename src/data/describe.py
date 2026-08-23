#!/usr/bin/python3

import csv
import sys


def read_numeric_csv(filename):
    """
    Read a CSV and return a dictionary for the columns of numeric values.
    Ignore non numeric columns and non castable to float values.
    """
    numeric_data = {}

    try:
        with open(filename, newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                print("ERROR: CSV without headers")
                return {}

            for col in reader.fieldnames:
                numeric_data[col] = []

            for row in reader:
                for col, val in row.items():
                    val = val.strip()
                    if val == "" or val.lower() == "nan":
                        continue
                    try:
                        num = float(val)
                        numeric_data[col].append(num)
                    except ValueError:
                        continue

        numeric_data = {
            col: vals for col, vals in numeric_data.items() if vals
            }

    except FileNotFoundError:
        print(f"ERROR: '{filename}' file doesn't exist")
        return {}
    except Exception as e:
        print(f"Unexpected ERROR: {e}")
        return {}

    return numeric_data


def count(vals):
    """Return the length of the data"""
    count = 0
    for _ in vals:
        count += 1
    return count


def ft_min(vals):
    """Return the minimum of a list"""
    if count(vals) == 0:
        return None
    min_val = vals[0]
    for val in vals:
        if val < min_val:
            min_val = val
    return min_val


def ft_max(vals):
    """Return the maximum of a list"""
    if count(vals) == 0:
        return None
    max_val = vals[0]
    for val in vals:
        if val > max_val:
            max_val = val
    return max_val


def ft_sort(vals):
    """Sort a list without using sorted()"""
    n = count(vals)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if vals[j] < vals[min_idx]:
                min_idx = j
        vals[i], vals[min_idx] = vals[min_idx], vals[i]
    return vals


def mean(vals):
    """
    Avoid the division by 0 and return the sample arithmetic mean of data
    """
    if not vals:
        return 0
    total = 0
    counter = 0
    for val in vals:
        total += val
        counter += 1
    mean = total / counter
    return mean


def median(vals):
    """Return the sample median of data"""
    vals = ft_sort(vals)
    n = count(vals)
    if n % 2 != 0:
        q2_index = int(0.5 * (n - 1))
        median = vals[q2_index]
    else:
        q2_index1 = int(0.5 * (n - 1))
        median1 = vals[q2_index1]
        q2_index2 = int(0.5 * n)
        median2 = vals[q2_index2]
        median = (median1 + median2) / 2
    return median


def quartile(vals):
    """
    Divide data into 4 continuous equal intervals
    and return the first and the third
    """
    vals = ft_sort(vals)
    n = count(vals)
    q1_index = int(0.25 * (n - 1))
    q3_index = int(0.75 * (n - 1))
    q1 = float(vals[q1_index])
    q3 = float(vals[q3_index])
    return [q1, q3]


def variance(vals):
    """Return the sample variance of data."""
    n = count(vals)
    if n < 2:
        return 0

    m = mean(vals)
    sum_squared_dev = 0

    for val in vals:
        sum_squared_dev += (val - m) ** 2

    return sum_squared_dev / (n - 1)


def std(vals):
    """Return the sample standard deviation (square root of variance)."""
    return variance(vals) ** 0.5


def main():
    """
    Compute and print descriptive statistics
    for each numeric column in a CSV file.

    Reads a CSV file, calculates key statistics
    (count, mean, std, min, quartiles, max) for each
    numeric column, and displays results in a formatted table.

    Usage:
        python describe.py dataset.csv

    Args:
        None (uses sys.argv[1] as the input CSV filename)

    Exits:
        1 if incorrect number of arguments or file not found
        0 if no numeric data is found or on successful completion
    """
    if len(sys.argv) != 2:
        print("Usage: python describe.py dataset.csv")
        sys.exit(1)

    filename = sys.argv[1]
    data = read_numeric_csv(filename)

    if not data:
        print("No numeric data found.")
        sys.exit(0)

    stats = {}

    for col, vals in data.items():
        stats[col] = {
            'Count': count(vals),
            'Mean': mean(vals),
            'Std': std(vals),
            'Min': ft_min(vals),
            '25%': quartile(vals)[0],
            '50%': median(vals),
            '75%': quartile(vals)[1],
            'Max': ft_max(vals)
        }

    col_width = 29
    cols = list(stats.keys())

    for i in range(0, len(cols), 3):
        packet = cols[i:i+3]

        header = " ".join([f"{col:>{col_width}}" for col in packet])
        print("\n" + " " * 11 + header)

        for stat in [
            'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'
                    ]:
            values = [f"{stats[col][stat]:>{col_width}.3f}" for col in packet]
            print(f"{stat:<11}" + " ".join(values))


if __name__ == "__main__":
    main()
