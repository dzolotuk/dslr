# DSLR — Datascience × Logistic Regression 🎩

The Sorting Hat has broken down, and Hogwarts needs a replacement. This project rebuilds it with data science: it explores a dataset of student grades, then trains a **multiclass logistic regression (one-vs-all) written from scratch** to predict each student's house: **Gryffindor, Hufflepuff, Ravenclaw or Slytherin**.

> 42 Paris project — Data Science / AI branch.

---

## Table of contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Part 1 — Data analysis](#part-1--data-analysis)
- [Part 2 — Data visualisation](#part-2--data-visualisation)
- [Part 3 — Logistic regression](#part-3--logistic-regression)
- [Bonus](#bonus)
- [How the model works](#how-the-model-works)
- [Project structure](#project-structure)
- [Tech stack](#tech-stack)

---

## Dataset

| File | Rows | Content |
|---|---|---|
| `dataset_train.csv` | 1 600 | Students with their house |
| `dataset_test.csv` | 400 | Students without their house (column left empty) |
| `dataset_truth.csv` | 400 | True houses of the test set, used to measure accuracy |

Each student has:

- **Identity:** `Index`, `First Name`, `Last Name`, `Birthday`, `Best Hand`
- **13 course grades:** Arithmancy, Astronomy, Herbology, Defense Against the Dark Arts, Divination, Muggle Studies, Ancient Runes, History of Magic, Transfiguration, Potions, Care of Magical Creatures, Charms, Flying

Some grades are missing.

---

## Installation

```bash
git clone https://github.com/dzolotuk/dslr.git
cd dslr
./setup.sh
source local_env/bin/activate
```

`setup.sh` creates a virtual environment `local_env` and installs `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`, `scipy` and `PyQt5` (graphical backend for matplotlib).

Run all commands **from the repository root**: the scripts read and write files in the current directory.

---

## Part 1 — Data analysis

```bash
python3 src/data/describe.py dataset_train.csv
```

A re-implementation of `pandas.DataFrame.describe()` **without using any statistics library**. For each numeric column, it computes:

`Count` · `Mean` · `Std` · `Min` · `25%` · `50%` · `75%` · `Max`

Count, mean, sample standard deviation (n − 1), min, max, sorting, median and quartiles are all written by hand. Empty and `NaN` values are ignored.

---

## Part 2 — Data visualisation

All three scripts read `dataset_train.csv` and open matplotlib windows.

### Histogram

```bash
python3 src/data/histogram.py
```

Shows, for each course, the distribution of grades per house (one window per course). It answers the question: **which course has a homogeneous grade distribution across all four houses?**

### Scatter plot

```bash
python3 src/data/scatter_plot.py
```

Finds the **two most similar features** by going through the correlation matrix and keeping the pair with the highest absolute correlation. It prints that pair and displays their scatter plot coloured by house.

### Pair plot

```bash
python3 src/data/pair_plot.py
```

A seaborn scatter plot matrix of 10 courses, coloured by house. It is used to choose which features to keep for logistic regression.

---

## Part 3 — Logistic regression

### Train

```bash
python3 src/logreg/logreg_train.py dataset_train.csv
```

1. Splits the data 80 / 20 (train / validation) and prints the **validation accuracy**.
2. Retrains the model on the **whole** dataset.
3. Saves the weights and normalisation parameters to `weights.json`.

### Predict

```bash
python3 src/logreg/logreg_predict.py dataset_test.csv weights.json
```

Writes the predictions to `houses.csv`:

```
Index,Hogwarts House
0,Hufflepuff
1,Ravenclaw
...
```

### Evaluate

```bash
python3 src/logreg/accuracy_tester.py
```

Compares `houses.csv` with `dataset_truth.csv` and prints the accuracy (`sklearn.metrics.accuracy_score`).

---

## Bonus

### Describe for non-numeric columns

```bash
python3 src/data/describe_bonus.py dataset_train.csv
```

For each non-numeric column (house, names, birthday, best hand), it computes `Count`, `Unique`, `Top` (most frequent value) and `Freq`, again without any library.

### Feature analysis: χ² and VIF

```bash
python3 src/data/vif.py
```

- **`Best Hand`:** a χ² test of independence between `Best Hand` and `Hogwarts House` checks whether the dominant hand says anything about the house.
- **Multicollinearity:** computes the **Variance Inflation Factor** of each course and iteratively removes the feature with the highest VIF until all of them are below 10.

---

## How the model works

### Features

The model uses **8 courses**:

`Arithmancy` · `Herbology` · `Defense Against the Dark Arts` · `Divination` · `Muggle Studies` · `History of Magic` · `Transfiguration` · `Potions`

Removed after the visualisation and VIF analysis: Astronomy, Ancient Runes, Care of Magical Creatures, Charms, Flying and Best Hand.

### Preprocessing

1. **Missing values** are replaced by the median of their column.
2. **Standardisation** (z-score): `(x − mean) / std`. The means and standard deviations come from training and are saved in `weights.json`, so the test data is transformed in exactly the same way.
3. A **bias term** (constant 1) is added at the start of each row.

### One-vs-all

Four independent binary classifiers are trained, one per house ("this house" vs "all the others"). Each one is a logistic regression:

```
h(x) = sigmoid(θᵀx) = 1 / (1 + e^(−θᵀx))
```

- **Loss:** binary cross-entropy
- **Optimisation:** batch gradient descent, learning rate `0.01`, `2000` iterations, weights initialised at zero
- **Gradient:** `∂J/∂θⱼ = (1/m) · Σ (h(xᵢ) − yᵢ) · xᵢⱼ`

For prediction, each student is scored by the four classifiers and assigned to the house with the **highest probability**.

Everything (sigmoid, loss, gradient, gradient descent, median, standardisation, split) is written in pure Python, without NumPy or scikit-learn models.

### `weights.json`

```json
{
  "theta": [[...], [...], [...], [...]],
  "means": [...],
  "stds": [...]
}
```

One weight vector of length 9 (bias + 8 features) per house, in the order Gryffindor, Hufflepuff, Ravenclaw, Slytherin.

---

## Project structure

```
dslr/
├── setup.sh
├── dataset_train.csv
├── dataset_test.csv
├── dataset_truth.csv
└── src/
    ├── data/
    │   ├── describe.py          # Part 1: describe from scratch
    │   ├── describe_bonus.py    # bonus: stats for non-numeric columns
    │   ├── histogram.py         # Part 2: histograms by house
    │   ├── scatter_plot.py      # Part 2: two most similar features
    │   ├── pair_plot.py         # Part 2: scatter plot matrix
    │   └── vif.py               # bonus: χ² on Best Hand, VIF selection
    ├── logreg/
    │   ├── model.py             # sigmoid, loss, gradient, one-vs-all
    │   ├── logreg_train.py      # training → weights.json
    │   ├── logreg_predict.py    # prediction → houses.csv
    │   └── accuracy_tester.py   # accuracy against dataset_truth.csv
    └── utils/
        └── utils.py             # loading, median, standardisation, split
```

---

## Tech stack

Python 3 · pandas · matplotlib · seaborn · scikit-learn (accuracy only) · statsmodels · scipy

## Author

- [Dmytro Zolotukhin](https://github.com/dzolotuk) — 42 Paris
