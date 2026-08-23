#!/bin/bash
python3 -m venv local_env

source local_env/bin/activate

pip install --upgrade pip

pip install pandas matplotlib scikit-learn seaborn statsmodels scipy PyQt5

echo "Dependencies installed"