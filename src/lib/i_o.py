import pandas as pd


def read_csv(file, base_path):
    path = base_path + file
    print(f"Read data from {path}")
    return pd.read_csv(base_path + file)
