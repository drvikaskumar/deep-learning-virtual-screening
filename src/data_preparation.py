"""
Data preparation script for QSAR regression workflow.
"""

import pandas as pd


def load_dataset(csv_file):
    """Load dataset from CSV."""
    df = pd.read_csv(csv_file)
    return df


if __name__ == "__main__":
    print("QSAR data preparation module")
