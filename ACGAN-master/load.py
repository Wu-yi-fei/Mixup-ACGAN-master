import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import csv

sys.path.append('..')
import numpy as np


def load_solar_data():
    # Example dataset created for controllable GANs PV scenarios generation
    # Data from NREL solar integrated datasets
    with open('data/solar.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows)
    rows = rows[1:104833, 1:]
    rows = np.array(rows, dtype=float)
    m = np.max(rows, axis=0)
    m = np.tile(m[np.newaxis,], [104832, 1])
    rows = rows / m
    row = np.concatenate([rows[:, 0, np.newaxis], rows[:, 1, np.newaxis]])
    X = np.reshape(row.T, (-1, 576))
    return X


if __name__ == '__main__':
    X = load_solar_data()
