import numpy as np


def pearson_correlation(X, Y):
    return np.corrcoef(X, Y)[0, 1]


