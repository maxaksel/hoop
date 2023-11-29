from typing import Tuple
from libsvm.svmutil import *
import numpy as np


def libsvm_to_numpy(filename: str, num_features: int) -> Tuple[np.array, np.array]:
    labels, features = svm_read_problem(filename)
    data_matrix = np.zeros((len(features), num_features))

    for i in range(len(features)):
        for j in features[i].keys():
            data_matrix[i, j - 1] = features[i][j]

    labels = np.array(labels)
    labels[labels == -1] = 0
    labels[labels == 2] = 0

    labels = labels.reshape((len(labels), 1))

    return data_matrix, labels
