from typing import Tuple
import torch
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


def libsvm_to_tensor(filename: str, num_features: int, gpu_device: torch.device) -> Tuple[np.array, np.array]:
  labels, features = svm_read_problem(filename)
  # print(labels)
  data_matrix = torch.zeros((len(features), num_features+1)).double().to(gpu_device)

  for i in range(len(features)):
    for j in features[i].keys():
      data_matrix[i, j - 1] = features[i][j]

  data_matrix[:, -1] = 1.0

  labels = torch.Tensor(labels).double().to(gpu_device)
  labels[labels == 2] = -1
  labels[labels == 0] = -1

  return data_matrix, labels
