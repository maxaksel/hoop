import torch
import numpy as np


def sigma(x):
    return 1/(1+torch.exp(-x))


class LogisticFunctions:
    def __init__(self, data_matrix: torch.Tensor, labels: torch.Tensor, gpu_device: torch.device):
        """

        :param data_matrix:
        :param labels:
        :param gpu_device:
        """
        self.gpu_device = gpu_device
        self.data_matrix = data_matrix
        self.labels = labels

    def loss_func(self, weights: torch.Tensor) -> torch.Tensor:
        """

        :param weights:
        :return:
        """
        loss = torch.sum(torch.log(1 + torch.exp(-self.labels * (self.data_matrix @ weights))))
        return loss / self.data_matrix.shape[0]

    def loss_func_grad(self, theta: torch.Tensor) -> torch.Tensor:
        """

        :param theta:
        :return:
        """
        z = -self.labels / (1 + torch.exp(self.labels * (self.data_matrix @ theta)))
        return self.data_matrix.T @ z / self.data_matrix.shape[0]

    def loss_func_hessian(self, theta: torch.Tensor) -> torch.Tensor:
        """

        :param theta:
        :return:
        """
        H = torch.zeros((len(theta), len(theta))).double().to(self.gpu_device)

        sigma_lookup = sigma(self.data_matrix @ theta) * (1 - sigma(self.data_matrix @ theta))

        for j in range(len(theta)):
            for k in range(j, len(theta)):
                H[j, k] = torch.sum(self.data_matrix[:, j] * self.data_matrix[:, k] * sigma_lookup)
                H[k, j] = H[j, k]

        return H / self.data_matrix.shape[0]

    def grad_lipschitz(self) -> float:
        """

        max_sigma^2 is the largest eigenvalue of X'X.
        The first-order Lipschitz constant is bounded above by
        1/(4N) sqrt(lambda(X'X)), which is proportional to the
        induced two-norm of X'X.

        :return: upper bound of the first-order Lipschitz constant.
        """

        max_sigma = torch.linalg.svdvals(self.data_matrix)[0]
        return (1 / (4 * self.data_matrix.shape[0])) * max_sigma * max_sigma

    def hessian_lipschitz(self) -> float:
        """
        max_sigma^2 is the largest eigenvalue of X'X.
        The second-order Lipschitz constant is bounded above by
        1/(3*sqrt(6)) * max(norm(x_i)) * sqrt(lambda(X'X)), which
        is proportional to the induced two-norm of X'X.

        :return: upper bound of the second-order Lipschitz constant.
        """
        max_sigma = torch.linalg.svdvals(self.data_matrix)[0]
        return 1 / (6 * np.sqrt(3)) * (1 / self.data_matrix.shape[0]) * torch.max(
            torch.linalg.norm(self.data_matrix, axis=0)) * max_sigma
