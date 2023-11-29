import math
from typing import Tuple
import numpy as np
import torch


def softmax(gamma, in_vector):
    return gamma*torch.logsumexp(in_vector/gamma, 0)


def grad_smax(gamma, x: torch.Tensor) -> torch.Tensor:
    # print("Lambda:")
    # print(x)
    scaled_x = x/gamma
    scaled_x -= scaled_x.max()
    exp_vector = torch.exp(scaled_x)
    # print("Exp vector:")
    # print(exp_vector)
    # print(exp_vector.max())
    # print(torch.sum(exp_vector))
    result = exp_vector / torch.sum(exp_vector)
    # print(result)

    return result


def hessian_smax(gamma, x: torch.Tensor) -> torch.Tensor:
    grad = grad_smax(gamma, x)
    return 1/gamma * (torch.diag(grad) - torch.outer(grad, grad))


class OptimalTransport:
    def __init__(self, n: int, gamma: float, gpu_device: torch.device):
        """

        :param n:
        :param gamma:
        :param gpu_device:
        """
        self.n = n
        self.gamma = gamma
        self.gpu_device = gpu_device
        self.p = None
        self.q = None
        self.b = None

        self.M = torch.zeros((n, n)).double().to(gpu_device)
        for i in range(n):
            for j in range(n):
                self.M[i, j] = (i - j)*(i - j)/(n*n)  # normalize by matrix size to stabilize numerically

        self.A = torch.zeros((2*n, n*n)).double().to(gpu_device)
        for k in range(n):
            self.A[0:n, k * n:k * n + n] = torch.eye(n)
        for i in range(n, 2 * n):
            self.A[i, (i - n) * n:(i - n + 1) * n] = 1.0

    def set_distributions(self, p: torch.Tensor, q: torch.Tensor):
        """

        :param p: column vector
        :param q: column vector
        :return:
        """
        self.p = p
        self.q = q
        self.b = torch.hstack((p, q)).to(self.gpu_device)

    def loss_func(self, lam: torch.Tensor) -> float:
        smax_in = torch.matmul(torch.transpose(self.A, 0, 1), lam) - torch.flatten(self.M)
        # print(lam.shape)
        # print(self.b.shape)
        return softmax(self.gamma, smax_in) - torch.dot(lam, self.b)

    def grad_func(self, lam: torch.Tensor) -> torch.Tensor:
        return self.A @ grad_smax(self.gamma, torch.transpose(self.A, 0, 1) @ lam - torch.flatten(self.M)) - self.b

    def hessian_func(self, lam: torch.Tensor) -> torch.Tensor:
        return self.A @ hessian_smax(self.gamma,
                                     torch.transpose(self.A, 0, 1) @ lam - torch.flatten(self.M)) @ torch.transpose(self.A, 0, 1)

    def wasserstein_distance(self, opt_lam: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the optimal transport plan and entropy-regularized Wasserstein distance for optimal ksi and eta
        parameters of a particular pair of distributions.

        :param opt_lam: optimal ksi and eta parameters for a particular pair of distributions obtained by running some
        minimizer on loss_func, grad_func, and hessian_func.
        :return: a tuple containing a Kantorovich transport plan followed by the entropy-regularized Wasserstein
        distance.
        """
        xi = opt_lam[0:self.n]
        eta = opt_lam[self.n:]

        M_adjust = torch.zeros((self.n, self.n)).double().to(self.gpu_device)
        for i in range(self.n):
            for j in range(self.n):
                M_adjust[i, j] = (-self.M[i, j] + xi[i] + eta[j]) / self.gamma
        M_adjust -= M_adjust.max()

        transport_plan = torch.exp(M_adjust) / torch.exp(M_adjust).sum()

        # entropy = -torch.sum(torch.log(transport_plan + 1e-14) * transport_plan)
        # wasserstein_distance = torch.sum(self.M * transport_plan) - self.gamma * entropy
        wasserstein_distance = self.n*self.n*torch.sum(self.M * transport_plan)

        return transport_plan, wasserstein_distance

    def lipschitz_constant(self, p: int) -> torch.Tensor:
        numerator = ((p+1)/(np.log(p+2)))**(p+1)*math.factorial(p)
        denominator = self.gamma ** p
        return torch.Tensor([numerator/denominator])
