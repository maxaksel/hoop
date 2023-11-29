import math
import torch
import numpy as np


class NesterovFunctions:
    def __init__(self, k: int, p: int, d: int, gpu_device: torch.device):
        """

        :param k:
        :param p: order of the method being tested
        :param d: dimension of the input vector of the loss function
        """
        self.k = k
        self.p = p
        self.d = d
        self.gpu_device = gpu_device

        Uk = torch.eye(k)
        for i in range(k - 1):
            Uk[i, i + 1] = -1

        Ink = torch.eye(d - k)
        self.Ak = torch.zeros((d, d))

        self.Ak[0:k, 0:k] = Uk
        self.Ak[k:, k:] = Ink

        self.Ak = self.Ak.double().to(gpu_device)

    def nu_p(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        return 1 / (self.p + 1) * torch.sum(torch.pow(torch.abs(x), self.p + 1))

    def nesterov_f(self, x: torch.Tensor) -> float:
        """

        :param x:
        :return:
        """
        return self.nu_p(torch.matmul(self.Ak, x)) - x[0]

    def nesterov_grad(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        Akx = self.Ak @ x
        eta_gradient = torch.pow(torch.abs(Akx), self.p - 1) * Akx
        gradient_final = self.Ak.T @ eta_gradient
        gradient_final[0] -= 1

        return gradient_final

    def nesterov_hessian(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        Akx = self.Ak @ x
        return 3 * self.Ak.T @ torch.diag(Akx) @ torch.diag(Akx) @ self.Ak

    def nesterov_lipschitz(self) -> float:
        """

        :return:
        """
        val = 2 * math.factorial(self.p)
        return float(val)

    def ideal_minimum(self) -> float:
        """

        :return:
        """
        return -self.k*self.p/(self.p+1)

    def ideal_params(self) -> torch.Tensor:
        """

        :return:
        """
        theta_solution = torch.Tensor(np.ones((self.d, 1))).double().to(self.gpu_device)
        for i in range(0, self.d):
            val = self.k - (i + 1) + 1
            theta_solution[i] = val if val > 0 else 0

        return theta_solution
