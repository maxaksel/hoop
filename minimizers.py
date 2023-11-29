import numpy as np
import torch
from scipy.optimize import newton, fsolve
from typing import Tuple, Callable, List
from tqdm import tqdm


def gradient_descent_general(start_theta: torch.Tensor, num_iters: int, loss_func: Callable, grad_func: Callable,
                             grad_lipschitz_val: float, gpu_device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    alpha = 1/grad_lipschitz_val
    print(alpha)
    theta_hat = start_theta
    loss = torch.zeros((int(num_iters),)).double().to(gpu_device)

    for i in tqdm(range(int(num_iters))):
        loss[i] = loss_func(theta_hat)
        theta_hat -= alpha*grad_func(theta_hat)

    return loss, theta_hat


def gradient_descent_tracked(start_theta: torch.Tensor, num_iters: int, loss_func: Callable,
                             grad_func: Callable,
                             grad_lipschitz_val: float,
                             gpu_device: torch.device) -> Tuple[torch.Tensor, List, torch.Tensor]:
    alpha = 1/grad_lipschitz_val
    print(alpha)
    theta_hat = start_theta
    loss = torch.zeros((int(num_iters),)).double().to(gpu_device)
    iterates = []

    for i in tqdm(range(int(num_iters))):
        loss[i] = loss_func(theta_hat)
        iterates.append(theta_hat)
        theta_hat -= alpha*grad_func(theta_hat)

    return loss, iterates, theta_hat


def acc_gradient_descent(start_theta: torch.Tensor, num_iters: int, loss_func: Callable, grad_func: Callable,
                         grad_lipschitz_val: float, gpu_device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    alpha = 1/grad_lipschitz_val
    print(alpha)
    lam = torch.zeros((int(num_iters+1),)).double().to(gpu_device)
    loss = torch.zeros((int(num_iters),)).double().to(gpu_device)
    theta_hat = torch.clone(start_theta)
    y_prev = torch.clone(start_theta)
    y_curr = torch.clone(start_theta)

    for i in tqdm(range(1, int(num_iters + 1))):
        lam[i] = 0.5*(1+torch.sqrt(1+4*lam[i-1]*lam[i-1]))

    for i in tqdm(range(int(num_iters))):
        loss[i] = loss_func(theta_hat)
        grads = grad_func(theta_hat)

        # print(i)
        # print(grads)
        # time.sleep(100000)

        gamma = (1 - lam[i]) / (lam[i+1])
        y_curr = theta_hat - alpha*grads
        theta_hat = (1 - gamma) * y_curr + gamma * y_prev
        y_prev = y_curr

    return loss, theta_hat


def plain_newton(start_theta: torch.Tensor, num_iters: int, loss_func: Callable, grad_func: Callable,
                 hessian_func: Callable, alpha: float, gamma: float,
                 gpu_device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    :param start_theta:
    :param num_iters:
    :param loss_func:
    :param grad_func:
    :param hessian_func:
    :param alpha:
    :param gamma:
    :param gpu_device:
    :return:
    """
    loss = torch.zeros((int(num_iters),)).double().to(gpu_device)
    theta_hat = torch.clone(start_theta)

    for i in tqdm(range(int(num_iters))):
        loss[i] = loss_func(theta_hat)
        grads = grad_func(theta_hat)
        H = hessian_func(theta_hat)

        h = -gamma*torch.linalg.inv(H + (1.0/alpha)*torch.eye(H.shape[0])) @ grads
        theta_hat += torch.real(h)

    return loss, theta_hat


def grad_f_reg(theta: np.array, *args) -> float:
    """
    Gradient of regularized subproblem to solve.
    """
    theta_orig, grads, H, M = args
    return grads + H@(theta-theta_orig) + (M/2)*np.sqrt(np.linalg.norm(theta-theta_orig)) * (theta-theta_orig)


def non_accelerated_cubic_newton(start_theta: torch.Tensor, num_iters: int, loss_func: Callable, grad_func: Callable,
                                 hessian_func: Callable, hessian_lipschitz_val: torch.Tensor,
                                 gpu_device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    :param start_theta:
    :param num_iters:
    :param loss_func:
    :param grad_func:
    :param hessian_func:
    :param hessian_lipschitz_val:
    :param gpu_device:
    :return:
    """
    loss = torch.zeros((int(num_iters),)).double().to(gpu_device)
    theta_hat = torch.clone(start_theta)
    M = hessian_lipschitz_val
    d = len(start_theta)

    # print(type(M.cpu().numpy()))

    for i in tqdm(range(int(num_iters))):
        loss[i] = loss_func(theta_hat)

        grads = grad_func(theta_hat)
        H = hessian_func(theta_hat)

        theta_hat = fsolve(grad_f_reg, np.zeros(d), args=(theta_hat.cpu().numpy(), grads.cpu().numpy(), H.cpu().numpy(),
                                                          M.cpu().numpy()))
        theta_hat = torch.Tensor(theta_hat).double().to(gpu_device)

    return loss, theta_hat


def opt_subproblem(r: float, *args) -> float:
    """

    :param r:
    :param args:
    :return:
    """
    Lambda = args[0]
    g_bar = args[1]
    M = args[2]

    # print(type(Lambda))

    result = 0
    for i in range(Lambda.shape[0]):
        result += g_bar[i]**2 / (Lambda[i] + M*r/2)**2

    return result - r*r


def accelerated_cubic_newton(start_theta: torch.Tensor, num_iters: int, loss_func: Callable, grad_func: Callable,
                             hessian_func: Callable, hessian_lipschitz_val: torch.Tensor,
                             gpu_device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    :param start_theta:
    :param num_iters:
    :param loss_func:
    :param grad_func:
    :param hessian_func:
    :param hessian_lipschitz_val:
    :param gpu_device:
    :return:
    """
    loss = torch.zeros((int(num_iters),)).double().to(gpu_device)
    theta_hat = torch.clone(start_theta)
    M = hessian_lipschitz_val

    for i in tqdm(range(int(num_iters))):
        loss[i] = loss_func(theta_hat)

        grads = grad_func(theta_hat)
        H = hessian_func(theta_hat)
        Lambda, U = torch.linalg.eig(H)
        U = torch.real(U).double()  # U should be real-valued
        Lambda = torch.real(Lambda).double()

        g_bar = U.T@grads

        r_star = newton(opt_subproblem, 0.5, maxiter=1000, args=(Lambda.cpu().numpy(), g_bar.cpu().numpy(),
                                                                 M.cpu())).double().to(gpu_device)
        if r_star < 0:
            raise Exception("r < 0!")

        h = -U@torch.linalg.inv(torch.diag(Lambda) + M*r_star/2*torch.eye(H.shape[0]).to(gpu_device))@g_bar
        theta_hat += torch.real(h)

    return loss, theta_hat
