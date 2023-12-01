# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from transport import OptimalTransport
from minimizers import gradient_descent_general, gradient_descent_tracked, acc_gradient_descent
from minimizers import non_accelerated_cubic_newton, accelerated_cubic_newton
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import imageio


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    n = 100
    gamma = 0.1
    gpu_device = torch.device('cpu')  # run on CPU for now

    transport_problem = OptimalTransport(n=n, gamma=gamma, gpu_device=gpu_device)

    p = torch.normal(mean=torch.zeros(n,), std=torch.ones(n,)).double().to(gpu_device)
    p = (p - p.min())
    p[int(n / 2):] = 0
    p /= p.sum()
    q = torch.normal(mean=torch.zeros(n,), std=torch.ones(n,)).double().to(gpu_device)
    q = (q - q.min())
    q[:int(n / 2)] = 0
    q /= q.sum()
    # p = torch.Tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).double().to(gpu_device)
    # q = torch.Tensor([0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0]).double().to(gpu_device)

    # epsilon = 1e-5
    #
    # p[0:] = epsilon
    # q[0:] = epsilon
    # p[0] = 1 - 99*epsilon
    # q[-1] = 1 - 99*epsilon

    # print(p)
    # print(q)

    plt.figure()
    plt.stem(p)
    plt.show()

    plt.figure()
    plt.stem(q)
    plt.show()

    transport_problem.set_distributions(p, q)

    num_iters = 200

    # Gradient Descent
    theta_hat = torch.Tensor(np.random.rand(2 * n)).double().to(gpu_device)
    orig_theta_hat = torch.clone(theta_hat)
    loss_gd, theta_i, theta_star = gradient_descent_tracked(theta_hat, num_iters, transport_problem.loss_func,
                                                            transport_problem.grad_func,
                                                            10, gpu_device)

    # Accelerated Gradient Descent
    theta_hat = torch.clone(orig_theta_hat)
    loss_agd, theta_star = acc_gradient_descent(theta_hat, num_iters, transport_problem.loss_func,
                                                transport_problem.grad_func,
                                                10, gpu_device)

    # # Cubic Regularization Slow
    # theta_hat = torch.clone(orig_theta_hat)
    # loss_na2, theta_star = non_accelerated_cubic_newton(theta_hat, num_iters, transport_problem.loss_func,
    #                                                     transport_problem.grad_func,
    #                                                     transport_problem.hessian_func,
    #                                                     transport_problem.lipschitz_constant(2), gpu_device)
    #
    # # Cubic Regularization Fast
    # theta_hat = torch.clone(orig_theta_hat)
    # loss_a2, theta_star = accelerated_cubic_newton(theta_hat, num_iters, transport_problem.loss_func,
    #                                                transport_problem.grad_func,
    #                                                transport_problem.hessian_func,
    #                                                transport_problem.lipschitz_constant(2), gpu_device)

    plt.plot(loss_gd, label="Gradient Descent")
    plt.plot(loss_agd, label="Acc. Gradient Descent")
    # plt.plot(loss_na2, label="Cubic Reg. Newton Method")
    # plt.plot(loss_a2, label="Acc. Cubic Reg. Newton Method")
    plt.legend()
    plt.show()

    T, dist = transport_problem.wasserstein_distance(theta_star)
    # print("Theta star")
    # print(theta_star)
    # print("Transport plan:")
    # print(T)
    plt.imshow(T)
    plt.title("2-Wasserstein Distance: " + str(dist.item()))
    plt.show()
