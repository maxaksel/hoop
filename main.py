"""
Runs optimization methods on difficult functions presented by Nesterov.
"""
import torch
from badfuncs import NesterovFunctions
from minimizers import gradient_descent_general, acc_gradient_descent, non_accelerated_cubic_newton
from minimizers import accelerated_cubic_newton, plain_newton
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    gpu_device = torch.device('cpu')  # run on CPU for now
    bf = NesterovFunctions(k=10, p=2, d=25, gpu_device=gpu_device)

    # Gradient Descent
    theta_hat = torch.Tensor(np.random.rand(25)).double().to(gpu_device)
    orig_theta_hat = torch.clone(theta_hat)
    loss_gd, theta_star = gradient_descent_general(theta_hat, 100, bf.nesterov_f, bf.nesterov_grad,
                                                   2*bf.nesterov_lipschitz(), gpu_device)
    error_gd = torch.abs(loss_gd - bf.ideal_minimum())

    # Accelerated Gradient Descent
    theta_hat = torch.clone(orig_theta_hat)
    loss_agd, theta_star = acc_gradient_descent(theta_hat, 100, bf.nesterov_f, bf.nesterov_grad,
                                                2*bf.nesterov_lipschitz(), gpu_device)
    error_agd = torch.abs(loss_agd - bf.ideal_minimum())
    print(bf.nesterov_lipschitz())

    # Quadratic Newton
    theta_hat = torch.clone(orig_theta_hat)
    loss_qn, theta_star = plain_newton(theta_hat, 100, bf.nesterov_f, bf.nesterov_grad,
                                       bf.nesterov_hessian, 10, 1, gpu_device)
    error_qn = torch.abs(loss_qn - bf.ideal_minimum())

    # Non-Accelerated Cubic Newton
    theta_hat = torch.clone(orig_theta_hat)
    loss_nan, theta_star = non_accelerated_cubic_newton(theta_hat, 100, bf.nesterov_f, bf.nesterov_grad,
                                                        bf.nesterov_hessian, torch.Tensor([bf.nesterov_lipschitz()]),
                                                        gpu_device)
    error_nan = torch.abs(loss_nan - bf.ideal_minimum())

    # Accelerated Cubic Newton
    theta_hat = torch.clone(orig_theta_hat)
    loss_an, theta_star = accelerated_cubic_newton(theta_hat, 100, bf.nesterov_f, bf.nesterov_grad, bf.nesterov_hessian,
                                                   torch.Tensor([bf.nesterov_lipschitz()]), gpu_device)
    error_an = torch.abs(loss_an - bf.ideal_minimum())

    plt.plot(loss_gd, label="Gradient Descent")
    plt.plot(loss_agd, label="Acc. Gradient Descent")
    # plt.plot(loss_qn, label="Quadratic Newton")
    plt.plot(loss_nan, label="Cubic Newton")
    plt.plot(loss_an, label="Acc. Cubic Newton")
    plt.ylabel("Function Value")
    plt.xlabel("Iterations")
    plt.legend()
    plt.show()
