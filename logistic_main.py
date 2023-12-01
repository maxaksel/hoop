import torch
from minimizers import gradient_descent_general, gradient_descent_tracked, acc_gradient_descent
from minimizers import non_accelerated_cubic_newton, accelerated_cubic_newton
from minimizers import plain_newton
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from logistic import sigma, LogisticFunctions
import imageio


def classification_accuracy(labels: torch.Tensor, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """

    :param labels: data labels in {-1, 1}.
    :param X: data matrix with features as columns.
    :param theta:
    :return:
    """
    labels_hat = sigma(X @ theta)
    # print(labels_hat)
    labels_hat = (labels_hat >= 0.5).double()
    labels_hat = 2*labels_hat - 1  # convert to {-1, 1}
    # print(labels)
    # print(labels_hat)

    diff = labels_hat - labels
    num_different = torch.count_nonzero(diff)

    return (len(labels) - num_different) / len(labels)


if __name__ == '__main__':
    gpu_device = torch.device('cpu')
    d = 10  # dimension of feature space
    N = 100  # number of data points
    X = torch.tensor(2 * (np.random.rand(N, d + 1) - 0.5)).to(
        gpu_device)  # data matrix (features are columns) randomly generated with uniform distribution [-1, 1]
    X[:, -1] = 1
    weights = torch.tensor(np.random.rand(d + 1)).to(gpu_device)
    weights /= torch.linalg.norm(weights)  # normalize weights

    y = 2 * (sigma(X @ weights.T) > 0.5).double() - 1  # generate labels in {-1, 1}

    classification_accuracy(y, X, weights)

    X = X.to(gpu_device)
    weights = weights.to(gpu_device)
    y = y.to(gpu_device)

    logistic_problem = LogisticFunctions(X, y, gpu_device)

    # Gradient Descent
    theta_hat = torch.Tensor(np.random.rand(d + 1)).double().to(gpu_device)
    orig_theta_hat = torch.clone(theta_hat)

    start_time = time.time()
    loss_o1_q, theta_hat = gradient_descent_general(theta_hat, 200, logistic_problem.loss_func,
                                                    logistic_problem.loss_func_grad, logistic_problem.grad_lipschitz(),
                                                    gpu_device)
    end_time = time.time()
    print("Gradient Descent\n=================")
    print("Run time:")
    print(end_time - start_time)
    print(loss_o1_q[-1])

    # plt.semilogy(loss_o1_q.cpu())
    # plt.xlabel(r"Iterations")
    # plt.ylabel(r"$l(\theta; X, y)$")
    # plt.show()

    print("Classification accuracy:")
    print(classification_accuracy(y, X, theta_hat))

    # Accelerated Gradient Descent
    theta_hat = torch.clone(orig_theta_hat)
    start_time = time.time()
    loss_o1_aq, theta_hat = acc_gradient_descent(theta_hat, 200, logistic_problem.loss_func,
                                                    logistic_problem.loss_func_grad, logistic_problem.grad_lipschitz(),
                                                    gpu_device)
    end_time = time.time()
    print("Acc. Gradient Descent\n=================")
    print("Run time:")
    print(end_time - start_time)
    print(loss_o1_aq[-1])

    # plt.semilogy(loss_o1_aq.cpu())
    # plt.xlabel(r"Iterations")
    # plt.ylabel(r"$l(\theta; X, y)$")
    # plt.show()

    print("Classification accuracy:")
    print(classification_accuracy(y, X, theta_hat))

    # Quadratic Newton
    theta_hat = torch.clone(orig_theta_hat)
    start_time = time.time()
    loss_o2_q, theta_hat = plain_newton(theta_hat, 200, logistic_problem.loss_func,
                                        logistic_problem.loss_func_grad,
                                        logistic_problem.loss_func_hessian,
                                        10,
                                        1,
                                        gpu_device)
    end_time = time.time()
    print("Quadratic Newton\n=================")
    print("Run time:")
    print(end_time - start_time)
    print(loss_o2_q[-1])

    # plt.semilogy(loss_o2_q.cpu())
    # plt.xlabel(r"Iterations")
    # plt.ylabel(r"$l(\theta; X, y)$")
    # plt.show()

    print("Classification accuracy:")
    print(classification_accuracy(y, X, theta_hat))

    # Cubic Newton
    theta_hat = torch.clone(orig_theta_hat)
    start_time = time.time()
    loss_o2_c, theta_hat = non_accelerated_cubic_newton(theta_hat, 200, logistic_problem.loss_func,
                                                        logistic_problem.loss_func_grad,
                                                        logistic_problem.loss_func_hessian,
                                                        logistic_problem.hessian_lipschitz(),
                                                        gpu_device)
    end_time = time.time()
    print("Cubic Newton\n=================")
    print("Run time:")
    print(end_time - start_time)
    print(loss_o2_c[-1])

    print("Classification accuracy:")
    print(classification_accuracy(y, X, theta_hat))

    # Acc. Cubic Newton
    theta_hat = torch.clone(orig_theta_hat)
    start_time = time.time()
    loss_o2_ac, theta_hat = accelerated_cubic_newton(theta_hat, 200, logistic_problem.loss_func,
                                                    logistic_problem.loss_func_grad,
                                                    logistic_problem.loss_func_hessian,
                                                    logistic_problem.hessian_lipschitz(),
                                                    gpu_device)
    end_time = time.time()
    print("Acc. Cubic Newton\n=================")
    print("Run time:")
    print(end_time - start_time)

    print(loss_o2_ac[-1])

    # plt.semilogy(loss_o1_q.cpu())
    # plt.xlabel(r"Iterations")
    # plt.ylabel(r"$l(\theta; X, y)$")
    # plt.show()

    print("Classification accuracy:")
    print(classification_accuracy(y, X, theta_hat))

    # Combined Graph
    plt.semilogy(loss_o1_q.cpu(), label="Gradient Descent")
    plt.semilogy(loss_o1_aq.cpu(), label="Acc. Gradient Descent")
    plt.semilogy(loss_o2_q.cpu(), label="Quadratic Newton")
    plt.semilogy(loss_o2_c.cpu(), label="Cubic Newton")
    plt.semilogy(loss_o2_ac.cpu(), label="Acc. Cubic Newton")
    plt.xlabel(r"Iterations")
    plt.ylabel(r"$l(\theta; X, y)$")
    plt.legend()
    plt.show()



