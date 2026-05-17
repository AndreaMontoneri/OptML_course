# -*- coding: utf-8 -*-
"""Lab 3.

Gradient descent
"""

import numpy as np


def calculate_mse(e):
    """
    Calculate the mean squared error for vector e.
    Arguments:
        e: error vector -> (n)
    Returns:
        MSE: Mean squared error of error vector -> float
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # Compute mean squared error
    # ***************************************************
    n = e.shape[0]
    MSE = (0.5 / n) * (np.linalg.norm(e) ** 2)
    return MSE


def compute_gradient(b, A, x):
    """
    Compute the gradient of MSE loss

    Arguments:
        b: Label vector -> (n)
        A: Data matrix -> (n,d)
        x: Parameters vector -> (d)
    Returns:
        (grad, err)
        - grad: Gradient of MSE at x -> (d)
        - err: MSE loss for x -> float
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # Compute gradient and objective
    # ***************************************************
    n = b.shape[0]
    error = A @ x - b
    grad = (1 / n) * A.T @ error
    return grad, error


def gradient_descent(b, A, initial_x, max_iters, gamma):
    """
    Gradient descent algorithm.

    Arguments:
        b: Label vector -> (n)
        A: Data matrix -> (n,d)
        initial_x: Initial parameters vector -> (d)
        max_iters: maximum iteration for the algorithm -> integer
        gamma: learning rate -> float
    Returns:
        (objectives, x_liste)
            - objectives: loss values at each epoch -> list[float]
            - x_liste: parameter vectors at each epcoch -> list[(d)]
    """
    # Define parameters to store x and objective func. values
    x_liste = [initial_x]
    objectives = []
    for n_iter in range(max_iters):
        x_old = x_liste[-1]
        # ***************************************************
        # INSERT YOUR CODE HERE
        # Compute gradient and objective function
        # ***************************************************
        grad, error = compute_gradient(b, A, x_old)
        obj = calculate_mse(error)
        # ***************************************************
        # INSERT YOUR CODE HERE
        # Update x by a gradient descent step
        # ***************************************************
        x_new = x_old - gamma * grad
        # store x and objective function value
        x_liste.append(x_new)
        objectives.append(obj)
        print(
            "Gradient Descent({bi}/{ti}): objective={l}".format(
                bi=n_iter, ti=max_iters - 1, l=obj
            )
        )
    return objectives, x_liste
