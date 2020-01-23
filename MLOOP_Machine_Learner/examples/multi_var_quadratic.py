import numpy as np
from sklearn import *


def rand_quad(x, dim):
     # Quadratic of form  x.T*A*x + B.T*x + C
    # Symmetric positive definite matrix
    A = datasets.make_spd_matrix(dim, random_state=2)
    np.random.seed(1)
    b = np.random.rand(dim)
    c = np.random.rand()
    t1 = np.dot(x.T, A)
    t2 = np.dot(t1, x)
    t3 = np.dot(b.T, x)
    Z = t2 + t3 + c
    # local minima:  c - 1/4  * b.T * A^(-1) * b
    local_minima = c - 0.25*np.dot(np.dot(b.T, np.linalg.inv(A)), b)
    # loca minima  at: -1/2 * A^(-1)* b
    local_minima_vals = -0.5 * np.dot(np.linalg.inv(A), b)
    return Z, local_minima, local_minima_vals


# Template to run function with dimensions d
d = 48
x = np.zeros(d)  # set the value for variables
Z, local_minima, local_minima_vals = rand_quad(x, d)
print(Z)
print(local_minima)
print(local_minima_vals)
