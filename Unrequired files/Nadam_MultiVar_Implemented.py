import numpy as np
import scipy as sc
from sympy import *
import sys
sys.displayhook = pprint
init_printing()
print('\n\nInitial Setup Complete')


def nadam(c_f, f):
    print("f(x) = ", c_f)
    f_dash_x = diff(c_f, x)
    f_dash_y = diff(c_f, y)

    print("df(x)/dx = ", f_dash_x)
    print("df(x)/dy = ", f_dash_y)

    a = np.array([3, 6])  # initial approximation
    x0_y0 = a

    n = np.array([10, 10])  # learning rate
    # first step to calculate gradient (step_in_x_weight, step_in_y_weight)
    n0 = np.array([0.1, 0.1])

    print("Starting Adam")
    print("	x0_y0 = ", a)
    print("	f(x0) = ", f(a[0], a[1]))

    iter_count = 0
    number_of_iterations = 25

    x_y_t = x0_y0

    ###### Initial gradient calculation ##########
    x0_y0_1 = x0_y0
    x0_y0_2 = x_y_t + n0
    # DO: read photon count
    photon_count1 = 10  # read photon before step
    # DO: apply weights to voltage
    # DO: send voltage
    ### Pause ###
    # DO: read photon count
    photon_count2 = 11  # read photon after voltage with new weight applied
    # ft_dash = np.divide((photon_count2 - photon_count1),
    #                     (x0_y0_2 - x0_y0_1))  # calculate gradient
    # print("initial grad", ft_dash)

    print("Params", x0_y0_1)

    x_y_prev = np.array([0.0, 0.0])
    m0 = np.array([0.0, 0.0])
    mt = np.array([0.0, 0.0])
    v0 = np.array([0.0, 0.0])
    vt = np.array([0.0, 0.0])
    b1 = np.array([0.9, 0.9])  # decay constant
    b2 = np.array([0.999, 0.999])  # decaying mean
    epsilon = np.exp(-8)

########### Nesterov-accelerated adaptive moment estimation #########

    for i in range(number_of_iterations):

        photon_count_1 = 777  # read photon count

        iter_count = iter_count + 1

        x0_y0 = x_y_t  # previous value for params
        m0 = mt
        v0 = vt

        ft_dash = (lambdify([x, y], [f_dash_x, f_dash_y], "numpy"))(
            x_y_t[0], x_y_t[1])

        gt = ft_dash
        gc_t = gt/(1-b1**iter_count)

        mk = b1*m0 + (1-b1)*gt
        mc_k = mk / (1-b1**iter_count)

        vt = b2*v0+(1-b2)*(np.power(gt, 2))
        vc_k = vt / (1-b2**iter_count)

        mb_t = (1-b1)*gc_t + b1*mc_k

        x_y_t = x_y_t - np.multiply(n, (mb_t/(np.sqrt(vc_k)+epsilon)))

        # DO: apply voltage with new params
        # DO: read photon count
        photon_count_2 = 778  # read photon count

        # ft_dash = np.divide((photon_count_2 - photon_count_1),
        #                     (x_y_t - x0_y0))  # calculate gradient

        ft_dash = (lambdify([x, y], [f_dash_x, f_dash_y], "numpy"))(
            x_y_t[0], x_y_t[1])

        print(x_y_t)

    print("Number of Iterations = ", number_of_iterations)
    print("	Minima is at = ", x_y_t)
    print("	Minimum value of Cost Function= ", f(x_y_t[0], x_y_t[1]))


# Syntax Constraints
'''
Syntax Constraints for entering function -
x**y means x raised to the power of y
Function must be algebraic combination of one or more of -
p(x)      Polynomials
exp(x)    Mathematical constant e (2.71828...) raised to power x
pi        Mathematical constant 3.14159...
log(x)    Natural Logarithm
acos(x)   Arc cosine of x
asin(x)   Arc sine of x
atan(x)   Arc tangent of x
cos(x)    Cosine of x
sin(x)    Sine of x
tan(x)    Tangent of x
acosh(x)  Inverse hyperbolic cosine of x
asinh(x)  Inverse hyperbolic sine of x
atanh(x)  Inverse hyperbolic tangent of x
cosh(x)   Hyperbolic cosine of x
sinh(x)   Hyperbolic cosine of x
tanh(x)   Hyperbolic tangent of x

		'''
# example cost functions are given
# 'x**2+y**2'
cost_function = '(-cos(x) * cos(y)) * exp(-(x - pi) ** 2 - (y - pi) ** 2)'
x = Symbol('x')
y = Symbol('y')
# cost_function = input("Enter cost function f(x):  ").strip()
c_f = sympify(cost_function)
# will lambdify c_f for fast parrallel multipoint computation
f = lambdify([x, y], c_f, "numpy")
# print("Verify f(0.9)")
# print N(f(0.9))

nadam(c_f, f)
