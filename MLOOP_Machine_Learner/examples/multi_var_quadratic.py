import numpy as np
from sklearn import *


class quadratic():
    '''Quadratic of form  x.T*A*x + b.T*x + b '''

    def __init__(self,
                 dimension,  # set dimension
                 isseed=True,  # seed the randomised parameters
                 seed=1  # define the seed parameter space
                 ):

        self.dimension = dimension
        self.seed = seed

        if isseed:
            np.random.seed(self.seed)
        # set up coefficients
        # A is a symmetric positive definite matrix
        self.A = datasets.make_spd_matrix(
            self.dimension, random_state=self.seed)
        self.b = np.random.rand(self.dimension)
        self.c = np.random.rand()

    def evaluate_at_params(self, parameters):

        if not len(parameters) == self.dimension:
            print("Number of params must be same length as number of dimensions")
            raise ValueError
        parameters = np.array(parameters)

        t1 = np.dot(parameters.T, self.A)
        t2 = np.dot(t1, parameters)
        t3 = np.dot(self.b.T, parameters)

        self.Z = t2 + t3 + self.c
        return self.Z

    def get_min(self):
        '''Gets local minima value:  c - 1/4  * b.T * A^(-1) * b'''

        self.local_minima = self.c - 0.25 * \
            np.dot(np.dot(self.b.T, np.linalg.inv(self.A)),
                   self.b)  # calculate minima

        return self.local_minima

    def get_min_parameters(self):
        ''' Gets local minima parameters using -1/2 * A^(-1)* b'''
        self.local_minima_vals = -0.5 * \
            np.dot(np.linalg.inv(self.A), self.b)  # calculate minima values

        return self.local_minima_vals
