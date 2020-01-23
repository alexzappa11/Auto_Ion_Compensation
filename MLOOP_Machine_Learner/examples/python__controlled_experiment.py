# Imports for python 2 compatibility
from __future__ import absolute_import, division, print_function
__metaclass__ = type

# Imports for M-LOOP
import mloop.interfaces as mli
import mloop.controllers as mlc
import mloop.visualizations as mlv
import mloop.neuralnet as nn

# Other imports
import numpy as np
import time
from multi_var_quadratic import *
from sklearn import *

# Declare your custom class that inherets from the Interface class


class CustomInterface(mli.Interface):

    # Initialization of the interface, including this method is optional
    def __init__(self):
        # You must include the super command to call the parent class, Interface, constructor
        super(CustomInterface, self).__init__()

        # Attributes of the interface can be added here
        # If you want to precalculate any variables etc. this is the place to do it
        dim = 48
        self.A = datasets.make_spd_matrix(dim, random_state=2)
        np.random.seed(1)
        self.b = np.random.rand(dim)
        self.c = np.random.rand()
        # local minima:  c - 1/4  * b.T * A^(-1) * b
        self.local_minima = self.c - 0.25 * \
            np.dot(np.dot(self.b.T, np.linalg.inv(self.A)), self.b)
        # loca minima  at: -1/2 * A^(-1)* b
        self.local_minima_vals = -0.5 * np.dot(np.linalg.inv(self.A), self.b)

    # You must include the get_next_cost_dict method in your class
    # this method is called whenever M-LOOP wants to run an experiment
    def get_next_cost_dict(self, params_dict):

        # Get parameters from the provided dictionary
        params = params_dict['params']

        # Here you can include the code to run your experiment given a particular set of parameters
        t1 = np.dot(params.T, self.A)
        t2 = np.dot(t1, params)
        t3 = np.dot(self.b.T, params)
        Z = t2 + t3 + self.c
        cost = Z
        # cost = np.sum(params)
        # There is no uncertainty in our result
        uncer = 0
        # The evaluation will always be a success
        bad = False
        # Add a small time delay to mimic a real experiment
        # time.sleep(1)

        # The cost, uncertainty and bad boolean must all be returned as a dictionary
        # You can include other variables you want to record as well if you want
        cost_dict = {'cost': cost, 'uncer': uncer, 'bad': bad}
        return cost_dict


def main():
    # M-LOOP can be run with three commands

    # First create your interface
    interface = CustomInterface()
    # Next create the controller, provide it with your controller and any options you want to set
    controller = mlc.create_controller(interface,
                                       #   controller_type="gaussian_process",
                                       controller_type='differential_evolution',
                                       #   controller_type='nelder_mead',
                                       #    controller_type='neural_net',
                                       max_num_runs=100,
                                       num_params=48,
                                       #    target_cost=-9.627,
                                       min_boundary=np.full(48, -10),
                                       max_boundary=np.full(48, 10),
                                       # first parameters to try in initial training
                                       #    first_params=interface.local_minima_vals*1.8,
                                       trust_region=0.01,  # maximum move % from best params

                                       ###### Evolution strategy####
                                       # evolution strategy can be 'best1', 'best2', 'rand1' and 'rand2'. Best uses the best point, rand uses a random one, the number indicates the number of directions added.
                                       #    evolution_strategy='best2',
                                       population_size=100,  # a multiplier for the population size of a generation
                                       # the minimum and maximum value for the mutation scale factor. Each generation is randomly selected from this. Each value must be between 0 and 2.
                                       #    mutation_scale=(0.5, 1),
                                       # the probability a parameter will be resampled during a mutation in a new generation
                                       cross_over_probability=1,
                                       # the fraction the standard deviation in the costs of the population must reduce from the initial sample, before the search is restarted.
                                       restart_tolerance=0.00001,
                                       predict_global_minima_at_end=True,  # find predicted global minima at end
                                       #    cost_has_noise=True,
                                       #    noise_level=0.5,

                                       ##### Neural Net ###
                                       training_type='differential_evolution',

                                       ###### Nelder Mead ####
                                       # initial corner of the simplex
                                       initial_simplex_corner=interface.local_minima_vals*1.8,
                                       initial_simplex_displacements=np.full(
                                           48, 0.1)


                                       #    # initial lengths scales for GP
                                       #    length_scale=[0.01],

                                       #    update_hyperparameters=True,  # whether noise level and lengths scales are updated

                                       #    default_bad_cost=10,  # default cost for bad run
                                       #    default_bad_uncertainty=1,  # default uncertainty for bad run
                                       #    learner_archive_filename='GP_learning_data',  # filename of gp archive
                                       #    learner_archive_file_type='txt',  # file type of archive

                                       #    # whether to wait for the GP to make predictions or not. Default True (do not wait)
                                       #    no_delay=True,

                                       #    # Training source options
                                       #    training_type='nelder_mead'  # training type can be random or nelder_mead

                                       #    #    filename for training from previous experiment
                                       #    gp_training_filename='GP_learning_data_2020-01-23_09-58.txt',
                                       #    gp_training_file_type='txt'  # training data file type

                                       )
    # To run M-LOOP and find the optimal parameters just use the controller method optimize
    controller.optimize()
    print('Best parameters found:')
    print(controller.best_params)
    # You can also run the default sets of visualizations for the controller with one command
    mlv.show_all_default_visualizations(controller)

    ####### Neural Net #####
    # neural_network = nn.NeuralNet(
    #     num_params=48,
    #     fit_hyperparameters=True
    # )
    # neural_network.init()
    # initial_net = neural_network._make_net(0)

    # neural_network.fit_neural_net(controller.out_params, controller.in_costs)
    # neural_network.start_opt()
    # print("Predicted Cost", neural_network.predict_cost(
    # interface.local_minima_vals))
    # The results of the optimization will be saved to files and can also be accessed as attributes of the controller.
    single_neural_net = SingleNeuralNet()

    )


# Ensures main is run when this code is run as a script
if __name__ == '__main__':
    main()
