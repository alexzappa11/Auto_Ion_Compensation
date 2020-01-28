# Imports for python 2 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

# Imports for M-LOOP
import mloop.interfaces as mli
import mloop.controllers as mlc
import mloop.visualizations as mlv
import mloop.neuralnet as nn


# Other imports
import numpy as np
import time
import multi_var_quadratic as mvq
from sklearn import *
import tensorflow as tf
import _pickle as pickle
import time
# Declare your custom class that inherets from the Interface class


class CustomInterface(mli.Interface):

    def __init__(self, simulation_function):

        super(CustomInterface, self).__init__()

        self.simulationFunction = simulation_function
        self.min_parameters = simulation_function.get_min_parameters
        self.min = simulation_function.get_min_parameters

    # You must include the get_next_cost_dict method in your class
    # this method is called whenever M-LOOP wants to run an experiment
    def get_next_cost_dict(self, params_dict):

        # Get parameters from the provided dictionary
        params = params_dict['params']

        # Here you can include the code to run your experiment given a particular set of parameters
        cost = self.simulationFunction.evaluate_at_params(params)
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

    def min_parameters(self):
        return self.self.min_parameters

    def get_min(self):
        return self.min


def main():
    # M-LOOP can be run with three commands

    # First create your interface
    interface = CustomInterface(mvq.quadratic(48))
    # Next create the controller, provide it with your controller and any options you want to set
    controller = mlc.create_controller(interface,
                                       #    controller_type="gaussian_process",
                                       #    controller_type='differential_evolution',
                                       #   controller_type='nelder_mead',
                                    #    controller_type='neural_net',
                                       #    controller_type='random',
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
                                       #    training_type='differential_evolution',

                                       ###### Nelder Mead ####
                                       #    # initial corner of the simplex
                                       #    initial_simplex_corner=interface.min_parameters()*1.8,
                                       #    initial_simplex_displacements=np.full(
                                       #        48, 0.1)


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
    # controller.optimize()
    # controller.run()
    # print('Best parameters found:')
    # print(controller.best_params)
    # You can also run the default sets of visualizations for the controller with one command

    # mlv.show_all_default_visualizations(controller)

#     ####### Neural Net #####
#     neural_network = nn.NeuralNet(
#         num_params=48,
#         fit_hyperparameters=False
#     )
#     neural_network.init()
#     initial_net = neural_network._make_net(1)
#     neural_network.fit_neural_net(controller.out_params, controller.in_costs)

#     print("Predicted Cost", neural_network.predict_cost(
#         interface.min_parameters()))


# # #### Save Neural Net to file #######
#     timestr = time.strftime("%Y%m%d-%H%M%S")
#     saved_net = neural_network.save()
#     with open('savedNet'+timestr+'.txt', 'wb') as handle:
#         pickle.dump(saved_net, handle)
#     with open('savedNetReadable'+timestr+'.txt', 'w') as f:
#         print(saved_net, file=f)


# ## Load Neural Net from file ###
#     with open('savedNet.txt', 'rb') as handle:
#         saved_net = pickle.loads(handle.read())

#     #### Load Previous Network ####
#     neural_network = nn.NeuralNet(
#         num_params=48,
#         fit_hyperparameters=False
#     )

#     neural_network.load(saved_net)  # load neural net

#     print("Predicted Cost", neural_network.predict_cost(
#         interface.min_parameters()))


# Ensures main is run when this code is run as a script
if __name__ == '__main__':
    main()
