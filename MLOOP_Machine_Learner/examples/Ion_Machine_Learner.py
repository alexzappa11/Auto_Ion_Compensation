# Imports for python 2 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

# Imports for M-LOOP
import mloop.interfaces as mli
import mloop.controllers as mlc
import mloop.visualizations as mlv
import mloop.neuralnet as nn
import mloop.learners as mll


# Other imports
import numpy as np
import time
import multi_var_quadratic as mvq
from sklearn import *
import tensorflow as tf
import _pickle as pickle
import time


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
    interface = CustomInterface(mvq.quadratic(10))

    NeuralNetLearningController = mlc.create_controller(
        interface,
        controller_type='differential_evolution',
        num_training_runs=50,
        max_num_runs=50,
        no_delay=False,
        num_params=10,
        min_boundary=np.full(10, -10),
        max_boundary=np.full(10, 10),
        trust_region=0.01,  # maximum move % from best params,
    )
    NeuralNetLearningController.optimize()

    learner = mll.NeuralNetLearner(
        num_params=10,
        min_boundary=np.full(10, -10),
        max_boundary=np.full(10, 10),
        # learner_archive_filename=default_learner_archive_filename,
        # learner_archive_file_type=default_learner_archive_file_type,
        start_datetime=None,
        trust_region=0.01,
        default_bad_cost=None,
        default_bad_uncertainty=None,
        nn_training_filename=None,
        nn_training_file_type='txt',
        minimum_uncertainty=1e-8,
        predict_global_minima_at_end=True,
    )
    learner.create_neural_net()
    learner.all_params = NeuralNetLearningController.out_params
    learner.all_costs = NeuralNetLearningController.in_costs
    learner.run()
    learner.find_global_minima()
    print(learner.predicted_best_cost)


if __name__ == '__main__':
    main()
