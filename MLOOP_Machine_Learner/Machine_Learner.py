# Imports for python 2 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

# # Imports for M-LOOP
import mloop.interfaces as mli
import mloop.controllers as mlc
import mloop.visualizations as mlv
import mloop.neuralnet as nn
import mloop.learners as mll


# Other imports
import numpy as np
import time
# imports the quadratic simulator
import Experiment_Controllers.Quadractic_Simulator as mvq
# Imports the function for the ion trap
import Experiment_Controllers.Ion_Trap_Function as itf
from sklearn import *
import tensorflow as tf
import _pickle as pickle
import time

default_learner_archive_filename = 'learner_archive'
default_learner_archive_file_type = 'txt'


class CustomInterface(mli.Interface):

    def __init__(self, simulation_function, experiment_function):

        super(CustomInterface, self).__init__()

        self.simulationFunction = simulation_function
        self.min_parameters = simulation_function.get_min_parameters
        self.min = simulation_function.get_min

        self.experiment_function = experiment_function

    # this method is called whenever M-LOOP wants to run an experiment
    def get_next_cost_dict(self, params_dict):

        # Get parameters from the provided dictionary
        params = params_dict['params']

        # Get cost from the simulation function
        cost = self.simulationFunction.evaluate_at_params(params)

        # Get cost from the Ion Function
        # cost = self.experiment_function.Evaluate_at_voltage(
        #     params)  # !!!!Param length must be 48!!!

        # Define the uncertainty
        uncer = 0

        # The evaluation will always be a success
        bad = False

        # The cost, uncertainty and bad boolean must all be returned as a dictionary
        # You can include other variables you want to record as well if you want
        cost_dict = {'cost': cost, 'uncer': uncer, 'bad': bad}
        return cost_dict

    def min_parameters(self):
        return self.self.min_parameters

    def get_min(self):
        return self.min


def main():
    interface = CustomInterface(mvq.quadratic(48), itf.Ion_Trap_control())

    NeuralNetLearningController = mlc.create_controller(
        interface,
        controller_type='neural_net',
        training_type='differential_evolution',
        # num_training_runs=100,
        max_num_runs=3000,
        no_delay=True,  # If True, there is never any delay between a returned cost and the next parameters to run for the experiment. In practice, this means if the machine learning learner has not prepared the next parameters in time the learner defined by the initial training source is used instead. If false, the controller will wait for the machine learning learner to predict the next parameters and there may be a delay between runs.
        num_params=48,
        min_boundary=np.full(48, -10),
        max_boundary=np.full(48, 10),
        trust_region=0.01,  # maximum move % from best params,
        fit_hyperparameters=False,
        visualizations=True,
        # first_params=interface.min_parameters()*1.2,
        first_params=np.full(48, 0)


    )
    differential_evolution = mlc.create_controller(
        interface,
        controller_type='differential_evolution',
        # training_type='differential_evolution',
        # num_training_runs=10,
        max_num_runs=126,
        no_delay=True,
        num_params=48,
        min_boundary=np.full(48, -3),
        max_boundary=np.full(48, 3),
        trust_region=0.1,  # maximum move % from best params,
        fit_hyperparameters=False,
        visualizations=True

    )
    # differential_evolution.optimize()
    NeuralNetLearningController.optimize()
    mlv.show_all_default_visualizations(NeuralNetLearningController)

    # print distance actual is away from approximator scaled difference in the random function

    # NeuralNetLearningController.print_results()
    # NeuralNetLearningController.ml_learner.neural_net[0]

    print("Actual minimum value is: ", interface.min())
    print("Actual Parameters at minimum value: ", interface.min_parameters())

    # learner = mll.NeuralNetLearner(
    #     num_params=48,
    #     min_boundary=np.full(48, -10),
    #     max_boundary=np.full(48, 10),
    #     learner_archive_filename=default_learner_archive_filename,
    #     learner_archive_file_type=default_learner_archive_file_type,
    #     start_datetime=None,
    #     trust_region=0.01,
    #     default_bad_cost=None,
    #     default_bad_uncertainty=None,
    #     nn_training_filename=None,
    #     nn_training_file_type='txt',
    #     minimum_uncertainty=1e-8,
    #     predict_global_minima_at_end=True,
    # )

    # NeuralNetLearningController.optimize()
    # NeuralNetLearningController.print_results()
    # print("minimum value is: ", interface.min())
    # print("Parameters at minimum value: ", interface.min_parameters())
    # learner.all_params = NeuralNetLearningController.out_params
    # learner.all_costs = NeuralNetLearningController.in_costs
    # learner.create_neural_net()
    # learner.run()
    # learner.find_global_minima()
    # print(learner.predicted_best_cost)


if __name__ == '__main__':
    main()
