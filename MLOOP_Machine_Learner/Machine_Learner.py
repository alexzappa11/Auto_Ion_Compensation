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
import pandas as pd

default_learner_archive_filename = 'learner_archive'
default_learner_archive_file_type = 'txt'


class CustomInterface(mli.Interface):

    def __init__(self, simulation_function, experiment_function):

        super(CustomInterface, self).__init__()

        self.simulationFunction = simulation_function
        self.min_parameters = simulation_function.get_min_parameters
        self.min = simulation_function.get_min

        self.experiment_function = experiment_function

        # Get the voltage from the Adam Optimiser program and store it as the starting voltage
        # get_Voltage(weight_Params, position)
        final_voltage = pd.read_csv(
            '../Adam/VoltagePlaceHolder.csv', sep=',', header=None)
        self.starting_voltage = np.array(final_voltage.iloc[:, 0], dtype=float)
        print("Optimising from", self.starting_voltage)

    # this method is called whenever M-LOOP wants to run an experiment
    def get_next_cost_dict(self, params_dict):

        # Get parameters from the provided dictionary
        params = params_dict['params']

        # Get cost from the simulation function
        # cost = self.simulationFunction.evaluate_at_params(params)

        # Get cost from the Ion Function
        cost = self.experiment_function.Ion_function(params)

        print("Cost: ", cost)  # !!!!Param length must be 48!!!

        # Define the uncertainty
        uncer = 0.20*3000

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
    interface = CustomInterface(mvq.quadratic(
        48), itf.Ion_Trap_control(istest=True))  # initialise the interface

    NeuralNetLearningController = mlc.create_controller(
        interface,
        controller_type='neural_net',
        training_type='differential_evolution',
        # num_training_runs=100,
        max_num_runs=10,
        no_delay=True,  # If True, there is never any delay between a returned cost and the next parameters to run for the experiment. In practice, this means if the machine learning learner has not prepared the next parameters in time the learner defined by the initial training source is used instead. If false, the controller will wait for the machine learning learner to predict the next parameters and there may be a delay between runs.
        num_params=48,
        min_boundary=interface.starting_voltage - 0.5,
        max_boundary=interface.starting_voltage + 0.5,
        trust_region=0.01,  # maximum move % of cost from best params
        cost_has_noise=True,
        fit_hyperparameters=False,
        visualizations=True,
        # first_params=interface.min_parameters()*1.2, #use this for simulation
        first_params=interface.starting_voltage
    )

    NeuralNetLearningController.optimize()
    finalCost = interface.experiment_function.Ion_function(
        NeuralNetLearningController.best_params)  # Apply the best params form the run
    print("Best parameters Applied with a final cost of: ", finalCost)

    mlv.show_all_default_visualizations(NeuralNetLearningController)

    # TODO Print distance actual is away from approximation scaled difference in the simulation function
    # TODO Safety Net
    # TODO Load in the trained Network and have it find the optimal point (this will change though if the electric field in the trap changes)
    # TODO Live graph of costs

    # NeuralNetLearningController.print_results()
    # NeuralNetLearningController.ml_learner.neural_net[0]

    # For simulation Testing
    # print("Actual minimum value is: ", interface.min())
    # print("Actual Parameters at minimum value: ", interface.min_parameters())


if __name__ == '__main__':
    main()
