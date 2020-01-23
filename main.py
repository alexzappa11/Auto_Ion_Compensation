import numpy as np
import scipy as sc
from sympy import *
import sys
import nidaqmx
import time
import subprocess as sub
import pandas as pd
import matplotlib.pyplot as plt
import random
from slot_configuration import *


################## Read CSV files####################
compExTol = pd.read_csv(
    'microwave_20121112_compExTol.csv', sep=',', header=None)
compEyTol = pd.read_csv(
    'microwave_20121112_compEyTol.csv', sep=',', header=None)
harmonic = pd.read_csv('microwave_20121112_harmonic.csv', sep=',', header=None)
uniform_Quad = pd.read_csv(
    'microwave_20121112_uniform_Quad.csv', sep=',', header=None)
BeamVxByPosition = pd.read_csv("BeamVxByPosition.csv", sep=',', header=None)

compExTol = np.array(compExTol.iloc[1:, :], dtype=float)
compEyTol = np.array(compEyTol.iloc[1:, :], dtype=float)
harmonic = np.array(harmonic.iloc[1:, :], dtype=float)
uniform_Quad = np.array(uniform_Quad.iloc[1:, :], dtype=float)
BeamVxByPosition = np.array(BeamVxByPosition.iloc[:, :], dtype=float)


def CSV_Read(position, array):
    '''reads voltage data from csv and return voltage corresponding to the position'''
    i = np.digitize(position, array[:, 0])
    m = (array[i, 1:]-array[i-1, 1:])/(array[i, 0]-array[i-1, 0])  # gradient
    y_int = array[i-1, 1:]  # y intercept
    # y = mx+c where x is the input position and y is the waveform array
    V_f = m*(position-array[i-1, 0])+y_int
    return V_f


def data_processing(parameter_input, output_vals, parameter_size, num_its):

    ########## Plotting Data ###########
    plt.plot(output_vals, label="Photon Count")  # plot photon count
    plt.legend()
    plt.show()
    parameter1 = np.array([parameter_input[i::parameter_size]
                           for i in range(0, parameter_size)])
    xdim = len(parameter1) + 3
    fig, axs = plt.subplots(xdim//4, 4)
    n = 0
    for j in range(4):
        for i in range(0, xdim//4):
            if n == len(parameter1):
                break
            axs[i][j].plot(parameter1[n], label=str(n+1))
            axs[i][j].legend(loc='lower left', frameon=False)
            n += 1
    plt.show()
##### Plot waveform over iterations #####
    waveform = []
    for i in range(0, len(parameter_input), parameter_size):
        waveform.append(parameter_input[i:i+parameter_size])
    waveform2 = np.asarray(waveform) - np.asarray(waveform)[0]
    waveform = np.asarray(waveform)

    plt.imshow(waveform2.T, cmap="hot")
    plt.colorbar()
    plt.xlabel("Iteration Number")
    plt.ylabel("Electrode")
    plt.title("$V_n-V_0$")
    plt.show()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    np.savetxt(r'Output_Data\parameters_each_iteration'+timestr+'.csv',
               waveform.T, delimiter=',')
    np.savetxt(r'Output_Data\output_each_iteration'+timestr+'.csv',
               output_vals, delimiter=',')


def ion_weight_function(ion_position, ion_weights):
    '''get photon count from the applied weights and specified position'''
    ion_voltage = get_Voltage(ion_weights, ion_position)
    write_voltage(ion_voltage)
    ion_photon_count = take_photon()
    return ion_photon_count


def ion_voltage_function(ion_position, ion_voltage):
    '''get the photon count from the applied voltage '''
    write_voltage(ion_voltage)
    ion_photon_count = take_photon()
    return ion_photon_count


def ion_position_function(ion_weights, ion_position):
    '''get photon count from the applied position and specified position'''
    ion_voltage = get_Voltage(weight_Params, ion_position[0])
    write_voltage(ion_voltage)
    ion_photon_count = take_photon()
    return ion_photon_count


def func_new_position_ion(old_position, new_position):
    '''get photon count from the new applied position '''
    ion_voltage = get_Voltage(weight_Params, new_position[0])
    write_voltage(ion_voltage)
    ion_photon_count = take_photon()
    return ion_photon_count


def func_new_weights_ion(old_params, new_params):
    '''function to create array of new photon counts for each new updated weight'''
    no_of_params = len(old_params)
    final_photon_count = np.full((no_of_params), 0)
    temp_params = old_params
    # for loop to get the change photon count after a new parameter is applied independently
    for i in range(0, no_of_params):
        old_params[i] = new_params[i]  # apply new parameter
        # get the photon count after new parameter is applied
        final_photon_count[i] = ion_weight_function(position, old_params)
        old_params = temp_params  # return to previous parameters
    return final_photon_count


def func_new_voltage_ion(old_voltage, new_voltage):
    '''function to create array of new photon counts for each new updated voltage'''
    no_of_params = len(old_voltage)
    final_photon_count = np.full((no_of_params), 0)
    temp_params = old_voltage
    # for loop to get the change in photon count after a new parameter is applied independently
    for i in range(0, no_of_params):
        old_voltage[i] = new_voltage[i]  # apply new parameter
        # get the photon count after new parameter is applied
        final_photon_count[i] = ion_voltage_function(position, old_voltage)
        old_voltage = temp_params  # return to previous parameters
    return final_photon_count


def rate_of_change(x_1, x_2, y_1, y_2):
    '''calculates the slope for multi-dimensional function by getting the change in photon count / change in dependant var (weight, position, voltage) '''
    epsilon = np.exp(-8)
    deltax = np.subtract(x_2, x_1)
    deltay = np.subtract(y_2, y_1)
    deltax = np.add(deltax, epsilon)  # prevent divide by zero
    gradient_list = np.divide(deltay, deltax)
    return gradient_list  # returns gradient for each dimension in a list


def get_Voltage(weight_Params, position):
    '''Get waveform for specific direction, apply the weights and return voltage array'''
    outV_compEx = CSV_Read(position, compExTol)
    outV_compEx = outV_compEx*weight_Params[0]
    outV_compEy = CSV_Read(position, compEyTol)
    outV_compEy = outV_compEy*weight_Params[1]
    outV_harmonic = CSV_Read(position, harmonic)
    outV_harmonic = outV_harmonic*weight_Params[2]
    outV_uniform_Quad = CSV_Read(position, uniform_Quad)
    outV_uniform_Quad = outV_uniform_Quad*weight_Params[3]
    # get the voltage for the mirror from position. Furthest length is 1086 (length of chip)
    Beam_Vx = CSV_Read(position, BeamVxByPosition) * \
        (position/1086) - 2.52  # Vx offset
    # sum all waveform files to return one waveform array
    final = outV_compEx + outV_compEy + outV_harmonic + outV_uniform_Quad
    final[0] = Beam_Vx  # controls the x axis of cooling laser
    return final/2  # doubling amplifier in the DAQ


def write_voltage(final_voltage):
    # ############# writes voltages to DAQ device for each slot given the waveform ###########
    # with nidaqmx.Task() as slot2, nidaqmx.Task() as slot3, nidaqmx.Task() as slot4, nidaqmx.Task() as slot5, nidaqmx.Task() as slot6, nidaqmx.Task() as slot7, nidaqmx.Task() as slot8, nidaqmx.Task() as slot9, nidaqmx.Task() as slot10, nidaqmx.Task() as slot11, nidaqmx.Task() as slot12, nidaqmx.Task() as slot13:
    #     # set the correct voltage to the port for each slot.
    #     voltage_slot_2 = np.array(
    #         [0, 0, final_voltage[16], final_voltage[12], 0, final_voltage[6], final_voltage[7], 0])
    #     voltage_slot_3 = np.array([final_voltage[13], final_voltage[17], 0,
    #                                0, 0, final_voltage[16], final_voltage[14], final_voltage[10]])
    #     voltage_slot_4 = np.array([final_voltage[8], final_voltage[0], 0, final_voltage[9],
    #                                final_voltage[11], final_voltage[15], final_voltage[17], 0])
    #     voltage_slot_5 = np.array([0, 0, final_voltage[19], final_voltage[23],
    #                                final_voltage[27], final_voltage[27], final_voltage[33], final_voltage[33]])
    #     voltage_slot_6 = np.array([final_voltage[37], final_voltage[41], 0,
    #                                0, 0, final_voltage[19], final_voltage[21], final_voltage[25]])
    #     voltage_slot_7 = np.array([final_voltage[29], 0, 0, final_voltage[31],
    #                                final_voltage[35], final_voltage[39], final_voltage[41], 0])
    #     voltage_slot_8 = np.array(
    #         [0, 0, final_voltage[45], final_voltage[5], 0, 0, 0, 0])
    #     voltage_slot_9 = np.array([final_voltage[4], final_voltage[44], 0,
    #                                0, 0, final_voltage[43], final_voltage[47], final_voltage[45]])
    #     voltage_slot_10 = np.array(
    #         [0, 0, 0, 0, final_voltage[44], final_voltage[46], final_voltage[42], 0])
    #     voltage_slot_11 = np.array([0, 0, final_voltage[40], final_voltage[36],
    #                                 final_voltage[32], final_voltage[32], final_voltage[26], final_voltage[26]])
    #     voltage_slot_12 = np.array([final_voltage[22], final_voltage[18],
    #                                 0, 0, 0, final_voltage[40], final_voltage[38], final_voltage[34]])
    #     voltage_slot_13 = np.array([final_voltage[30], 0, 0, final_voltage[28],
    #                                 final_voltage[24], final_voltage[20], final_voltage[18], 0])
    #     # open query to send voltages
    #     slot2.ao_channels.add_ao_voltage_chan(PXI1Slot2)
    #     slot3.ao_channels.add_ao_voltage_chan(PXI1Slot3)
    #     slot4.ao_channels.add_ao_voltage_chan(PXI1Slot4)
    #     slot5.ao_channels.add_ao_voltage_chan(PXI1Slot5)
    #     slot6.ao_channels.add_ao_voltage_chan(PXI1Slot6)
    #     slot7.ao_channels.add_ao_voltage_chan(PXI1Slot7)
    #     slot8.ao_channels.add_ao_voltage_chan(PXI1Slot8)
    #     slot9.ao_channels.add_ao_voltage_chan(PXI1Slot9)
    #     slot10.ao_channels.add_ao_voltage_chan(PXI1Slot10)
    #     slot11.ao_channels.add_ao_voltage_chan(PXI1Slot11)
    #     slot12.ao_channels.add_ao_voltage_chan(PXI1Slot12)
    #     slot13.ao_channels.add_ao_voltage_chan(PXI1Slot13)
    #     # apply voltages to DAQ writing to one slot per write
    #     slot2.write(voltage_slot_2)
    #     slot3.write(voltage_slot_3)
    #     slot4.write(voltage_slot_4)
    #     slot5.write(voltage_slot_5)
    #     slot6.write(voltage_slot_6)
    #     slot7.write(voltage_slot_7)
    #     slot8.write(voltage_slot_8)
    #     slot9.write(voltage_slot_9)
    #     slot10.write(voltage_slot_10)
    #     slot11.write(voltage_slot_11)
    #     slot12.write(voltage_slot_12)
    #     slot13.write(voltage_slot_13)
    #     # slot2.stop
    #     # slot3.stop
    #     # slot4.stop
    #     # slot5.stop
    #     # slot6.stop
    #     # slot7.stop
    #     # slot8.stop
    #     # slot9.stop
    #     # slot10.stop
    #     # slot11.stop
    #     # slot12.stop
    #     # slot13.stop
    return None


def take_photon():
    time.sleep(0.01)
    return -random.randint(2000, 5000)
    # channel = 5  # PMT channel for where the photons are being read from
    # #######   reads photon count #############
    # file = r'"C:\src\id800.exe"'
    # # -t time in s, -e exposure in ms, -c confidence bound
    # arguments2 = " -C -t 0.1 -e 50 -c 10"
    # command = file + arguments2
    # stream = sub.Popen(command,
    #                    stdout=sub.PIPE,
    #                    stderr=sub.PIPE, universal_newlines=True)
    # output = stream.communicate()  # save output of cmd to variable
    # photons = np.array(str(output).split(' ')[7:26])
    # photons = photons.astype(np.int)
    # # negative as the algorithm seeks to minimise the cost (cost being the photon count)
    # photons = -photons[(channel - 1)]
    # # photons = round(photons, -2)
    # return photons


def adam(function_new_output, function_output, function_read_ouput, parameters_to_optimise, fixed_parameter, param_range):
    ''' Adam estimation for photon count and weight compensation '''
    parameters_to_optimise_with_range = parameters_to_optimise[param_range[0]:param_range[1]]
    position = fixed_parameter
    no_of_params = len(parameters_to_optimise_with_range)
    iter_count = 0
    number_of_iterations = 11

    # keeps track of parameter evolution
    parameter_array = np.array([parameters_to_optimise_with_range])
    output_array = np.array([function_read_ouput()])

    print("Starting Adam for :")
    print("	x0_y0 = ", parameters_to_optimise_with_range,
          '\n with range: ', param_range)
    print("Number of iterations", number_of_iterations)

    alpha = np.full((no_of_params), 0.005)  # learning rate (and initial step)
    beta_1 = 0
    beta_2 = 0  # exponential decay rates for moment estimates
    epsilon = 1e-8
    theta_0 = parameters_to_optimise_with_range  # initialize the vector
    m_t = np.full((no_of_params), 0.0)
    v_t = np.full((no_of_params), 0.0)

    # add starting values and initial step, this can be anything random
    theta_0_initial_step = np.add(theta_0, alpha)

    ###### get initial gradient #########
    output_init = function_read_ouput()
    print("initial photon", output_init)
    output_2 = function_new_output(theta_0, theta_0_initial_step)
    g_t = rate_of_change(theta_0, theta_0_initial_step, output_init, output_2)
    g_t = np.divide(g_t, np.linalg.norm(g_t))

    # save photon count before the start of the interation
    output_init_const = output_init
    maxPhoton = output_init_const  # initialise max photon count
    # initialise max photon voltage config
    maxVoltage = parameter_array

    for i in range(number_of_iterations):
        iter_count = iter_count + 1
        m_t = beta_1*m_t + (1-beta_1)*g_t
        v_t = beta_2*v_t + (1-beta_2)*(np.power(g_t, 2))
        m_cap = m_t/(1-(beta_1**iter_count))
        v_cap = v_t/(1-(beta_2**iter_count))
        theta_0_prev = theta_0
        theta_0 = theta_0 - (np.multiply(alpha, m_cap)) / \
            (np.sqrt(v_cap)+epsilon)

        output_init = function_read_ouput()  # get photon count
        output_1 = np.full((no_of_params), output_init)

        parameters_to_optimise[param_range[0]:param_range[1]] = theta_0_prev
        # insert old values into full parameter list
        params_prev = parameters_to_optimise

        # print("old params", params_prev)

        parameters_to_optimise[param_range[0]:param_range[1]] = theta_0
        # insert new values into full parameter list
        params_final = parameters_to_optimise

        # print("new params", params_final)

        output_2 = function_new_output(params_prev, params_final)
        final_output = function_output(fixed_parameter, params_final)

        g_t = rate_of_change(theta_0_prev, theta_0, output_1,
                             output_2[param_range[0]:param_range[1]])
        g_t = g_t/np.linalg.norm(g_t)

        print("x, y, params: ", theta_0)
        print("final photon count: ", -final_output)
        print("Iteration #", iter_count)
        print("Gradient: ", g_t/np.linalg.norm(g_t))

        if final_output < maxPhoton:
            maxPhoton = final_output  # update max photon count and corresponding voltages
            maxVoltage = theta_0  # save voltage config for new photon max

        parameter_array = np.append(
            parameter_array, theta_0)  # save param array
        output_array = np.append(
            output_array, final_output)  # save output array

        # if statement to break optimisation if the photon count drops below a percentage of the starting photon count
        if (final_output/output_init_const) < 0.80:
            # apply voltage from max photon count
            exit_photon_count = function_output(fixed_parameter, maxVoltage)
            print("Loop terminated, final photon count:",
                  exit_photon_count)  # display final photon count
            # display final applied voltages
            print("Loop terminated, applied voltage:", maxVoltage)

            parameter_array = np.append(
                parameter_array, maxVoltage)  # save param array
            output_array = np.append(
                output_array, exit_photon_count)

            break  # break out of optimisation loop

    data_processing(parameter_array, output_array,
                    no_of_params, iter_count)  # plot data


# weight_Params = np.array([0.22, 0.4, 1.4, 0.3])  # starting weight params
# position1 = [0]  # initial position of the ion on the chip
# position = 865.4
# final_voltage = get_Voltage(weight_Params, position)
# print(final_voltage)


# write_voltage(final_voltage)

# adam(func_new_weights_ion, ion_weight_function, take_photon, weight_Params, position)
# adam(func_new_voltage_ion, ion_voltage_function,
#      take_photon, final_voltage, position, [0, 48])
# adam(func_new_position_ion, ion_position_function, take_photon, position1, weight_Params)
