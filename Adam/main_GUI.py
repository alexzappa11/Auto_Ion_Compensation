from tkinter import *
# from main import *
import sys
import numpy as np
import scipy as sc
from sympy import *
import nidaqmx
import time
import subprocess as sub
import pandas as pd
import matplotlib.pyplot as plt
import random
from DAQ_slot_configuration import *
from matplotlib import interactive
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

window = Tk()

# weight_Params = np.array([0.2, 0.5, 1, 0.8])  # starting weight params
# initial position of the ion on the chip


#######################################################################


################## Read CSV files####################
compExTol = pd.read_csv(
    'Waveform_files/microwave_20121112_compExTol.csv', sep=',', header=None)
compEyTol = pd.read_csv(
    'Waveform_files/microwave_20121112_compEyTol.csv', sep=',', header=None)
harmonic = pd.read_csv(
    'Waveform_files/microwave_20121112_harmonic.csv', sep=',', header=None)
uniform_Quad = pd.read_csv(
    'Waveform_files/microwave_20121112_uniform_Quad.csv', sep=',', header=None)
BeamVxByPosition = pd.read_csv(
    "Waveform_files/BeamVxByPosition.csv", sep=',', header=None)
CompExByPosition = pd.read_csv(
    "Waveform_files/CompExByPosition.csv", sep=',', header=None)
CompEyByPosition = pd.read_csv(
    "Waveform_files/CompEyByPosition.csv", sep=',', header=None)

CompExByPosition = np.array(CompExByPosition.iloc[:, :], dtype=float)
CompEyByPosition = np.array(CompEyByPosition.iloc[:, :], dtype=float)
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
    if graphOn.get():
        f1 = plt.figure(1)
        plt.plot(output_vals, label="Photon Count")  # plot photon count
        plt.legend()
        interactive(True)
        f1.show()
##### Plot waveform over iterations #####
    f2 = plt.figure(2)
    waveform = []
    for i in range(0, len(parameter_input), parameter_size):
        waveform.append(parameter_input[i:i+parameter_size])
    waveform2 = np.asarray(waveform) - np.asarray(waveform)[0]
    waveform = np.asarray(waveform)
    if graphOn.get():
        plt.imshow(waveform2.T, cmap="hot")
        plt.colorbar()
        plt.xlabel("Iteration Number")
        plt.ylabel("Electrode")
        plt.title("$V_n-V_0$")
        f2.show()
### plot voltages ##
    parameter1 = np.array([parameter_input[i::parameter_size]
                           for i in range(0, parameter_size)])
    parameter1 = np.around(parameter1, decimals=4)
    xdim = len(parameter1) + 3
    if graphOn.get():
        f3 = plt.figure(3)
        fig, axs = plt.subplots(xdim//4, 4)
        n = 0
        for j in range(4):
            for i in range(0, xdim//4):
                if n == len(parameter1):
                    break
                axs[i][j].plot(parameter1[n], label=str(n+1))
                axs[i][j].legend(loc='lower left', frameon=False)
                n += 1
        f3.show()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    np.savetxt(r'Output_Data\parameters_each_iteration'+timestr+'.csv',
               waveform.T, delimiter=',')
    np.savetxt(r'Output_Data\output_each_iteration'+timestr+'.csv',
               output_vals, delimiter=',')
    np.savetxt(r'VoltagePlaceHolder.csv',
               waveform[-1].T, delimiter=',')


def ion_weight_function(ion_position, ion_weights):
    '''get photon count from the applied weights and specified position'''
    ion_voltage = get_Voltage(ion_weights, ion_position)
    write_voltage(ion_voltage)
    ion_photon_count = take_photon()
    return ion_photon_count


def ion_voltage_function(ion_voltage):
    '''get the photon count from the applied voltage '''
    write_voltage(ion_voltage)
    ion_photon_count = take_photon()
    return ion_photon_count


# def ion_position_function(ion_weights, ion_position):
#     '''get photon count from the applied position and specified position'''
#     ion_voltage = get_Voltage(weight_Params, ion_position[0])
#     write_voltage(ion_voltage)
#     ion_photon_count = take_photon()
#     return ion_photon_count


# def func_new_position_ion(old_position, new_position):
#     '''get photon count from the new applied position '''
#     ion_voltage = get_Voltage(weight_Params, new_position[0])
#     write_voltage(ion_voltage)
#     ion_photon_count = take_photon()
#     return ion_photon_count


# def func_new_weights_ion(old_params, new_params):
#     '''function to create array of new photon counts for each new updated weight'''

#     position = float(input_goto_pos.get())
#     # weight_Params = np.array([float(input_ex.get()), float(
#     #     input_ey.get()), float(input_harm.get()), float(input_uni.get())])

#     no_of_params = len(old_params)
#     final_photon_count = np.full((no_of_params), 0)
#     temp_params = old_params
#     # for loop to get the change photon count after a new parameter is applied independently
#     for i in range(0, no_of_params):
#         old_params[i] = new_params[i]  # apply new parameter
#         # get the photon count after new parameter is applied
#         final_photon_count[i] = ion_weight_function(position, old_params)
#         old_params = temp_params  # return to previous parameters
#     return final_photon_count

def Ion_grad(x_prev, x):
    """ gradient function for the Ion given the previous and new values for the input parameters """
    dim = len(x_prev)  # get dimension of parameter space

    y_prev = take_photon()  # takes current photon before new weights
    y_prev = np.full(dim, y_prev)
    y = np.full(dim, 0)

    temp_x = x_prev
    # for loop to get y after a new parameter is applied independently
    for i in range(0, dim):
        x_prev[i] = x[i]  # apply new parameter
        # get the photon count after new parameter is applied
        y[i] = ion_voltage_function(x_prev)
        x_prev = temp_x  # return to previous parameters

    ## Get Gradients ###
    epsilon = np.exp(-8)
    deltax = np.subtract(x, x_prev)
    deltay = np.subtract(y, y_prev)
    deltax = np.add(deltax, epsilon)  # prevent divide by zero
    gradient_list = np.divide(deltay, deltax)
    return gradient_list  # returns gradient for each dimension in a list


def func_new_voltage_ion(old_voltage, new_voltage):
    '''function to create array of new photon counts for each new updated voltage'''

    position = float(input_goto_pos.get())
    weight_Params = np.array([float(input_ex.get()), float(
        input_ey.get()), float(input_harm.get()), float(input_uni.get())])

    no_of_params = len(old_voltage)
    new_photon_count = np.full((no_of_params), 0)
    temp_params = old_voltage
    # for loop to get the change in photon count after a new parameter is applied independently
    for i in range(0, no_of_params):
        old_voltage[i] = new_voltage[i]  # apply new parameter
        # get the photon count after new parameter is applied
        new_photon_count[i] = ion_voltage_function(old_voltage)
        old_voltage = temp_params  # return to previous parameters
    return new_photon_count


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

    ex_var.set(str(np.round(weight_Params[0], 2)))
    ey_var.set(str(np.round(weight_Params[1], 2)))
    harm_var.set(str(weight_Params[2]))
    uni_var.set(str(weight_Params[3]))

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

    ############# writes voltages to DAQ device for each slot given the waveform ###########
    with nidaqmx.Task() as slot2, nidaqmx.Task() as slot3, nidaqmx.Task() as slot4, nidaqmx.Task() as slot5, nidaqmx.Task() as slot6, nidaqmx.Task() as slot7, nidaqmx.Task() as slot8, nidaqmx.Task() as slot9, nidaqmx.Task() as slot10, nidaqmx.Task() as slot11, nidaqmx.Task() as slot12, nidaqmx.Task() as slot13:
        # set the correct voltage to the port for each slot.
        voltage_slot_2 = np.array(
            [0, 0, final_voltage[16], final_voltage[12], 0, final_voltage[6], final_voltage[7], 0])
        voltage_slot_3 = np.array([final_voltage[13], final_voltage[17], 0,
                                   0, 0, final_voltage[16], final_voltage[14], final_voltage[10]])
        voltage_slot_4 = np.array([final_voltage[8], final_voltage[0], 0, final_voltage[9],
                                   final_voltage[11], final_voltage[15], final_voltage[17], 0])
        voltage_slot_5 = np.array([0, 0, final_voltage[19], final_voltage[23],
                                   final_voltage[27], final_voltage[27], final_voltage[33], final_voltage[33]])
        voltage_slot_6 = np.array([final_voltage[37], final_voltage[41], 0,
                                   0, 0, final_voltage[19], final_voltage[21], final_voltage[25]])
        voltage_slot_7 = np.array([final_voltage[29], 0, 0, final_voltage[31],
                                   final_voltage[35], final_voltage[39], final_voltage[41], 0])
        voltage_slot_8 = np.array(
            [0, 0, final_voltage[45], final_voltage[5], 0, 0, 0, 0])
        voltage_slot_9 = np.array([final_voltage[4], final_voltage[44], 0,
                                   0, 0, final_voltage[43], final_voltage[47], final_voltage[45]])
        voltage_slot_10 = np.array(
            [0, 0, 0, 0, final_voltage[44], final_voltage[46], final_voltage[42], 0])
        voltage_slot_11 = np.array([0, 0, final_voltage[40], final_voltage[36],
                                    final_voltage[32], final_voltage[32], final_voltage[26], final_voltage[26]])
        voltage_slot_12 = np.array([final_voltage[22], final_voltage[18],
                                    0, 0, 0, final_voltage[40], final_voltage[38], final_voltage[34]])
        voltage_slot_13 = np.array([final_voltage[30], 0, 0, final_voltage[28],
                                    final_voltage[24], final_voltage[20], final_voltage[18], 0])
        # open query to send voltages
        slot2.ao_channels.add_ao_voltage_chan(PXI1Slot2)
        slot3.ao_channels.add_ao_voltage_chan(PXI1Slot3)
        slot4.ao_channels.add_ao_voltage_chan(PXI1Slot4)
        slot5.ao_channels.add_ao_voltage_chan(PXI1Slot5)
        slot6.ao_channels.add_ao_voltage_chan(PXI1Slot6)
        slot7.ao_channels.add_ao_voltage_chan(PXI1Slot7)
        slot8.ao_channels.add_ao_voltage_chan(PXI1Slot8)
        slot9.ao_channels.add_ao_voltage_chan(PXI1Slot9)
        slot10.ao_channels.add_ao_voltage_chan(PXI1Slot10)
        slot11.ao_channels.add_ao_voltage_chan(PXI1Slot11)
        slot12.ao_channels.add_ao_voltage_chan(PXI1Slot12)
        slot13.ao_channels.add_ao_voltage_chan(PXI1Slot13)
        # apply voltages to DAQ writing to one slot per write
        slot2.write(voltage_slot_2)
        slot3.write(voltage_slot_3)
        slot4.write(voltage_slot_4)
        slot5.write(voltage_slot_5)
        slot6.write(voltage_slot_6)
        slot7.write(voltage_slot_7)
        slot8.write(voltage_slot_8)
        slot9.write(voltage_slot_9)
        slot10.write(voltage_slot_10)
        slot11.write(voltage_slot_11)
        slot12.write(voltage_slot_12)
        slot13.write(voltage_slot_13)
        # slot2.stop
        # slot3.stop
        # slot4.stop
        # slot5.stop
        # slot6.stop
        # slot7.stop
        # slot8.stop
        # slot9.stop
        # slot10.stop
        # slot11.stop
        # slot12.stop
        # slot13.stop
        # print(final_voltage)
    return None


def take_photon():

    # time.sleep(0.01)
    # return -random.randint(2000, 5000)
    channel = 5  # PMT channel for where the photons are being read from
    #######   reads photon count #############
    file = r'"C:\src\id800.exe"'
    # -t time in s, -e exposure in ms, -c confidence bound
    arguments2 = " -C -t 0.1 -e 50 -c 10"
    command = file + arguments2
    stream = sub.Popen(command,
                       stdout=sub.PIPE,
                       stderr=sub.PIPE, universal_newlines=True)
    output = stream.communicate()  # save output of cmd to variable
    photons = np.array(str(output).split(' ')[7:26])
    photons = photons.astype(np.int)
    # negative as the algorithm seeks to minimise the cost (cost being the photon count)
    photons = -photons[(channel - 1)]
    # photons = round(photons, -2)
    return photons


def shuttle_ion(current_position, target_position, weight_Params, speed):
    target_position = np.around(target_position, decimals=1)

    if target_position > current_position:
        while current_position < target_position and running:
            current_position = current_position + \
                float(input_time_between_step.get())
            current_position = np.around(current_position, decimals=1)
            weight_Params[0] = CSV_Read(current_position, CompExByPosition)
            weight_Params[1] = CSV_Read(current_position, CompEyByPosition)
            # print(weight_Params)
            final_Voltage = get_Voltage(weight_Params, current_position)
            write_voltage(final_Voltage)
            input_current_pos.delete(0, END)
            input_current_pos.insert(0, current_position)
            window.update()
            time.sleep(0.01)  # adjust this for shuttling speed
            np.savetxt(r'VoltagePlaceHolder.csv',
                       final_Voltage.T, delimiter=',')
            final_Voltage = np.around(final_Voltage, decimals=4)
            list_voltages.delete(0, 100)
            for i in range(len(final_Voltage)):
                list_voltages.insert(i, "V" + str(i) +
                                     ": " + str(final_Voltage[i]))  # update the list
        while current_position > target_position and running:
            current_position = current_position - 0.1
            current_position = np.around(current_position, decimals=1)
            weight_Params[0] = CSV_Read(current_position, CompExByPosition)
            weight_Params[1] = CSV_Read(current_position, CompEyByPosition)
            # print(weight_Params)
            final_Voltage = get_Voltage(weight_Params, current_position)
            write_voltage(final_Voltage)
            input_current_pos.delete(0, END)
            input_current_pos.insert(0, current_position)
            window.update()
            time.sleep(0.01)  # adjust this for shuttling speed
            np.savetxt(r'VoltagePlaceHolder.csv',
                       final_Voltage.T, delimiter=',')
            final_Voltage = np.around(final_Voltage, decimals=4)
            list_voltages.delete(0, 100)
            for i in range(len(final_Voltage)):
                list_voltages.insert(i, "V" + str(i) +
                                     ": " + str(final_Voltage[i]))  # update the list

    elif target_position < current_position:
        while current_position > target_position and running:
            current_position = current_position - \
                float(input_time_between_step.get())  # stepsize
            current_position = np.around(current_position, decimals=1)

            weight_Params[0] = CSV_Read(current_position, CompExByPosition)
            weight_Params[1] = CSV_Read(current_position, CompEyByPosition)
            # print(weight_Params)
            final_Voltage = get_Voltage(weight_Params, current_position)
            write_voltage(final_Voltage)
            input_current_pos.delete(0, END)
            input_current_pos.insert(0, current_position)
            window.update()
            time.sleep(0.01)
            np.savetxt(r'VoltagePlaceHolder.csv',
                       final_Voltage.T, delimiter=',')
            final_Voltage = np.around(final_Voltage, decimals=4)
            list_voltages.delete(0, 100)
            for i in range(len(final_Voltage)):
                list_voltages.insert(i, "V" + str(i) +
                                     ": " + str(final_Voltage[i]))  # update the list
        while current_position < target_position and running:
            current_position = current_position + 0.1
            current_position = np.around(current_position, decimals=1)
            weight_Params[0] = CSV_Read(current_position, CompExByPosition)
            weight_Params[1] = CSV_Read(current_position, CompEyByPosition)
            # print(weight_Params)
            final_Voltage = get_Voltage(weight_Params, current_position)
            write_voltage(final_Voltage)
            input_current_pos.delete(0, END)
            input_current_pos.insert(0, current_position)
            window.update()
            time.sleep(0.01)  # adjust this for shuttling speed
            np.savetxt(r'VoltagePlaceHolder.csv',
                       final_Voltage.T, delimiter=',')
            final_Voltage = np.around(final_Voltage, decimals=4)
            list_voltages.delete(0, 100)
            for i in range(len(final_Voltage)):
                list_voltages.insert(i, "V" + str(i) +
                                     ": " + str(final_Voltage[i]))  # update the list
    else:
        None
    a.cla()
    a2.cla()
    a.plot(final_Voltage[1::2], color='blue', linestyle='-', marker='o')
    a2.plot(final_Voltage[2::2], color='blue', linestyle='-', marker='o')
    a.set_title("Waveform: 1, 3, 5...", fontsize=14)
    a2.set_title("Waveform:2, 4, 6...", fontsize=14)
    a.set_ylabel("Voltage (V)", fontsize=14)
    a2.set_ylabel("Voltage (V)", fontsize=14)
    a2.set_xlabel("Electrode", fontsize=14)
    fig.tight_layout()
    canvas.draw()
    return final_Voltage


def update_plots(photonCounts, waveform_in, parameter_length, waveform_list):
    ###### update waveform plot ######
    waveform1 = [waveform_list[i:i+parameter_length]
                 for i in range(0, len(waveform_list), parameter_length)]
    # print(waveform1)
    waveform3 = np.asarray(waveform1) - np.asarray(waveform1)[0]
    waveform1 = np.asarray(waveform1)
    a.cla()
    a2.cla()
    c.cla()
    a.plot(waveform_in[1::2], color='blue', linestyle='-', marker='o')
    a2.plot(waveform_in[2::2], color='blue', linestyle='-', marker='o')
    # print(waveform3)
    c.imshow(waveform3.T, cmap="hot")
    # c.colorbar()
    c.set_xlabel("Iteration Number")
    c.set_ylabel("Electrode")
    c.set_title("$V_n-V_0$")
    a.set_title("Waveform: 1, 3, 5...", fontsize=14)
    a2.set_title("Waveform:2, 4, 6...", fontsize=14)
    a.set_ylabel("Voltage (V)", fontsize=14)
    a2.set_ylabel("Voltage (V)", fontsize=14)
    a2.set_xlabel("Electrode", fontsize=14)

    ###### Update Photon Plot #######
    b.cla()
    b.plot(photonCounts * 20, color='blue',  linestyle='-', marker='o')
    b.set_title("Photon Count", fontsize=16)
    b.set_ylabel("Counts/Second", fontsize=14)
    b.set_xlabel("Iteration #", fontsize=14)
    fig.tight_layout()
    canvas.draw()


def adam(function_new_output, function_output, function_read_ouput, parameters_to_optimise, fixed_parameter, param_range):
    ''' Adam estimation for photon count and weight compensation '''

    parameters_to_optimise_with_range = parameters_to_optimise[param_range[0]:param_range[1]]

    no_of_params = len(parameters_to_optimise_with_range)
    iter_count = 0
    try:
        number_of_iterations = int(input_stepNum.get())
    except:
        popupmsg("Invalid Number of Iterations, enter an Integer")

    # keeps track of parameter evolution
    parameter_array = np.array([parameters_to_optimise_with_range])
    output_array = np.array([function_read_ouput()])

    try:
        # learning rate (and initial step) 0.005
        alpha = np.full((no_of_params), float(input_stepsize.get()))
        beta_1 = float(input_decay.get())
        # exponential decay rates for moment estimates
        beta_2 = float(input_expDecay.get())
    except:
        popupmsg("Invalid Parameter Input")
    epsilon = 1e-8
    theta_0 = parameters_to_optimise_with_range  # initialize the vector
    m_t = np.full((no_of_params), 0.0)
    v_t = np.full((no_of_params), 0.0)

    # add starting values and initial step, this can be anything random
    theta_0_initial_step = np.add(theta_0, alpha)

    ###### get initial gradient #########
    output_init = function_read_ouput()
    # print("initial photon", output_init)
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

        if final_output < maxPhoton:
            maxPhoton = final_output  # update max photon count and corresponding voltages
            maxVoltage = theta_0  # save voltage config for new photon max

        parameter_array = np.append(
            parameter_array, theta_0)  # save param array
        output_array = np.append(
            output_array, final_output)  # save output array

        ##update plots ##

        update_plots(-output_array, theta_0, no_of_params, parameter_array)

        theta_r = np.around(theta_0, decimals=4)

        list_voltages.delete(0, 100)
        for i in range(len(theta_r)):
            list_voltages.insert(i, "V" + str(i) +
                                 ": " + str(theta_r[i]))  # update the list
        window.update()

        np.savetxt(r'Output_Data\parameters_each_iteration_bin.csv',
                   parameter_array, delimiter=',')
        np.savetxt(r'Output_Data\output_each_iteration_bin.csv',
                   output_array, delimiter=',')

        # if statement to break optimisation if the photon count drops below a percentage of the starting photon count
        try:
            if (final_output/output_init_const) < float(input_threshold.get()):
                break  # break out of optimisation loop
            elif not(running):
                break

        except:
            popupmsg("Invalid threshold")


<<<<<<< HEAD


=======
>>>>>>> Gradient Function
####### Check if final photon count is less than the max over the iteration. If it is less, it applies the voltage configuration corresponding to the highest photon count ######
    if final_output > maxPhoton:
        parameter_array = np.append(
            parameter_array, maxVoltage)  # save param array
        exit_photon_count = function_output(fixed_parameter, maxVoltage)

        maxVoltage = np.around(maxVoltage, decimals=4)
        list_voltages.delete(0, 100)
        for i in range(len(maxVoltage)):
            list_voltages.insert(i, "V" + str(i) +
                                 ": " + str(maxVoltage[i]))  # update the list with max voltage
        np.savetxt(r'VoltagePlaceHolder.csv',
                   maxVoltage.T, delimiter=',')
        output_array = np.append(
            output_array, exit_photon_count)  # save output array
        # print("max", maxVoltage)
    else:
        np.savetxt(r'VoltagePlaceHolder.csv',
                   theta_0.T, delimiter=',')

    data_processing(parameter_array, output_array,
                    no_of_params, iter_count)  # plot data


######################################################################################################################

def startup():

    input_goto_pos.insert(0, "865.4")  # default starting position
    input_ex.insert(0, "0.22")  # default weights
    input_ey.insert(0, "0.4")
    input_harm.insert(0, "1.4")
    input_uni.insert(0, "0.3")
    input_stepsize.insert(0, "0.005")
    input_decay.insert(0, "0")
    input_expDecay.insert(0, "0")
    input_stepNum.insert(0, "10")
    input_threshold.insert(0, "0.7")
    # input_time_between_step.insert(0, "0.01")


def optimise():
    start()
    # get_Voltage(weight_Params, position)
    final_voltage = pd.read_csv('VoltagePlaceHolder.csv', sep=',', header=None)
    final_voltage = np.array(final_voltage.iloc[:, 0], dtype=float)

    if len(final_voltage) == 1:
        popupmsg("Update Position First")
    else:
        position = float(input_goto_pos.get())
        # weight_Params = np.array([float(input_ex.get()), float(
        #     input_ey.get()), float(input_harm.get()), float(input_uni.get())])
        # print(final_voltage)
        adam(func_new_voltage_ion, ion_voltage_function,
             take_photon, final_voltage, position, [0, 48])


def popupmsg(msg):
    popup = Tk()
    popup.wm_title("Error")
    popup.geometry('200x100')
    label = Label(popup, text=msg)
    label.pack(side="top", fill="x", pady=10)
    B1 = Button(popup, text="Okay", command=popup.destroy)
    B1.pack()
    popup.mainloop()


def clicked():
    # apply weights and update voltages.
    start()
    res = input_goto_pos.get()
    weight_Params = np.array([float(input_ex.get()), float(
        input_ey.get()), float(input_harm.get()), float(input_uni.get())])  # store weights from user input
    try:
        if res == '' or float(res) > 1000 or float(res) < 0:
            popupmsg("Position Out of range")
            sys.exit()
    except:
        popupmsg("Invalid Input")
    if float(res) == float(input_current_pos.get()) and weightOn.get():
        input_current_pos.delete(0, END)
        input_current_pos.insert(0, res)  # update the current position
        position = float(input_goto_pos.get())

        final_voltage = get_Voltage(weight_Params, position)
        write_voltage(final_voltage)  # write voltages to the DAQ

        final_voltage = np.around(final_voltage, decimals=4)
        # delete previously added voltages to the list

        list_voltages.delete(0, 100)
        for i in range(len(final_voltage)):
            list_voltages.insert(i, "V" + str(i) +
                                    ": " + str(final_voltage[i]))  # update the list
        np.savetxt(r'VoltagePlaceHolder.csv',
                   final_voltage.T, delimiter=',')
        a.cla()
        a2.cla()
        a.plot(final_voltage[1::2], color='blue', linestyle='-', marker='o')
        a2.plot(final_voltage[2::2], color='blue', linestyle='-', marker='o')
        a.set_title("Waveform: 1, 3, 5...", fontsize=14)
        a2.set_title("Waveform:2, 4, 6...", fontsize=14)
        a.set_ylabel("Voltage (V)", fontsize=14)
        a2.set_ylabel("Voltage (V)", fontsize=14)
        a2.set_xlabel("Electrode", fontsize=14)
        fig.tight_layout()
        canvas.draw()
        window.update()
    elif not(float(res) == float(input_current_pos.get())):
        final_voltage = shuttle_ion(float(input_current_pos.get()), float(
            input_goto_pos.get()), weight_Params, float(input_time_between_step.get()))
        final_voltage = np.around(final_voltage, decimals=4)
        list_voltages.delete(0, 100)
        for i in range(len(final_voltage)):
            list_voltages.insert(i, "V" + str(i) +
                                    ": " + str(final_voltage[i]))  # update the list
        np.savetxt(r'VoltagePlaceHolder.csv',
                   final_voltage.T, delimiter=',')

    # input_current_pos.configure(text=res)


running = True


def start():
    """Enable scanning by setting the global flag to True."""
    global running
    running = True


def stop():
    """Stop scanning by setting the global flag to False."""
    global running
    running = False


window.title("Controller and Adam Optimiser for Chip Ion Trap ")
window.geometry('1200x600')

np.savetxt(r'VoltagePlaceHolder.csv',
           [0], delimiter=',')

fontsize = 15

######### Setup plots ########
fig = Figure(figsize=(9, 6))
a = fig.add_subplot(222)  # waveform
a2 = fig.add_subplot(224)
b = fig.add_subplot(221)  # photon count
c = fig.add_subplot(223)  # voltage difference
fig.tight_layout()
canvas = FigureCanvasTkAgg(fig, master=window)
canvas.get_tk_widget().grid(column=4, row=3, rowspan=20, columnspan=8)
canvas.draw()


lbl_current_pos = Label(window, text="Current Position: ",
                        font=("Arial", fontsize))
lbl_goto_pos = Label(window, text="Go to Position: ", font=("Arial", fontsize))
input_current_pos = Entry(window,  text="0", width=10,
                          font=("Arial", fontsize),)
input_goto_pos = Entry(window, width=10, font=("Arial", fontsize))
btn_goto = Button(window, text="Go/Update",
                  command=clicked, font=("Arial", fontsize))

btn_stop = Button(window, text="Stop",
                  command=stop, font=("Arial", fontsize))


lbl_voltage = Label(window, text="Applied Voltages: ",
                    font=("Arial", fontsize))
list_voltages = Listbox(window, height=28, selectmode=MULTIPLE)

btn_optimise = Button(window, text="Optimise",
                      command=optimise, font=("Arial", fontsize))

lbl_weights = Label(text="Manual Weights: ", font=("Arial", fontsize))
lbl_weightsApplied = Label(text="Applied: ", font=("Arial", fontsize))
lbl_ex = Label(text="Ex:", font=("Arial", 16))
input_ex = Entry(window,  font=("Arial", 16), width=6)
lbl_ey = Label(text="Ey:", font=("Arial", 16))
input_ey = Entry(window,  font=("Arial", 16), width=6)
lbl_harm = Label(text="Harmonic:", font=("Arial", 16))
input_harm = Entry(window,  font=("Arial", 16), width=6)
lbl_uni = Label(text="Uniform:", font=("Arial", 16))
input_uni = Entry(window,  font=("Arial", 16), width=6)

ex_var = StringVar()
ey_var = StringVar()
harm_var = StringVar()
uni_var = StringVar()

lbl_ex_applied = Label(textvariable=ex_var, font=("Arial", 16))
lbl_ey_applied = Label(textvariable=ey_var, font=("Arial", 16))
lbl_harm_applied = Label(textvariable=harm_var, font=("Arial", 16))
lbl_uni_applied = Label(textvariable=uni_var, font=("Arial", 16))
window.update()

lbl_stepsize = Label(text="Step Size (Volts):", font=("Arial", 16))
input_stepsize = Entry(window, font=("Arial", 16), width=6)
lbl_stepNum = Label(text="Number of Steps:", font=("Arial", 16))
input_stepNum = Entry(window, font=("Arial", 16), width=6)
lbl_decay = Label(text="Average Decay: (0 <= x < 1)", font=("Arial", 8))
input_decay = Entry(window, font=("Arial", 8), width=6)
lbl_expDecay = Label(text="Exp Decay: (0 <= x < 1)", font=("Arial", 8))
input_expDecay = Entry(window, font=("Arial", 8), width=6)
lbl_threshold = Label(text="Threshold (0<=x<=1):", font=("Arial", 8))
lbl_threshold2 = Label(
    text="Optimiser will terminate if \n photon count = threshold*(starting photon count) \n i.e 0.7 means will terminate if photon count \n drops below 70% of starting photon count", font=("Arial", 8))
input_threshold = Entry(window, font=("Arial", 8), width=6)
lbl_time_between_step = Label(
    text="Shuttle Speed (um/10ms)", font=("Arial", 10))
input_time_between_step = Spinbox(
    values=(0.1, 0.2, 0.5, 1, 1.1, 1.2, 1.5, 2), width=6, font=("Arial", 10))


graphOn = IntVar()
weightOn = IntVar()
Checkbutton(window, text="Show Final Graphs",
            variable=graphOn).grid(row=0, column=1)
Checkbutton(window, text="Manual Weights",
            variable=weightOn).grid(row=4, column=0)


lbl_stepsize.grid(row=0, column=4)
input_stepsize.grid(row=0, column=5)
lbl_stepNum.grid(row=1, column=4)
input_stepNum.grid(row=1, column=5)

lbl_decay.grid(row=0, column=6)
input_decay.grid(row=0, column=7)
lbl_expDecay.grid(row=1, column=6)
input_expDecay.grid(row=1, column=7)

lbl_threshold.grid(row=0, column=8)
input_threshold.grid(row=0, column=9)
lbl_threshold2.grid(row=1, column=8, columnspan=3)

lbl_current_pos.grid(column=0, row=1, pady=(0, 0))
input_current_pos.grid(column=1, row=1, pady=(0, 0))
lbl_goto_pos.grid(column=0, row=2, padx=(0, 0), pady=(0, 0))
input_goto_pos.grid(column=1, row=2, pady=(0, 0), padx=(0, 0))
btn_goto.grid(column=1, row=2, columnspan=4)
btn_stop.grid(column=2, row=2, columnspan=4)
list_voltages.grid(column=12, row=1, pady=(0, 0), rowspan=10)
lbl_voltage.grid(column=12, row=0)

lbl_time_between_step.grid(row=3, column=0)
input_time_between_step.grid(row=3, column=1)

lbl_weights.grid(row=5, column=0)
lbl_weightsApplied.grid(row=5, column=2)
lbl_ex.grid(row=6, column=0)
input_ex.grid(row=6, column=1)
lbl_ey.grid(row=7, column=0)
input_ey.grid(row=7, column=1)
lbl_harm.grid(row=8, column=0)
input_harm.grid(row=8, column=1)
lbl_uni.grid(row=9, column=0)
input_uni.grid(row=9, column=1)
btn_optimise.grid(row=0, column=0)

lbl_ex_applied.grid(row=6, column=2)
lbl_ey_applied.grid(row=7, column=2)
lbl_harm_applied.grid(row=8, column=2)
lbl_uni_applied.grid(row=9, column=2)

startup()

window.mainloop()
