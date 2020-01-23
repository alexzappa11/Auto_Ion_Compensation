import nidaqmx
import pandas as pd
import numpy as np
import time as time
import tkinter
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


def CSV_Read(position, array):
    # print(array[:, 0])
    # print(np.digitize(position, array[:, 0])
    i = np.digitize(position, array[:, 0])
    m = (array[i, 1:]-array[i-1, 1:])/(array[i, 0]-array[i-1, 0])
    y_int = array[i-1, 1:]
    V_f = m*(position-array[i-1, 0])+y_int
    return V_f


def shuttle_ion(current_position, target_position, weight_Params, speed, step_size):
    ############# Shuttle if target is greater than current position ###################
    if target_position > current_position:
        # get number of steps for specified step size
        increase_amount = int(
            (target_position - current_position)*(1/step_size))
        for i in range(increase_amount):
            current_position = current_position + step_size
            final_Voltage = get_Voltage(weight_Params, current_position)
            # write_voltage(final_Voltage)
            time.sleep(speed)  # adjust this for shuttling speed

            print(final_Voltage)
            print("current position", current_position)

    ################# Shuttle if target is less than current position #####################
    if target_position < current_position:
        # get number of steps for specified step size
        increase_amount = int(
            (current_position - target_position)*(1/step_size))

        for i in range(increase_amount):
            current_position = current_position - step_size  # stepsize
            final_Voltage = get_Voltage(weight_Params, current_position)
            # write_voltage(final_Voltage)
            time.sleep(speed)

            print(final_Voltage)
            print("current position", current_position)


# [ Comp_ExTol, Comp_EyTol_Weight, Harmonic_Weight, Uniform_Quad_Weight]
weight_Params = np.array([0.22, 0.4, 1.4, 0.3])
############## Define positions ###################
current_position = 200
target_position = 100
time_between_pos = 0.01  # seconds
stepSize = 0.1  # nano meters (1/stepsize must be an integer)

shuttle_ion(current_position, target_position,
            weight_Params, time_between_pos, stepSize)
