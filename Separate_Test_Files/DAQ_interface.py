import nidaqmx
import pandas as pd
import numpy as np
import time
################## Read CSV files####################
print(time.time())
compExTol = pd.read_csv(
    'microwave_20121112_compExTol.csv', sep=',', header=None)
compEyTol = pd.read_csv(
    'microwave_20121112_compEyTol.csv', sep=',', header=None)
harmonic = pd.read_csv('microwave_20121112_harmonic.csv', sep=',', header=None)
uniform_Quad = pd.read_csv(
    'microwave_20121112_uniform_Quad.csv', sep=',', header=None)


def Read_CSV_Get_Voltage(position, CSV, uniform):

    ################### make array with position differences and save input vals #################
    position_diff = []
    function_index = 0
    for i in range(2, CSV.shape[0]):

        # get the difference from the preceding position value and append to array
        position_diff.append(float(CSV[0][i])-float(CSV[0][i-1]))

        # check to see if the input position is in the range here then get the relative number for the input of the voltage function by subtracting the first position value from spreadsheet. Also get the index for which function to use
        if (position >= float(CSV[0][i-1])) and (position <= float(CSV[0][i])):

            # [input, index]
            function_input_index = [position -
                                    float(CSV[0][i-1]), function_index]

            # print("relative function input value and function_index: ",function_input_index)
        function_index += 1

    if position > 1086:
        print("position out of bounds")

    ################## make array with change in voltage change divided by change in position ##################
    voltage_increase_per_unit_position = []

    # for each voltage channel get the differences in voltage for each increment of position
    for j in range(1, CSV.shape[1]):
        # print("j", j)
        n = 0  # index in position_diff array
        place_holder = []
        for i in range(2, CSV.shape[0]):

            # get the difference from the preceding voltage value then divide by change in position
             # array with voltage increase per unit position for each interval and the starting value (slope and y int)
            place_holder.append(
                [(float(CSV[j][i])-float(CSV[j][i-1]))/position_diff[n], float(CSV[j][i-1])])
            n += 1
        voltage_increase_per_unit_position.append(place_holder)

    # indexed by [channel][change per unit position for position]
    voltage_increase_per_unit_position = np.array(
        voltage_increase_per_unit_position)

    # get the function by indexing the channel then function index
    # print(voltage_increase_per_unit_position[41][int(function_input_index[1])])

    ####### Finally append list with voltage for each channel ##############
    voltage_list = []
    for i in range(0, CSV.shape[1]-1):

        # get the gradient, multiply by relative position then add the y intercept
        voltage_list.append(voltage_increase_per_unit_position[i][int(
            function_input_index[1])][0]*function_input_index[0] + voltage_increase_per_unit_position[i][int(function_input_index[1])][1])

    return np.array(voltage_list)


################### Position and Weights input ##################
position = 5
# Comp_ExTol_Weight = 0.2
# Comp_EyTol_Weight = 0.5
# Harmonic_Weight = 1
# Uniform_Quad_Weight = 0.8

Comp_ExTol_Weight = 1
Comp_EyTol_Weight = 0
Harmonic_Weight = 0
Uniform_Quad_Weight = 0

################## Get voltage from CSV file from given position #####################
compExTol_Voltage = Read_CSV_Get_Voltage(position, compExTol, False)
compEyTol_Voltage = Read_CSV_Get_Voltage(position, compEyTol, False)
harmonic_voltage = Read_CSV_Get_Voltage(position, harmonic, False)
uniform_Quad_voltage = Read_CSV_Get_Voltage(position, uniform_Quad, True)


#######Apply the weights ##########
compExTol_Voltage = compExTol_Voltage*Comp_ExTol_Weight
compEyTol_Voltage = compEyTol_Voltage*Comp_EyTol_Weight
harmonic_voltage = harmonic_voltage*Harmonic_Weight
uniform_Quad_voltage = uniform_Quad_voltage*Uniform_Quad_Weight


# print("Ex: ", compExTol_Voltage)
# print("Ey: ", compEyTol_Voltage)
# print("Harmonic: ", harmonic_voltage)
# print("UNiform Quad: ", uniform_Quad_voltage)


############## Add all voltages together to get final voltage ##########
final_voltage = compExTol_Voltage + compEyTol_Voltage + \
    harmonic_voltage + uniform_Quad_voltage
print(time.time())
print(final_voltage)
############### Write Voltage to DAQ #################
"""
# start new operation
with nidaqmx.Task() as task

    # physical channel = Device1/port:AO0, name of function = '', voltage range = (-10, 10)
    task.ao_channels.add_ao_voltage_chan('Dev1/ao0', '', -10, 10)

    # Set the DC voltage value (volts)
    task.write(2.0)

    # end operation
    task.stop()
"""
