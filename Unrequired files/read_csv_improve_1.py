import nidaqmx
import pandas as pd
import numpy as np
import time as time

################## Read CSV files####################
print(time.time())
compExTol = pd.read_csv(
    'microwave_20121112_compExTol.csv', sep=',', header=None)
compEyTol = pd.read_csv(
    'microwave_20121112_compEyTol.csv', sep=',', header=None)
harmonic = pd.read_csv('microwave_20121112_harmonic.csv', sep=',', header=None)
uniform_Quad = pd.read_csv(
    'microwave_20121112_uniform_Quad.csv', sep=',', header=None)


compExTol = np.array(compExTol.iloc[1:, :], dtype=float)
compEyTol = np.array(compEyTol.iloc[1:, :], dtype=float)
harmonic = np.array(harmonic.iloc[1:, :], dtype=float)
uniform_Quad = np.array(uniform_Quad.iloc[1:, :], dtype=float)

# print(compExTol)
# print(compEyTol)
# print(harmonic)
# print(uniform_Quad)


def CSV_Read(position, array):
    # print(array[:, 0])
    # print(np.digitize(position, array[:, 0]))

    i = np.digitize(position, array[:, 0])
    m = (array[i, 1:]-array[i-1, 1:])/(array[i, 0]-array[i-1, 0])
    y_int = array[i-1, 1:]
    V_f = m*(position-array[i-1, 0])+y_int

    # print(V_f)
    return V_f


# [ Comp_ExTol, Comp_EyTol_Weight, Harmonic_Weight, Uniform_Quad_Weight]
position = 874.6
weight_Params = np.array([0.2, 0.5, 1, 0.8])


outV_compEx = CSV_Read(position, compExTol)
outV_compEx = outV_compEx*weight_Params[0]

outV_compEy = CSV_Read(position, compEyTol)
outV_compEy = outV_compEy*weight_Params[1]

outV_harmonic = CSV_Read(position, harmonic)
outV_harmonic = outV_harmonic*weight_Params[2]

outV_uniform_Quad = CSV_Read(position, uniform_Quad)
outV_uniform_Quad = outV_uniform_Quad*weight_Params[3]

final_Voltage = outV_compEx + outV_compEy + outV_harmonic + outV_uniform_Quad


print(time.time())
print(final_Voltage)
