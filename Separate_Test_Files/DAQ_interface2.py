import nidaqmx
import pandas as pd
import numpy as np
import time

############### Write Voltage to DAQ #################

# start new operation
with nidaqmx.Task() as task:

    # physical channel = Device1/port:AO0, name of function = '', voltage range = (-10, 10)
    task.ao_channels.add_ao_voltage_chan('Dev1/ao0', '', -10, 10)

    # Set the DC voltage value (volts)
    task.write(2.0)

    # end operation
    task.stop()
