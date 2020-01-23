import nidaqmx
with nidaqmx.Task() as task:
    task.ao_channels.add_ao_voltage_chan("PXI1Slot2/ao0")
    task.write(2.0)
    task.stop()

    XI1Slot2 / a2       17
    XI1Slot2 / a3       13

    XI1Slot2 / a2       20