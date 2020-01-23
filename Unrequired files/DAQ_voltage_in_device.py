import numpy as np
import nidaqmx
import time


final_voltage = np.array([0, 0, 0, 0, -4.89323688, 4.87633803, 4.21009563, 0.54137305, -2.56616087, -5.90619072, 3.383148, -0.24544995, 2.68213093, -1.14521202, 2.26031778, -1.56279098, 1.92823141, -1.91687055, 1.92823141, -1.91687055, 1.92823141, -1.91687055, 1.92823141, -1.91687055, 1.92823141, -1.91687055, 1.92823141, -1.91687055, 1.92823141, -1.91687055, 1.92823141, -1.91687055, 1.92823141, -1.91687055, 1.92823141, -1.91687055, 1.92823141, -1.91687055,  1.92823141, -1.91687055, 1.92823141, -1.91687055,
                          1.92823141, -1.91687055, 1.92823141, -1.91687055,  1.92823141, -1.91687055])

PXI1Slot2 = 'PXI1Slot2/ao0, PXI1Slot2/ao1, PXI1Slot2/ao2, PXI1Slot2/ao3, PXI1Slot2/ao4, PXI1Slot2/ao5, PXI1Slot2/ao6, PXI1Slot2/ao7'
PXI1Slot3 = 'PXI1Slot3/ao0, PXI1Slot3/ao1, PXI1Slot3/ao2, PXI1Slot3/ao3, PXI1Slot3/ao4, PXI1Slot3/ao5, PXI1Slot3/ao6, PXI1Slot3/ao7'
PXI1Slot4 = 'PXI1Slot4/ao0, PXI1Slot4/ao1, PXI1Slot4/ao2, PXI1Slot4/ao3, PXI1Slot4/ao4, PXI1Slot4/ao5, PXI1Slot4/ao6, PXI1Slot4/ao7'
PXI1Slot5 = 'PXI1Slot5/ao0, PXI1Slot5/ao1, PXI1Slot5/ao2, PXI1Slot5/ao3, PXI1Slot5/ao4, PXI1Slot5/ao5, PXI1Slot5/ao6, PXI1Slot5/ao7'
PXI1Slot6 = 'PXI1Slot6/ao0, PXI1Slot6/ao1, PXI1Slot6/ao2, PXI1Slot6/ao3, PXI1Slot6/ao4, PXI1Slot6/ao5, PXI1Slot6/ao6, PXI1Slot6/ao7'
PXI1Slot7 = 'PXI1Slot7/ao0, PXI1Slot7/ao1, PXI1Slot7/ao2, PXI1Slot7/ao3, PXI1Slot7/ao4, PXI1Slot7/ao5, PXI1Slot7/ao6, PXI1Slot7/ao7'
PXI1Slot8 = 'PXI1Slot8/ao0, PXI1Slot8/ao1, PXI1Slot8/ao2, PXI1Slot8/ao3, PXI1Slot8/ao4, PXI1Slot8/ao5, PXI1Slot8/ao6, PXI1Slot8/ao7'
PXI1Slot9 = 'PXI1Slot9/ao0, PXI1Slot9/ao1, PXI1Slot9/ao2, PXI1Slot9/ao3, PXI1Slot9/ao4, PXI1Slot9/ao5, PXI1Slot9/ao6, PXI1Slot9/ao7'
PXI1Slot10 = 'PXI1Slot10/ao0, PXI1Slot10/ao1, PXI1Slot10/ao2, PXI1Slot10/ao3, PXI1Slot10/ao4, PXI1Slot10/ao5, PXI1Slot10/ao6, PXI1Slot10/ao7'
PXI1Slot11 = 'PXI1Slot11/ao0, PXI1Slot11/ao1, PXI1Slot11/ao2, PXI1Slot11/ao3, PXI1Slot11/ao4, PXI1Slot11/ao5, PXI1Slot11/ao6, PXI1Slot11/ao7'
PXI1Slot12 = 'PXI1Slot12/ao0, PXI1Slot12/ao1, PXI1Slot12/ao2, PXI1Slot12/ao3, PXI1Slot12/ao4, PXI1Slot12/ao5, PXI1Slot12/ao6, PXI1Slot12/ao7'
PXI1Slot13 = 'PXI1Slot13/ao0, PXI1Slot13/ao1, PXI1Slot13/ao2, PXI1Slot13/ao3, PXI1Slot13/ao4, PXI1Slot13/ao5, PXI1Slot13/ao6, PXI1Slot13/ao7'

voltage_slot_2 = np.array(
    [0, 0, final_voltage[16], final_voltage[12], 0, final_voltage[6], final_voltage[7], 0])

voltage_slot_3 = np.array([final_voltage[13], final_voltage[17], 0, 0, 0, final_voltage[16], final_voltage[14], final_voltage[10])
voltage_slot_4 = np.array([final_voltage[8], 0, 0, final_voltage[9], final_voltage[11], final_voltage[15], final_voltage[17], 0])
voltage_slot_5 = np.array([0, 0, final_voltage[19], final_voltage[23], final_voltage[27], final_voltage[27], final_voltage[33], final_voltage[33]])
voltage_slot_6 = np.array([final_voltage[37], final_voltage[41], 0, 0, 0, final_voltage[19], final_voltage[21], final_voltage[25]])
voltage_slot_7 = np.array([final_voltage[29], 0, 0, final_voltage[31], final_voltage[35], final_voltage[39], final_voltage[41], 0])
voltage_slot_8 = np.array([0, 0, final_voltage[45], final_voltage[5], 0, 0, 0, 0])
voltage_slot_9 = np.array([final_voltage[4], final_voltage[44], 0, 0, 0, final_voltage[43], final_voltage[47], final_voltage[45]])
voltage_slot_10 = np.array([0, 0, 0, 0, final_voltage[44], final_voltage[46], final_voltage[42], 0])
voltage_slot_11 = np.array([0, 0, final_voltage[40], final_voltage[36], final_voltage[32], final_voltage[32], final_voltage[26], final_voltage[26]])
voltage_slot_12 = np.array([final_voltage[22], final_voltage[18], 0, 0, 0, final_voltage[40], final_voltage[38], final_voltage[34]])
voltage_slot_13 = np.array([final_voltage[30], 0, 0, final_voltage[28], final_voltage[24], final_voltage[20], final_voltage[18], 0])


with nidaqmx.Task() as slot2, nidaqmx.Task() as slot3, nidaqmx.Task() as slot4, nidaqmx.Task() as slot5, nidaqmx.Task() as slot6, nidaqmx.Task() as slot7, nidaqmx.Task() as slot8, nidaqmx.Task() as slot9, nidaqmx.Task() as slot10, nidaqmx.Task() as slot11, nidaqmx.Task() as slot12, nidaqmx.Task() as slot13:
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

    slot2.write(voltage_slot_2)
    slot3.write(voltage_slot_2)
    slot4.write(voltage_slot_2)
    slot5.write(voltage_slot_2)
    slot6.write(voltage_slot_2)
    slot7.write(voltage_slot_2)
    slot8.write(voltage_slot_2)
    slot9.write(voltage_slot_2)
    slot10.write(voltage_slot_2)
    slot11.write(voltage_slot_2)
    slot12.write(voltage_slot_2)
    slot13.write(voltage_slot_2)

    slot2.stop
    slot3.stop
    slot4.stop
    slot5.stop
    slot6.stop
    slot7.stop
    slot8.stop
    slot9.stop
    slot10.stop
    slot11.stop
    slot12.stop
    slot13.stop
