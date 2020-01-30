from DAQ_slot_configuration import *  # import the config for the DAQ slots
import nidaqmx
import numpy as np


class DAQ_control():
    def __init__(
        self
    ):

        # inititalise input parameters for the voltage
        self.input_voltage = np.zeros(48)

        self.PXI1Slot2 = PXI1Slot2
        self.PXI1Slot3 = PXI1Slot3
        self.PXI1Slot4 = PXI1Slot4
        self.PXI1Slot5 = PXI1Slot5
        self.PXI1Slot6 = PXI1Slot6
        self.PXI1Slot7 = PXI1Slot7
        self.PXI1Slot8 = PXI1Slot8
        self.PXI1Slot9 = PXI1Slot9
        self.PXI1Slot10 = PXI1Slot10
        self.PXI1Slot11 = PXI1Slot11
        self.PXI1Slot12 = PXI1Slot12
        self.PXI1Slot13 = PXI1Slot13

       """This allocates all the voltages to the right channel in each slot. This will vary depending on your wire configuration of the DAQ to the experiment"""
       self.voltage_slot_2 = np.array(
            [0, 0, self.input_voltage[16], self.input_voltage[12], 0, self.input_voltage[6], self.input_voltage[7], 0])
        self.voltage_slot_3 = np.array([self.input_voltage[13], self.input_voltage[17], 0,
                                   0, 0, self.input_voltage[16], self.input_voltage[14], self.input_voltage[10]])
        self.voltage_slot_4 = np.array([self.input_voltage[8], self.input_voltage[0], 0, self.input_voltage[9],
                                   self.input_voltage[11], self.input_voltage[15], self.input_voltage[17], 0])
        self.oltage_slot_5 = np.array([0, 0, self.input_voltage[19], self.input_voltage[23],
                                   self.input_voltage[27], self.input_voltage[27], self.input_voltage[33], self.input_voltage[33]])
        self.voltage_slot_6 = np.array([self.input_voltage[37], self.input_voltage[41], 0,
                                   0, 0, self.input_voltage[19], self.input_voltage[21], self.input_voltage[25]])
        self.voltage_slot_7 = np.array([self.input_voltage[29], 0, 0, self.input_voltage[31],
                                   self.input_voltage[35], self.input_voltage[39], self.input_voltage[41], 0])
        self.voltage_slot_8 = np.array(
            [0, 0, self.input_voltage[45], self.input_voltage[5], 0, 0, 0, 0])
        self.voltage_slot_9 = np.array([self.input_voltage[4], self.input_voltage[44], 0,
                                   0, 0, self.input_voltage[43], self.input_voltage[47], self.input_voltage[45]])
        self.voltage_slot_10 = np.array(
            [0, 0, 0, 0, self.input_voltage[44], self.input_voltage[46], self.input_voltage[42], 0])
        self.voltage_slot_11 = np.array([0, 0, self.input_voltage[40], self.input_voltage[36],
                                    self.input_voltage[32], self.input_voltage[32], self.input_voltage[26], self.input_voltage[26]])
        self.voltage_slot_12 = np.array([self.input_voltage[22], self.input_voltage[18],
                                    0, 0, 0, self.input_voltage[40], self.input_voltage[38], self.input_voltage[34]])
        self.voltage_slot_13 = np.array([self.input_voltage[30], 0, 0, self.input_voltage[28],
                                    self.input_voltage[24], self.input_voltage[20], self.input_voltage[18], 0])
    def write_voltages(self, input_voltage):
        self.input_voltage = input_voltage
        with nidaqmx.Task() as slot2, nidaqmx.Task() as slot3, nidaqmx.Task() as slot4, nidaqmx.Task() as slot5, nidaqmx.Task() as slot6, nidaqmx.Task() as slot7, nidaqmx.Task() as slot8, nidaqmx.Task() as slot9, nidaqmx.Task() as slot10, nidaqmx.Task() as slot11, nidaqmx.Task() as slot12, nidaqmx.Task() as slot13:
            # open query to send voltages
            slot2.ao_channels.add_ao_voltage_chan(self.PXI1Slot2)
            slot3.ao_channels.add_ao_voltage_chan(self.PXI1Slot3)
            slot4.ao_channels.add_ao_voltage_chan(self.PXI1Slot4)
            slot5.ao_channels.add_ao_voltage_chan(self.PXI1Slot5)
            slot6.ao_channels.add_ao_voltage_chan(self.PXI1Slot6)
            slot7.ao_channels.add_ao_voltage_chan(self.PXI1Slot7)
            slot8.ao_channels.add_ao_voltage_chan(self.PXI1Slot8)
            slot9.ao_channels.add_ao_voltage_chan(self.PXI1Slot9)
            slot10.ao_channels.add_ao_voltage_chan(self.PXI1Slot10)
            slot11.ao_channels.add_ao_voltage_chan(self.PXI1Slot11)
            slot12.ao_channels.add_ao_voltage_chan(self.PXI1Slot12)
            slot13.ao_channels.add_ao_voltage_chan(self.PXI1Slot13)
            # apply voltages to DAQ writing to one slot per write command 
            slot2.write(self.voltage_slot_2)
            slot3.write(self.voltage_slot_3)
            slot4.write(self.voltage_slot_4)
            slot5.write(self.voltage_slot_5)
            slot6.write(self.voltage_slot_6)
            slot7.write(self.voltage_slot_7)
            slot8.write(self.voltage_slot_8)
            slot9.write(self.voltage_slot_9)
            slot10.write(self.voltage_slot_10)
            slot11.write(self.voltage_slot_11)
            slot12.write(self.voltage_slot_12)
            slot13.write(self.voltage_slot_13)
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
    def test():
        None




