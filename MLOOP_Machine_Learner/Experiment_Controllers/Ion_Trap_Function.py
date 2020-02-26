import nidaqmx
import numpy as np
import subprocess as sub
import random


class Ion_Trap_control():
    """This class incorporates all control and output for the DAQ and PMT for the chip trap setup with 48 electrodes.
    Optional: Can adjust the channel, readtime for the PMT, exposure, and confidence bound. When the class is called, pass in the specified variable.
                e.g:
                    IonTrap1 = Ion_Trap_control(channel = 3, readtime = 0.3, exposure = 2000)"""

    def __init__(
        self,
        channel=5,  # PMT channel for where the photons are being read from
        readtime=0.1,  # -t time in s
        exposure=50,  # -e exposure in ms
        confidence_bound=10,  # -c confidence bound
        threshold=0.4,  # maximum drop in photon count before the program terminates
        istest=False  # set to true if running test without PMT of DAQ connected

    ):
        # Initialise PMT parameters
        self.channel = channel
        self.readtime = str(readtime)
        self.exposure = str(exposure)
        self.confidence_bound = str(confidence_bound)
        self.istest = istest

        # multiply the threshold with the starting photon count to get minimum allowed photon count
        if not self.istest:
            self.minimumAllowedPhotonCount = self._get_count()*threshold

        # Inititalise input parameters for the voltage.
        # Assign the slots from the slot configuration this will vary depending on the slots you use for the DAQ
        self.PXI1Slot2 = 'PXI1Slot2/ao0, PXI1Slot2/ao1, PXI1Slot2/ao2, PXI1Slot2/ao3, PXI1Slot2/ao4, PXI1Slot2/ao5, PXI1Slot2/ao6, PXI1Slot2/ao7'
        self.PXI1Slot3 = 'PXI1Slot3/ao0, PXI1Slot3/ao1, PXI1Slot3/ao2, PXI1Slot3/ao3, PXI1Slot3/ao4, PXI1Slot3/ao5, PXI1Slot3/ao6, PXI1Slot3/ao7'
        self.PXI1Slot4 = 'PXI1Slot4/ao0, PXI1Slot4/ao1, PXI1Slot4/ao2, PXI1Slot4/ao3, PXI1Slot4/ao4, PXI1Slot4/ao5, PXI1Slot4/ao6, PXI1Slot4/ao7'
        self.PXI1Slot5 = 'PXI1Slot5/ao0, PXI1Slot5/ao1, PXI1Slot5/ao2, PXI1Slot5/ao3, PXI1Slot5/ao4, PXI1Slot5/ao5, PXI1Slot5/ao6, PXI1Slot5/ao7'
        self.PXI1Slot6 = 'PXI1Slot6/ao0, PXI1Slot6/ao1, PXI1Slot6/ao2, PXI1Slot6/ao3, PXI1Slot6/ao4, PXI1Slot6/ao5, PXI1Slot6/ao6, PXI1Slot6/ao7'
        self.PXI1Slot7 = 'PXI1Slot7/ao0, PXI1Slot7/ao1, PXI1Slot7/ao2, PXI1Slot7/ao3, PXI1Slot7/ao4, PXI1Slot7/ao5, PXI1Slot7/ao6, PXI1Slot7/ao7'
        self.PXI1Slot8 = 'PXI1Slot8/ao0, PXI1Slot8/ao1, PXI1Slot8/ao2, PXI1Slot8/ao3, PXI1Slot8/ao4, PXI1Slot8/ao5, PXI1Slot8/ao6, PXI1Slot8/ao7'
        self.PXI1Slot9 = 'PXI1Slot9/ao0, PXI1Slot9/ao1, PXI1Slot9/ao2, PXI1Slot9/ao3, PXI1Slot9/ao4, PXI1Slot9/ao5, PXI1Slot9/ao6, PXI1Slot9/ao7'
        self.PXI1Slot10 = 'PXI1Slot10/ao0, PXI1Slot10/ao1, PXI1Slot10/ao2, PXI1Slot10/ao3, PXI1Slot10/ao4, PXI1Slot10/ao5, PXI1Slot10/ao6, PXI1Slot10/ao7'
        self.PXI1Slot11 = 'PXI1Slot11/ao0, PXI1Slot11/ao1, PXI1Slot11/ao2, PXI1Slot11/ao3, PXI1Slot11/ao4, PXI1Slot11/ao5, PXI1Slot11/ao6, PXI1Slot11/ao7'
        self.PXI1Slot12 = 'PXI1Slot12/ao0, PXI1Slot12/ao1, PXI1Slot12/ao2, PXI1Slot12/ao3, PXI1Slot12/ao4, PXI1Slot12/ao5, PXI1Slot12/ao6, PXI1Slot12/ao7'
        self.PXI1Slot13 = 'PXI1Slot13/ao0, PXI1Slot13/ao1, PXI1Slot13/ao2, PXI1Slot13/ao3, PXI1Slot13/ao4, PXI1Slot13/ao5, PXI1Slot13/ao6, PXI1Slot13/ao7'

    def _get_count(self):
        """Reads photon count: read photon count from executable for PMTid800 software. Command is found in manual"""
        try:
            file = r'"C:\src\id800.exe"'
            arguments = " -C -t " + self.readtime + " -e " + \
                        self.exposure + " -c " + self.confidence_bound
            command = file + arguments
            # print(command)
            stream = sub.Popen(command,
                               stdout=sub.PIPE,
                               stderr=sub.PIPE, universal_newlines=True)
            output = stream.communicate()  # save output of cmd to variable
            photons = np.array(str(output).split(' ')[7:26])
            photons = photons.astype(np.int)
            # negative as the algorithm seeks to minimise the cost (cost being the photon count)
            photons = -photons[(self.channel - 1)]
            # photons = round(photons, -2) #this can be implemented if rounding of the photon count is nessecary
        except:
            print("Photon Read Error: Photon Count returned as photon count = 0")
            photons = 0
            exit()

        return photons

    def _write_voltage(self, input_voltage):

        if not len(input_voltage) == 48:
            print("Parameter array length is not 48")
            exit()

        """Given a voltage for the DAQ, this will be appled to the DAQ and the photon count will be returned """
        self.input_voltage = input_voltage

        """This allocates all the voltages to the right channel in each slot. This will vary depending on your wire configuration of the DAQ to the experiment"""
        self.voltage_slot_2 = np.array(
            [0, 0, self.input_voltage[16], self.input_voltage[12], 0, self.input_voltage[6], self.input_voltage[7], 0])
        self.voltage_slot_3 = np.array([self.input_voltage[13], self.input_voltage[17], 0,
                                        0, 0, self.input_voltage[16], self.input_voltage[14], self.input_voltage[10]])
        self.voltage_slot_4 = np.array([self.input_voltage[8], self.input_voltage[0], 0, self.input_voltage[9],
                                        self.input_voltage[11], self.input_voltage[15], self.input_voltage[17], 0])
        self.voltage_slot_5 = np.array([0, 0, self.input_voltage[19], self.input_voltage[23],
                                        self.input_voltage[27], self.input_voltage[27], self.input_voltage[33],
                                        self.input_voltage[33]])
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
                                         self.input_voltage[32], self.input_voltage[32], self.input_voltage[26],
                                         self.input_voltage[26]])
        self.voltage_slot_12 = np.array([self.input_voltage[22], self.input_voltage[18],
                                         0, 0, 0, self.input_voltage[40], self.input_voltage[38],
                                         self.input_voltage[34]])
        self.voltage_slot_13 = np.array([self.input_voltage[30], 0, 0, self.input_voltage[28],
                                         self.input_voltage[24], self.input_voltage[20], self.input_voltage[18], 0])

        try:
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
        except:
            print("Voltage Write Error, ensure DAQ is connected")

          # Calls the get count function to return the current photon count after the voltages are applied

    def Ion_function(self, input_voltage):
        """Returns the photon count with the given inputted voltage to the DAQ"""

        if self.istest:
             # set photon count to random if istest is true
            self.photon_count = random.randint(0, 2000)
        else:
             # write voltage values to the DAQ
            self._write_voltage(input_voltage)
            self.photon_count = self._get_count()  # Get the photon count

        return self.photon_count  # return photon count


##### For testing Purposes ensure this is commented before running any external controller ###
# Ion_Trap_control = Ion_Trap_control()
# best_voltage =  np.array([-0.76946399,  0.03126644,  0.01227583, -0.02074481, -0.81386208,        0.84565323,  0.33459753, -0.27779589,  0.31202691, -0.31500174,        0.35165478, -0.32228858,  0.33917303, -0.3041594 ,  0.32604872,       -0.30917424,  0.31003287, -0.32733473,  0.36496983, -0.31687441,        0.33490359, -0.31542131,  0.3440404 , -0.3266181 ,  0.32299595,       -0.33479072,  0.32458658, -0.31810331,  0.3304424 , -0.28309829,        2.74354768,  2.17268162,  0.94281746,  0.48938595, -3.16064925,       -3.52517838, -1.55874336, -1.94320054,  2.05913316,  1.57945178,        1.60988956,  1.0389809 ,  0.32177613, -0.34905581,  0.34300037,       -0.32592017,  0.34501902, -0.29680224])
#
# photonCount = Ion_Trap_control.Ion_function(best_voltage)
# photonCount = Ion_Trap_control._get_count()
# print(photonCount)
