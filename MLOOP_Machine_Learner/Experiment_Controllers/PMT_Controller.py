
import subprocess as sub
import random
import numpy as np


class photon_count():
    def __init__(
        self,
        channel=5,  # PMT channel for where the photons are being read from
        readtime=0.1,  # -t time in s
        exposure=50,  # -e exposure in ms
        confidence_bound=10  # -c confidence bound
    ):
        self.channel = channel
        self.readtime = str(readtime)
        self.exposure = str(exposure)
        self.confidence_bound = str(confidence_bound)

    def get_count(self):
        """reads photon count: read photon count from executable for PMTid800 software. Command is found in manual"""
        file = r'"C:\src\id800.exe"'
        arguments = " -C -t " + self.readtime + " -e " + \
            self.exposure + " -c " + self.confidence_bound
        command = file + arguments
        print(command)
        stream = sub.Popen(command,
                           stdout=sub.PIPE,
                           stderr=sub.PIPE, universal_newlines=True)
        output = stream.communicate()  # save output of cmd to variable
        photons = np.array(str(output).split(' ')[7:26])
        photons = photons.astype(np.int)
        # negative as the algorithm seeks to minimise the cost (cost being the photon count)
        photons = -photons[(self.channel - 1)]
        # photons = round(photons, -2)
        return photons

    def testOutput():
        """return random number for testing purposes"""
        time.sleep(0.01)  # measurement delay
        return -random.randint(2000, 5000)

##### For testing Purposes ensure this is commented before running any external controller ###
# PMT = photon_count()
# print(PMT.get_count())

