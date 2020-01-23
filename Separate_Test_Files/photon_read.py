import subprocess as sub
import time as t
import numpy as np


def take_data():
    file = r'"C:\src\id800.exe"'
    arguments = " -f test.txt"
    arguments2 = " -C -t 0.1 -e 10 -c 10"
    command = file + arguments2
    # output = sub.call(command)

    proc = sub.Popen(command, shell=True, stdout=sub.PIPE).stdout
    output = proc.read()

    return output

photon_data = take_data()
photons = np.array(str( photon_data).split(' ')[7:26])
photons = photons.astype(np.int)
print(photons[4])
