import subprocess as sub
import time as t
def take_data():
    file = r'"C:\src\id800.exe"'
    arguments = " -f test.txt"
    arguments2 = " -C -t 0.1 -e 100 -c 10"
    command = file + arguments2
    # output = sub.call(command)

    proc = sub.Popen(command, shell=True, stdout=sub.PIPE).stdout
    output = proc.read()

    return output
time1 = t.time()
photons = []
for i in range(10):
   # photons.append(str(take_data()).split(' ')[7:26])
   photons.append(str(take_data()))
print(t.time()-time1)
print(photons)


