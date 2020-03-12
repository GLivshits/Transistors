from Probestation_via_classes import Probestation_FETs
from matplotlib import pyplot as plt
import serial
import pyvisa
import numpy as np
import matplotlib as mpl
import os
import time

measurement = Probestation_FETs('Isd-Vg')
measurement.turn_on_keysight()
measurement.turn_on_drive()

measurements_folder = r'C:\Users\kpebe\OneDrive\Рабочий стол\Transistor measurements'
chip_name = r'HUINA'
os.makedirs(measurements_folder + r'\\' + chip_name, exist_ok = True)

Vg_start = -10
Vg_finish = 10
Vsd = [-1,1]
number_of_cycles = 2
Vg_steps = 11
cols = 3
rows = 3
x_dist = 500
y_dist = 200
x_start = 2
y_start = 2


for i in range(y_start, rows+1):

    for j in range(x_start, cols+1):

        print('Device {}_{}'.format(i, j))

        measurement.measure(Vs1 = Vg_start,
                            Vs2 = Vg_finish,
                            num_Vs = Vg_steps,
                            Vc = Vsd,
                            num_cycles = number_of_cycles)

        measurement.write_to_csv(filepath_kernel = measurements_folder + r'\\' + chip_name,
                                 current_x = j,
                                 current_y = i)

        measurement.move_xyz(x_dist = x_dist,
                             y_dist = y_dist,
                             cols = cols,
                             rows = rows,
                             current_x = j,
                             current_y = i)


print('I reached return function!')
measurement.move_to_zero()


