from FET_calculation_class import FETs_calculation
import numpy as np
import os

path = r'C:\Users\kpebe\OneDrive\Рабочий стол\Transistors 2020'

chip_names = os.listdir(path)

length_array = np.zeros((20, 8))

for j in range(0, 8):

    length_array[:, j] = (60 + 20 * j) / 1e4

calc = FETs_calculation(columns = 8, rows = 20, filepath_kernel=path,
                        chip_name='1st regime 140 s', measurement='IV',
                        measurement_type = 'Isd-Vsd', length_array=length_array,
                        width=150/1e4, t_ox=500)

calc.num_sd_bias = 7
calc.plot_graphs()




# for chip in chip_names:
#     measurements = os.listdir(path + '\\' + chip)
#     for measurement in measurements:
#         filepath = path + '\\' + chip + '\\' + measurement
#
#         length_array = np.zeros((20, 8))
#
#         for j in range(0, 8):
#
#             length_array[:, j] = (60 + 20 * j) / 1e4
#
#         print(chip)
#         calc = FETs_calculation(columns = 8, rows = 20, filepath_kernel=path,
#                                 chip_name=chip, measurement=measurement,
#                                 measurement_type = 'Isd-Vg', length_array=length_array,
#                                 width=150/1e4, t_ox=500)
#         calc.params_calc()
#         calc.heatmaps()
#         # calc.plot_graphs()
#         calc.mobility_calc()