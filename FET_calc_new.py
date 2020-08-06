from FET_calculation_class import FETs_calculation
import numpy as np
import os

def calculate_capacitance(thickness, epsilon):

    assert len(thickness) == len(epsilon)

    C = 0

    for i in range(len(thickness)):

        C += thickness[i]/epsilon[i] * 1e-7

    return (C / (8.85 * 1e-14)) ** (-1)


path = r'C:\Users\kpebe\OneDrive\Рабочий стол\N2O'

chip_names = os.listdir(path)

for chip in chip_names:

    measurements = os.listdir(path + '\\' + chip)
    print(chip)

    for measurement in measurements:

        filepath = path + '\\' + chip + '\\' + measurement

        # length_array = np.zeros((8, 4))
        #
        # for j in range(4):
        #
        #     length_array[:, j] = (60 + 20 * j) / 1e4

        if '10min32s' in chip:
            calc = FETs_calculation(columns = 13, rows = 24, filepath_kernel=path,
                                chip_name = chip, measurement = measurement,
                                measurement_type = 'Isd-Vg', num_sd_bias = 1)

            calc.params_calc()
            calc.heatmaps()
            calc.plot_graphs()
            # calc.mobility_calc(length_array = length_array, C = 1, width = 1, plot_graphs=True, plot_heatmaps=True)
            # calc.export_data()
            # calc.lch_dependence()
