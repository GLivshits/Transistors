from FET_calculation_class import FETs_calculation
import numpy as np
import os


def calculate_capacitance(thickness, epsilon):
    assert len(thickness) == len(epsilon)

    C = 0

    for i in range(len(thickness)):
        C += thickness[i] / epsilon[i] * 1e-7

    return (C / (8.85 * 1e-14)) ** (-1)

# основной путь к папке с данными по разным чипам
path = r'C:\Users\kpebe\OneDrive\Рабочий стол\Transistors\Measurements\Pristine series'

chip_names = os.listdir(path)

a = {'G3S1': 3, 'G3S3': 1, 'G3S4': 2, 'G3S7': 2}
for chip in chip_names:

    measurements = os.listdir(path + '\\' + chip)

    for measurement in measurements:

        filepath = path + '\\' + chip + '\\' + measurement

        length_array = np.zeros((24, 10))

        # prototype = np.tile([80,120,160,200], 5)
        lengths = np.array([5, 10, 20, 30, 50, 80, 100, 150, 200, 250])
        for i in range(24):
            if i % 2 == 0:
                length_array[i, :] = lengths/1e4
            else:
                length_array[i, :] = lengths[::-1]/ 1e4

        if chip:

            print(chip)

            calc = FETs_calculation(columns=10, rows=24, filepath_kernel=path,
                                    chip_name=chip, measurement=measurement,
                                    measurement_type='Isd-Vg', num_sd_bias=1)

            calc.params_calc()
            calc.heatmaps()
            # calc.plot_graphs()
            # calc.mobility_calc(length_array=length_array, C=calculate_capacitance([300], [3.9]),
            #                    width= 150/1e4, plot_graphs=False, plot_heatmaps=False)
            # calc.export_data()
            # calc.lch_dependence()

print('Ready!')
