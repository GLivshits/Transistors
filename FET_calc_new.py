from FET_calculation_class import FETs_calculation
import numpy as np
import os


def calculate_capacitance(thickness, epsilon):
    assert len(thickness) == len(epsilon)

    C = 0

    for i in range(len(thickness)):
        C += thickness[i] / epsilon[i] * 1e-7

    return (C / (8.85 * 1e-14)) ** (-1)


path = r'C:\Users\kpebe\OneDrive\Рабочий стол\Transistors\Measurements\Graphene'

chip_names = os.listdir(path)
a = {'G3S1': 3, 'G3S3': 1, 'G3S4': 2, 'G3S7': 2}
for chip in chip_names:

    measurements = os.listdir(path + '\\' + chip)

    for measurement in measurements:

        filepath = path + '\\' + chip + '\\' + measurement

        length_array = np.zeros((24, 13))

        for j in range(13):
            length_array[:, j] = (20 + 5 * j) / 1e4

        if 'G3' in chip:

            print(chip)

            calc = FETs_calculation(columns=a[chip], rows=1, filepath_kernel=path,
                                    chip_name=chip, measurement=measurement,
                                    measurement_type='Isd-Vg', num_sd_bias=5)

            calc.params_calc()
            calc.heatmaps()
            calc.plot_graphs()
            calc.mobility_calc(length_array=length_array, C=calculate_capacitance([300, 81], [3.9, 2]),
                               width=150 / 1e4, plot_graphs=True, plot_heatmaps=True)
            # calc.export_data()
            # calc.lch_dependence()

print('Ready!')
