import os
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

# эта функция обсчитывает все
def whole_calc(filepath, chip, measurement, length_array, width):
    # здесь делаются csvшки на основе dat

    devices_num = 0
    for root, dirs, files in os.walk(filepath + r'\data'):
        for file in files:
            if os.path.splitext(file)[1] == '.dat':
                devices_num += 1

    if devices_num == 0:
        return

    devices_index = np.zeros((devices_num, 2))
    i = 0
    for root, dirs, files in os.walk(filepath + r'\data'):
        for file in files:
            if os.path.splitext(file)[1] == '.dat':
                pre_ind = os.path.splitext(file)[0].split('_')
                devices_index[i, 0] = int(pre_ind[1])
                devices_index[i, 1] = int(pre_ind[2])
                i += 1

    n = int(np.max(devices_index[:, 0]))
    m = int(np.max(devices_index[:, 1]))
    devices_index = np.zeros((n * m, 2))

    for i in range(0, n):
        for j in range(0, m):
            try:
                some_arr = []
                filepath_dat = (filepath + r'\data\fet_{}_{}.dat').format(i + 1, j + 1)
                f = open(filepath_dat, "r")
                lines = f.readlines()
                for line in lines:
                    if 'l' in line:
                        some_arr.append(lines.index(line))
                for line in lines:
                    if line.startswith('0,00000E+0') or line.startswith('0.00000E+0'):
                        some_arr.append(lines.index(line))
                some_arr = np.array(some_arr)
                for item in some_arr:
                    lines.pop(item)
                    some_arr -= 1
                f.close()
                f = open((filepath + r'\data\fet_{}_{}.dat').format(i + 1, j + 1), 'w')
                for line in lines:
                    line = line.replace(',', '.')
                    f.write(line)
                f.close()
            except FileNotFoundError:
                pass

    for i in range(0, n):
        for j in range(0, m):
            try:
                filepath_dat = r'{}\\data\\fet_{}_{}.dat'.format(filepath, i + 1, j + 1)
                filepath_csv = r'{}\\data\\fet_{}_{}.csv'.format(filepath, i + 1, j + 1)
                with open(filepath_dat) as f:
                    with open(filepath_csv, "w") as f1:
                        for line in f:
                            f1.write(line)
                f.close()
                f1.close()
            except FileNotFoundError:
                pass

    # создание папок

    os.makedirs(filepath + r'\Transfer curves', exist_ok=True)
    os.makedirs(filepath + r'\Onoff heatmaps', exist_ok=True)
    os.makedirs(filepath + r'\Ropen and Rclosed heatmaps', exist_ok=True)
    os.makedirs(filepath + r'\Mobility graphs', exist_ok=True)
    os.makedirs(filepath + r'\Onoff dependence from Lch', exist_ok=True)
    os.makedirs(filepath + r'\Ropen dependence from Lch', exist_ok=True)
    os.makedirs(filepath + r'\Normalized transfers', exist_ok=True)
    os.makedirs(filepath + r'\Mobility heatmaps', exist_ok=True)

    # инициализация всяких массивов и нужных переменных

    filepath_csv = (filepath + r'\data\fet_{}_{}.csv').format(n, m)
    df = pd.read_csv(filepath_csv, engine='python', sep='	', header=None)
    num_sd_bias = int(df[0].nunique())
    m1 = int(df.shape[0] / num_sd_bias)
    n1 = df.shape[1]
    number_of_cycles = int((n1 - 2) / 2)

    data = np.ndarray(([n, m, n1, m1, num_sd_bias]), dtype=float)  # это пятимерный тензор, содержащий все данные в себе
    data.fill(np.nan)
    on_off_ratio = np.zeros((n, m, num_sd_bias, 4))
    on_off_ratio.fill(np.nan)
    single_transistor_currents = np.zeros((m1, number_of_cycles))  # это массив для токов для транзистора
    # (то есть по одному транзистору)
    single_transistor_gate_currents = np.zeros((m1,
                                                number_of_cycles))  # это массив сопротивления гейта для каждого
    # напряжения, чтоб проверить, не пробит ли он
    gate_current = np.zeros((n, m, num_sd_bias))  # это матрица сопротивлений гейта
    gate_current.fill(np.nan)
    max_forward_p, max_backward_p, max_forward_n, max_backward_n = int(0), int(m1 - 1), int((m1 - 2) / 2), int(
        (m1 - 2) / 2 + 1)  # это точки где токи должны быть максимальны или минимальны, то есть по краям графиков
    min_forward, min_backward = 0, 0
    max_currents, min_currents = np.zeros((n, m, num_sd_bias, 4)), np.zeros(
        (n, m, num_sd_bias, 2))  # массивы максимальных и минимальных токов, нужны для рассчета on/off
    max_currents.fill(np.nan)
    min_currents.fill(np.nan)
    open_state_resistance = np.zeros((n, m, num_sd_bias, 4))  # массив сопротивлений транзисторов в открытом состоянии
    open_state_resistance.fill(np.nan)
    status_array = np.full((n, m), float('NaN'))  # массив статусов
    closed_state_resistance = np.zeros((n, m, num_sd_bias, 2))
    closed_state_resistance.fill(np.nan)

# В случае абивалентности используется 4 on/off и другие соответствующие характеристики в следующем порядке:
# p-тип прямая ветвь, n-тип прямая ветвь, p-тип обратная ветвь, n-тип обратная ветвь
    branches_names = ['p-type forward', 'n-type forward', 'p-type backward', 'n-type backward']

    # этот цикл заполняет пятимерный тензор всеми данными по чипу

    for i in range(0, n):
        for j in range(0, m):
            try:
                filepath_csv = (filepath + r'\data\fet_{}_{}.csv').format(i + 1, j + 1)
                df = pd.read_csv(filepath_csv, engine='python', sep='	', header=None)
                for z in range(0, num_sd_bias):
                    for k in range(0, n1):
                        for l in range(0, m1):
                            data[i, j, k, l, z] = df.iloc[l + m1 * z, k]
            except FileNotFoundError:
                pass

    # этот цикл считает отношение тока включения к току выключения
    idk = 0
    for i in range(0, n):
        for j in range(0, m):
            devices_index[idk, 0] = i + 1
            devices_index[idk, 1] = j + 1
            idk += 1
            min_forward = 0
            min_backward = 0
            for z in range(0, num_sd_bias):
                for l in range(0, m1):
                    for k in range(0, number_of_cycles):
                        single_transistor_currents[l, k] = data[i, j, 2 + 2 * k, l, z]
                        single_transistor_gate_currents[l, k] = abs(data[i, j, 3 + 2 * k, l, z]) / number_of_cycles
                        if single_transistor_currents[l, k] * np.sign(data[i, j, 0, 0, z]) < 8 / 1e13:
                            single_transistor_currents[l, k] = 8 / 1e13 * np.sign(data[i, j, 0, 0, z])
                gate_current[i, j, z] = np.mean(single_transistor_gate_currents)
                max_currents[i, j, z, 0] = np.mean(single_transistor_currents[max_forward_p, :])
                max_currents[i, j, z, 1] = np.mean(single_transistor_currents[max_forward_n, :])
                max_currents[i, j, z, 2] = np.mean(single_transistor_currents[max_backward_p, :])
                max_currents[i, j, z, 3] = np.mean(single_transistor_currents[max_backward_n, :])
                if max_currents[i, j, z, 0] < max_currents[i, j, z, 1]:
                    k = max_currents[i, j, z, 0]
                    max_currents[i, j, z, 0] = max_currents[i, j, z, 1]
                    max_currents[i, j, z, 1] = k
                    k = max_currents[i, j, z, 2]
                    max_currents[i, j, z, 2] = max_currents[i, j, z, 3]
                    max_currents[i, j, z, 3] = k
                for k in range(0, number_of_cycles):
                    min_forward += np.amin(single_transistor_currents[:max_forward_n, k])/number_of_cycles
                    min_backward += np.amin(single_transistor_currents[max_backward_n:, k]) / number_of_cycles
                min_currents[i, j, z, 0] = min_forward
                min_currents[i, j, z, 1] = min_backward
                on_off_ratio[i, j, z, 0] = max_currents[i, j, z, 0] / min_currents[i, j, z, 0]
                on_off_ratio[i, j, z, 1] = max_currents[i, j, z, 1] / min_currents[i, j, z, 0]
                on_off_ratio[i, j, z, 2] = max_currents[i, j, z, 2] / min_currents[i, j, z, 1]
                on_off_ratio[i, j, z, 3] = max_currents[i, j, z, 3] / min_currents[i, j, z, 1]
                open_state_resistance[i, j, z, 0] = abs(data[i, j, 0, 0, z] / max_currents[i, j, z, 0])
                open_state_resistance[i, j, z, 1] = abs(data[i, j, 0, 0, z] / max_currents[i, j, z, 1])
                open_state_resistance[i, j, z, 2] = abs(data[i, j, 0, 0, z] / max_currents[i, j, z, 2])
                open_state_resistance[i, j, z, 3] = abs(data[i, j, 0, 0, z] / max_currents[i, j, z, 3])
                closed_state_resistance[i, j, z, 0] = abs(data[i, j, 0, 0, z] / min_currents[i, j, z, 0])
                closed_state_resistance[i, j, z, 1] = abs(data[i, j, 0, 0, z] / min_currents[i, j, z, 1])
                if np.mean(abs(max_currents[i, j, z, :])) < 10 ** (-11):  # условие для непроводящих
                    on_off_ratio[i, j, z, :] = float('NaN')
                    open_state_resistance[i, j, z, :] = float('NaN')
                    status_array[i, j] = 1
                    closed_state_resistance[i, j, z, :] = float('NaN')
                if gate_current[i, j, z] > 1e-8:  # условие для пробитого гейта
                    on_off_ratio[i, j, z, :] = float('NaN')
                    open_state_resistance[i, j, z, :] = float('NaN')
                    status_array[i, j] = 2
                if np.mean(abs(max_currents[i, j, z, :])) > 1e2 or np.mean(abs(min_currents[i, j, z, :])) > 1e2 or np.isnan(
                        data[i, j, :, :, z]).any():  # Данные говно
                    on_off_ratio[i, j, z, :] = float('NaN')
                    open_state_resistance[i, j, z, :] = float('NaN')
                    status_array[i, j] = 3

    #     # эта штука делает heatmap с on/off ratio

    patches = [mpl.patches.Patch(color='gray', label='Non-conductive'),
               mpl.patches.Patch(color='black', label='Gate leakage'),
               mpl.patches.Patch(color='rosybrown', label='No data')]
    cmap = mpl.colors.ListedColormap(['gray', 'black', 'rosybrown'])
    for i in range(0, 4):
        if np.mean(on_off_ratio[:, :, :, i][~np.isnan(on_off_ratio[:, :, :, i])]) > 2:
            for z in range(0, num_sd_bias):
                fig, ax = plt.subplots(figsize=(8, 8))
                df = pd.DataFrame(status_array)
                df.columns = range(1, m + 1)
                df.index = range(1, n + 1)
                sns.heatmap(df, linewidth=.3, ax=ax, cmap=cmap, vmin=1, vmax=3, cbar=False)
                df = pd.DataFrame(on_off_ratio[:, :, z, i])
                df.columns = range(1, m + 1)
                df.index = range(1, n + 1)
                sns.heatmap(df, linewidth=.3, annot=True,
                        fmt=".2g", ax=ax,
                        vmin=1, vmax=10 ** 7, cbar=True,
                        norm=mpl.colors.LogNorm(), cmap='rainbow',
                        cbar_kws={'ticks': [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]})
                plt.title(chip + '\n' + measurement + '\n' + 'On/off, bias = {} V'.format(data[0, 0, 0, 0, z])
                          + '\n' + '{}'.format(branches_names[i]))
                fig.legend(handles=patches, loc=8, ncol=3, fontsize='small')
                plt.savefig((filepath + r'\Onoff heatmaps' + r'\On off ratio {} V {}.png'.format(data[0, 0, 0, 0, z], branches_names[i]))
                            , dpi=300,
                        bbox_inches='tight', format='png')
                plt.clf()
                plt.close(fig)

    # здесь строится карта сопротивлений транзисторов в открытом состоянии
    for i in range (0, 4):
        if np.mean(on_off_ratio[:, :, :, i][~np.isnan(on_off_ratio[:, :, :, i])]) > 2:
            for z in range(0, num_sd_bias):
                fig, ax = plt.subplots(figsize=(8, 8))
                df = pd.DataFrame(status_array)
                df.columns = range(1, m + 1)
                df.index = range(1, n + 1)
                sns.heatmap(df, linewidth=.3, ax=ax, cmap=cmap, vmin=1, vmax=3, cbar=False)
                df = pd.DataFrame(open_state_resistance[:, :, z, i])
                df.columns = range(1, m + 1)
                df.index = range(1, n + 1)
                sns.heatmap(df, linewidth=.3, annot=True, fmt=".2g", vmin=10 ** 3,
                            vmax=10 ** 8, ax=ax, cbar=True, norm=mpl.colors.LogNorm(), cmap='rainbow',
                            cbar_kws={'ticks': [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]})
                plt.title(chip + '\n' + measurement + '\n' + 'Ropen, bias = {} V'.format(data[0, 0, 0, 0, z])
                          + '\n' + '{}'.format(branches_names[i]))
                fig.legend(handles=patches, loc=8, ncol=3, fontsize='small')
                plt.savefig((filepath + r'\Ropen and Rclosed heatmaps' +
                             r'\Ropen {} V {}.png'.format(data[0, 0, 0, 0, z], branches_names[i])),
                            format='png', dpi=300, bbox_inches='tight')
                plt.clf()
                plt.close(fig)

    # здесь строится карта сопротивлений транзисторов в закрытом состоянии
    direction = ['forward', 'backward']
    for i in range(0, 2):
        for z in range(0, num_sd_bias):
            fig, ax = plt.subplots(figsize=(8, 8))
            df = pd.DataFrame(status_array)
            df.columns = range(1, m + 1)
            df.index = range(1, n + 1)
            sns.heatmap(df, linewidth=.3, ax=ax, cmap=cmap, vmin=1, vmax=3, cbar=False)
            df = pd.DataFrame(closed_state_resistance[:, :, z, i])
            df.columns = range(1, m + 1)
            df.index = range(1, n + 1)
            sns.heatmap(df, linewidth=.3, annot=True, fmt=".2g", vmin=10 ** 6,
                        vmax=10 ** 11, ax=ax, cbar=True, norm=mpl.colors.LogNorm(), cmap='rainbow',
                        cbar_kws={'ticks': [1e6, 1e7, 1e8, 1e9, 1e10, 1e11]})
            plt.title(chip + '\n' + measurement + '\n' + 'Rclosed, bias = {} V {}'.format(data[0, 0, 0, 0, z], direction[i]))
            fig.legend(handles=patches, loc=8, ncol=3, fontsize='small')
            plt.savefig((filepath + r'\Ropen and Rclosed heatmaps' +
                         r'\Rclosed {} V {}.png'.format(data[0, 0, 0, 0, z], direction[i])),
                        format='png', dpi=300, bbox_inches='tight')
            plt.clf()
            plt.close(fig)

    # этот цикл сохраняет графики всех транзисторов в лог скейле

    # locmax = mpl.ticker.SymmetricalLogLocator(base=10.0, linthresh=1e-13, subs=(0.1, 1, 10, 100,))
    # locmin = mpl.ticker.SymmetricalLogLocator(base=10.0, linthresh=1e-13, subs=(0.2, 0.4, 0.6, 0.8, 1,))
    # for i in range(0, n):
    #     for j in range(0, m):
    #         maxt = np.zeros((num_sd_bias))
    #         mint = np.zeros((num_sd_bias))
    #         if status_array[i, j] != 3:
    #             fig, ax = plt.subplots(figsize=(6, 6))
    #             for z in range(0, num_sd_bias):
    #                 curr = np.zeros((m1))
    #                 for k in range(0, number_of_cycles):
    #                     curr += data[i, j, 2 + 2 * k, :, z]
    #                 curr /= number_of_cycles
    #                 plt.plot(data[i, j, 1, :, z], np.sign(data[i, j, 0, 0, z]) * curr,
    #                          label='{} V'.format(data[i, j, 0, 0, z]))
    #                 maxt[z] = max(np.sign(data[i, j, 0, 0, z]) * curr)
    #                 mint[z] = min(np.sign(data[i, j, 0, 0, z]) * curr)
    #             plt.title(chip + '\n' + measurement + '\n' + 'FET {}_{}'.format(i + 1, j + 1))
    #             plt.legend(loc=0)
    #             plt.yscale('symlog', linthreshy=10 ** (-13))
    #             plt.xlabel('Vg, V', fontsize=14)
    #             plt.ylabel('Isd, A', fontsize=14)
    #             ax.yaxis.set_major_locator(locmax)
    #             ax.yaxis.set_minor_locator(locmin)
    #             ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    #             ticks = np.array(list(ax.axes.get_yticks()))
    #             ticks = ticks[(abs(ticks) > 5 * 1e-14) | (ticks == 0)]
    #             ticks = ticks[ticks < max(maxt)]
    #             ticks = ticks[ticks > min(mint)]
    #             ax.axes.set_yticks(ticks)
    #             plt.grid(b=True, which='major', axis='both')
    #             plt.savefig((filepath + r'\Transfer curves\fet_{}_{}.png').format(i + 1, j + 1), format='png', dpi=150,
    #                         bbox_inches='tight')
    #             plt.clf()
    #             plt.close(fig)
    #
    # # этот цикл сохраняет графики всех транзисторов в лог скейле, нормализованные на баес
    #
    # for i in range(0, n):
    #     for j in range(0, m):
    #         maxt = np.zeros((num_sd_bias))
    #         mint = np.zeros((num_sd_bias))
    #         if status_array[i, j] != 3:
    #             fig, ax = plt.subplots(figsize=(6, 6))
    #             for z in range(0, num_sd_bias):
    #                 curr = np.zeros((m1))
    #                 for k in range(0, number_of_cycles):
    #                     curr += data[i, j, 2 + 2 * k, :, z]
    #                 curr /= number_of_cycles
    #                 plt.plot(data[i, j, 1, :, z], curr / data[i, j, 0, 0, z], label='{} V'.format(data[i, j, 0, 0, z]))
    #                 maxt[z] = max(curr / data[i, j, 0, 0, z])
    #                 mint[z] = min(curr / data[i, j, 0, 0, z])
    #             plt.title(chip + '\n' + measurement + '\n' + 'FET {}_{}'.format(i + 1, j + 1))
    #             plt.legend(loc=0)
    #             plt.yscale('symlog', linthreshy=10 ** (-13))
    #             plt.xlabel('Vg, V', fontsize=14)
    #             plt.ylabel('Conductance, S', fontsize=14)
    #             ax.yaxis.set_major_locator(locmax)
    #             ax.yaxis.set_minor_locator(locmin)
    #             ax.yaxis.set_minor_formatter(mpl.ticker.LogFormatter())
    #             ticks = np.array(list(ax.axes.get_yticks()))
    #             ticks = ticks[(abs(ticks) > 5 * 1e-14)]
    #             ticks = ticks[ticks < max(maxt)]
    #             ticks = ticks[ticks > min(mint)]
    #             ax.axes.set_yticks(ticks)
    #             plt.grid(b=True, which='major', axis='both')
    #             plt.savefig((filepath + r'\Normalized transfers\fet_{}_{}.png').format(i + 1, j + 1), format='png',
    #                         dpi=150, bbox_inches='tight')
    #             plt.clf()
    #             plt.close(fig)

    #     # мобильность, сучки

    max_mobility = np.zeros((n, m, num_sd_bias))
    max_mobility.fill(np.nan)
    C = 8.85 / 1e12 * 3.9 / (500 / 1e9) / 1e4
    for i in range(0, n):
        for j in range(0, m):
            if status_array[i, j] != 3:
                fig, ax = plt.subplots(figsize=(6, 6))
                for z in range(0, num_sd_bias):
                    curr = np.zeros((m1))
                    curr_diff = np.zeros((m1 - 1))
                    for k in range(0, number_of_cycles):
                        curr = data[i, j, 2 + 2 * k, :, z]
                    curr /= number_of_cycles
                    curr_diff = ((-np.diff(np.delete(curr, [max_backward_n]), n=1)) / np.diff(
                        np.delete(data[i, j, 1, :, z], [max_backward_n]), n=1))
                    curr_diff = length_array[i, j] / width / C * curr_diff / data[0, 0, 0, 0, z]
                    max_mobility[i, j, z] = np.max(curr_diff)
                    if status_array[i, j] >= 1:
                        max_mobility[i, j, :] = float('NaN')
                    plt.plot(np.delete(data[i, j, 1, :, z], [max_forward_p, max_backward_n]), curr_diff,
                             label='{} V'.format(data[i, j, 0, 0, z]))
                plt.title(chip + '\n' + measurement + '\n' + 'Mobility' + '\n' + 'FET {}_{}'.format(i + 1, j + 1))
                plt.legend(loc=0)
                plt.xlabel('Vg, V', fontsize=14)
                plt.ylabel('Mobility' + ' cm^2/(V*s)', fontsize=14)
                plt.grid(b=True, axis='both', which='major')
                plt.savefig((filepath + r'\Mobility graphs\fet_{}_{}.png').format(i + 1, j + 1), format='png', dpi=150,
                            bbox_inches='tight')
                plt.clf()
                plt.close(fig)

    for z in range(0, num_sd_bias):
        fig, ax = plt.subplots(figsize=(8, 8))
        df = pd.DataFrame(status_array)
        df.columns = range(1, m + 1)
        df.index = range(1, n + 1)
        sns.heatmap(df, linewidth=.3, ax=ax, cmap=cmap, vmin=1, vmax=3, cbar=False)
        df = pd.DataFrame(max_mobility[:, :, z])
        df.columns = range(1, m + 1)
        df.index = range(1, n + 1)
        sns.heatmap(df, linewidth=.3, annot=True, fmt=".2g", vmin=1e1,
                    vmax=1e3, ax=ax, cbar=True, norm=mpl.colors.LogNorm(), cmap='rainbow',
                    cbar_kws={'ticks': [1e1, 1e2, 1e3]})
        plt.title(chip + '\n' + measurement + '\n' + 'Max mobility' + '\n' + 'Bias = {} V'.format(data[0, 0, 0, 0, z]))
        fig.legend(handles=patches, loc=8, ncol=3, fontsize='small')
        plt.savefig((filepath + r'\Mobility heatmaps' + r'\Max mobility {} V.png'.format(data[0, 0, 0, 0, z])),
                    format='png', dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close(fig)

    # здесь строится зависимость on/off от длины канала

    # sns.set(style="ticks", rc={"lines.linewidth": 1})
    # if np.isnan(status_array).any() == True:
    #     for z in range(0, num_sd_bias):
    #         fig, ax = plt.subplots(figsize=(6, 6))
    #         df = pd.DataFrame(np.transpose(
    #             [np.ndarray.flatten(np.log10(on_off_ratio[:, :, z])), np.rint(np.ndarray.flatten(length_array[:n, :] * 1e4))]))
    #         df.columns = ('On/off ratio', 'Channel length, um')
    #         if (on_off_ratio == float('nan')).all() == False:
    #             sns.pointplot(x='Channel length, um', y='On/off ratio', color='green', data=df, ci='sd', errwidth=1,
    #                           capsize=0.5)
    #             sns.stripplot(x='Channel length, um', y='On/off ratio', data=df, jitter=False, color='orange')
    #             plt.title(chip + '\n' + measurement + '\n' + 'On/off ratio(Lch)' + '\n' + 'Bias = {} V'.format(
    #                 data[0, 0, 0, 0, z]))
    #             plt.xlabel('Lch, um', fontsize=12)
    #             plt.ylabel('Log10 on/off ratio', fontsize=12)
    #             fig.tight_layout(pad=1)
    #             plt.savefig((filepath + r'\Onoff dependence from Lch\On_off(Lch) {} V.png'.format(data[0, 0, 0, 0, z])),
    #                         format='png', dpi=150)
    #             plt.clf()
    #             plt.close(fig)
    #
    # # здесь строится зависимость Ropen от длины канала
    #
    # if np.isnan(status_array).any() == True:
    #     for z in range(0, num_sd_bias):
    #         fig, ax = plt.subplots(figsize=(6, 6))
    #         df = pd.DataFrame(np.transpose([np.ndarray.flatten(np.log10(open_state_resistance[:, :, z])),
    #                                         np.rint(np.ndarray.flatten(length_array[:n, :] * 1e4))]))
    #         df.columns = ('R open, Ohm', 'Channel length, um')
    #         if (on_off_ratio == float('nan')).all() == False:
    #             X = np.reshape(np.array(df['Channel length, um']), (-1, 1))
    #             y = np.reshape(np.array(df['R open, Ohm']), (-1, 1))
    #             #         estimator = LinearRegression().fit(X, y)
    #             sns.pointplot(x='Channel length, um', y='R open, Ohm', color='green', data=df, ci='sd', errwidth=1,
    #                           capsize=0.5)
    #             sns.stripplot(x='Channel length, um', y='R open, Ohm', data=df, jitter=False, color='orange')
    #             plt.title(
    #                 chip + '\n' + measurement + '\n' + 'R open(Lch)' + '\n' + 'Bias = {} V'.format(data[0, 0, 0, 0, z]))
    #             plt.xlabel('Lch, um', fontsize=12)
    #             plt.ylabel('R open, Ohm', fontsize=12)
    #             plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    #             fig.tight_layout(pad=1)
    #             plt.savefig((filepath + r'\Ropen dependence from Lch\R_open(Lch) {} V.png'.format(data[0, 0, 0, 0, z])),
    #                         format='png', dpi=150)
    #             fig.clf()
    #             plt.close(fig)
    #
    # i1 = 0
    # j1 = 0
    # status = np.array(status_array, dtype='str')
    # for i in range(0, n):
    #     for j in range(0, m):
    #         if status_array[i, j] == 1:
    #             status[i, j] = 'Non-conductive'
    #             i1 = i
    #             j1 = j
    #         if status_array[i, j] == 2:
    #             status[i, j] = 'Gate leakage'
    #             i1 = i
    #             j1 = j
    #         if status_array[i, j] == 3:
    #             status[i, j] = 'Bad/no data'
    #
    # for i in range(0, n):
    #     for j in range(0, m):
    #         if status[i, j] == 'nan':
    #             status[i, j] = 'Operable'
    #             i1 = i
    #             j1 = j

    # здесь делается csv с основными данными, а также строит гистограмки

    # for z in range(0, num_sd_bias):
    #     df = pd.DataFrame(np.transpose([np.array(np.repeat(chip, n * m)),
    #                                     np.array(np.repeat(measurement, n * m)),
    #                                     np.array(devices_index[:, 1], dtype='int'),
    #                                     np.array(devices_index[:, 0], dtype='int'),
    #                                     np.ndarray.flatten(status),
    #                                     np.repeat(data[0, 0, 0, 0, z], n * m),
    #                                     np.rint(np.ndarray.flatten(length_array[:n, :m] * 10 ** 4)),
    #                                     np.rint(np.ndarray.flatten(open_state_resistance[:, :, z])),
    #                                     np.rint(np.ndarray.flatten(closed_state_resistance[:, :, z])),
    #                                     np.rint(np.ndarray.flatten(on_off_ratio[:, :, z])),
    #                                     np.round(np.ndarray.flatten(max_mobility[:, :, z]), 3)]))
    #     df.columns = (
    #         'Chip name', 'Measurement', 'Column', 'Row', 'Status', 'Bias', 'Lch', 'R open', 'R closed', 'On/off',
    #         'Max mobility')
    #     fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    #     ax[0].hist(np.array(df['On/off'], dtype='float')[~np.isnan(np.array(df['On/off'], dtype='float'))],
    #                bins=np.logspace(np.log10(1e0), np.log10(1e6), 13), color='b')
    #     ax[0].set_xscale('log')
    #     ax[0].xaxis.set_ticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    #     ax[0].set_xlabel('On/off ratio')
    #     ax[0].set_ylabel('Count')
    #     ax[1].hist(np.array(df['R open'], dtype='float')[~np.isnan(np.array(df['R open'], dtype='float'))],
    #                bins=np.logspace(np.log10(1e4), np.log10(1e9), 11), color='m')
    #     ax[1].set_xscale('log')
    #     ax[1].xaxis.set_ticks([1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
    #     ax[1].set_xlabel('Open state resistance')
    #     ax[2].hist(np.array(df['Max mobility'], dtype='float')[~np.isnan(np.array(df['Max mobility'], dtype='float'))],
    #                bins=np.logspace(np.log10(1e-2), np.log10(1e3), 16), color='g')
    #     ax[2].set_xscale('log')
    #     ax[2].xaxis.set_ticks([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    #     ax[2].set_xlabel('Max mobility')
    #     # ax[0].set_xlim(left = 10**(np.floor(np.log10(np.min(np.array(df['On/off'], dtype = 'float')[~np.isnan(
    #     # np.array(df['On/off'], dtype = 'float'))])))), right = 10**(np.ceil(np.log10(np.max(np.array(df['On/off'],
    #     # dtype = 'float')[~np.isnan(np.array(df['On/off'], dtype = 'float'))]))))) ax[1].set_xlim(left = 10**(
    #     # np.floor(np.log10(np.min(np.array(df['R open'], dtype = 'float')[~np.isnan(np.array(df['R open'],
    #     # dtype = 'float'))])))), right = 10**(np.ceil(np.log10(np.max(np.array(df['R open'], dtype = 'float')[
    #     # ~np.isnan(np.array(df['R open'], dtype = 'float'))]))))) ax[2].set_xlim(left = 10**(np.floor(np.log10(
    #     # np.min(np.array(df['Max mobility'], dtype = 'float')[~np.isnan(np.array(df['Max mobility'],
    #     # dtype = 'float'))])))), right = 10**(np.ceil(np.log10(np.max(np.array(df['Max mobility'], dtype = 'float')[
    #     # ~np.isnan(np.array(df['Max mobility'], dtype = 'float'))])))))
    #     ax[0].minorticks_off()
    #     ax[1].minorticks_off()
    #     ax[2].minorticks_off()
    #     fig.suptitle(chip + '\n' + measurement + '\n' + 'Bias = {} V'.format(data[0, 0, 0, 0, z]))
    #     plt.savefig((filepath + r'\Histograms {} V.png'.format(data[0, 0, 0, 0, z])), format='png', dpi=300,
    #                 bbox_inches='tight')
    #     plt.close(fig)
    #     pd.DataFrame.to_csv(df, path_or_buf=filepath + r'\Results {} V.csv'.format(data[i1, j1, 0, 0, z]), sep=';',
    #                         header=True, index=False)

filepath_kernel = r'C:\Users\kpebe\OneDrive\Рабочий стол\Transistor measurements'
# chip_names = os.listdir(filepath_kernel)
# for chip in chip_names:
#     measurements = os.listdir(filepath_kernel + '\\' + chip)
#     for measurement in measurements:
#         filepath = filepath_kernel + '\\' + chip + '\\' + measurement
#         length_array = np.zeros((12, 4))
#         for j in range(0, 4):
#             length_array[:6, j] = 130 / 1e4 + 60 / 1e4 * j
#             length_array[6:, j] = 100 / 1e4 + 60 / 1e4 * j
#         if 'Canatu' in filepath:
#             length_array = np.zeros((36, 12))
#             for j in range(0,12):
#                 length_array[:18,j] = 45/1e4 + 30/1e4*j
#                 length_array[18:,j] = 30/1e4 + 30/1e4*j
#         if 'CO2' in filepath:
#             length_array = np.zeros((16, 5))
#             for j in range(0, 16):
#                 length_array[:8, j] = 90 / 1e4 + 60 / 1e4 * j
#                 length_array[8:, j] = 60 / 1e4 + 60 / 1e4 * j
#         print(filepath)
#         whole_calc(filepath,
#                    chip,
#                    measurement,
#                    length_array,
#                    150/1e4)


length_array = np.zeros((16, 5))
for j in range(0, 5):
    length_array[:8, j] = 90 / 1e4 + 60 / 1e4 * j
    length_array[8:, j] = 60 / 1e4 + 60 / 1e4 * j
path = r'C:\Users\kpebe\OneDrive\Рабочий стол\Transistor measurements' + r'\\' + r'18.12.19_N2O(30%, 600C)_2.2lpm(0.015CO2)_300sec' + r'\\' + 'First measurement'


whole_calc(filepath = path, chip = r'18.12.19_N2O(30%, 600C)_2.2lpm(0.015CO2)_300sec',
           measurement='First measurement', length_array=length_array, width=150/1e4)