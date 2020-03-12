import os
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

class FETs_calculation(object):

    def __init__(self, measurement_type, columns, rows, filepath_kernel,
                 chip_name, length_array, width, t_ox, measurement):

        self.measurement_type = measurement_type
        self.columns = columns
        self.rows = rows
        self.filepath_kernel = filepath_kernel + r'\\' + chip_name + r'\\' + measurement
        self.chip_name = chip_name
        self.measurement = measurement
        self.num_sd_bias = 0
        self.length_array = length_array
        self.width = width
        self.t_ox = t_ox
        self.status_array = np.full((self.rows, self.columns), float('NaN'))
        # os.makedirs(self.filepath_kernel + r'\Onoff dependence from Lch', exist_ok=True)
        # os.makedirs(self.filepath_kernel + r'\Ropen dependence from Lch', exist_ok=True)

        self.patches = [mpl.patches.Patch(color='gray', label='Non-conductive'),
                        mpl.patches.Patch(color='black', label='Gate leakage'),
                        mpl.patches.Patch(color='rosybrown', label='No data')]

        self.status_cmap = mpl.colors.ListedColormap(['gray', 'black', 'rosybrown'])

    def params_calc(self):

        for i in range(0, self.rows):

            for j in range(0, self.columns):

                try:

                    df = pd.read_csv(self.filepath_kernel + r'\\data\\fet_{}_{}.csv'.format(i + 1, j + 1),
                                     engine='python',
                                     sep=',')

                    df.columns = np.arange(0, df.shape[1], 1)
                    self.num_sd_bias = int(df[0].nunique())

                except:

                    pass

                if self.num_sd_bias != 0:

                    break

        self.on_off_ratio = np.full([self.rows, self.columns, self.num_sd_bias, 4], float('NaN'))
        self.open_state_resistance = np.full([self.rows, self.columns, self.num_sd_bias, 4], float('NaN'))
        self.closed_state_resistance = np.full([self.rows, self.columns, self.num_sd_bias, 2], float('NaN'))
        self.max_currents = np.zeros((self.rows, self.columns, self.num_sd_bias, 4))
        self.min_currents = np.zeros((self.rows, self.columns, self.num_sd_bias, 2))
        self.branches_names = ['p-type forward', 'n-type forward', 'p-type backward', 'n-type backward']
        self.sd_biases = np.zeros((self.num_sd_bias))

        for i in range(0, self.rows):

            for j in range(0, self.columns):

                try:

                    df = pd.read_csv(self.filepath_kernel + r'\\data\\fet_{}_{}.csv'.format(i + 1, j + 1),
                                     engine='python',
                                     sep=',')

                    df.columns = np.arange(0, df.shape[1], 1)
                    m1 = int(df.shape[0] / self.num_sd_bias)
                    n1 = df.shape[1]
                    number_of_cycles = int((n1 - 2) / 2)
                    max_forward_p = int(0)
                    max_backward_p = int(m1-1)
                    max_forward_n = int((m1-2)/2)
                    max_backward_n = int((m1-2)/2 + 1)
                    self.sd_biases = np.array(df[0].unique())

                    for k in range(0, self.num_sd_bias):

                        single_transistor_currents = np.zeros((m1, number_of_cycles))
                        gate_current = np.zeros((m1 * self.num_sd_bias))

                        for l in range(0, number_of_cycles):

                            single_transistor_currents[:,l] += df[2+2*l].iloc[k*m1:(k+1)*m1]/number_of_cycles
                            gate_current[:] += df[3+2*l]/number_of_cycles

                            for q in range(0, m1):

                                if single_transistor_currents[q, l] * np.sign(self.sd_biases[k]) < 8 / 1e13:
                                    single_transistor_currents[q, l] = 8 / 1e13 * np.sign(self.sd_biases[k])

                        self.max_currents[i, j, k, 0] = np.mean(single_transistor_currents[max_forward_p, :])
                        self.max_currents[i, j, k, 1] = np.mean(single_transistor_currents[max_forward_n, :])
                        self.max_currents[i, j, k, 2] = np.mean(single_transistor_currents[max_backward_p, :])
                        self.max_currents[i, j, k, 3] = np.mean(single_transistor_currents[max_backward_n, :])

                        if abs(self.max_currents[i, j, k, 0]) < abs(self.max_currents[i, j, k, 1]):

                            u = self.max_currents[i, j, k, 0]
                            self.max_currents[i, j, k, 0] = self.max_currents[i, j, k, 1]
                            self.max_currents[i, j, k, 1] = u

                            u = self.max_currents[i, j, k, 2]
                            self.max_currents[i, j, k, 2] = self.max_currents[i, j, k, 3]
                            self.max_currents[i, j, k, 3] = u

                        min_forward = 0
                        min_backward = 0

                        for l in range(0, number_of_cycles):

                            min_forward += np.amin(single_transistor_currents[:max_forward_n, l] * np.sign(self.sd_biases[k])) / number_of_cycles
                            min_backward += np.amin(single_transistor_currents[max_backward_n:, l] * np.sign(self.sd_biases[k])) / number_of_cycles

                        self.min_currents[i, j, k, 0] = min_forward * np.sign(self.sd_biases[k])
                        self.min_currents[i, j, k, 1] = min_backward * np.sign(self.sd_biases[k])

                        self.on_off_ratio[i, j, k, 0] = self.max_currents[i, j, k, 0] / self.min_currents[i, j, k, 0]
                        self.on_off_ratio[i, j, k, 1] = self.max_currents[i, j, k, 1] / self.min_currents[i, j, k, 0]
                        self.on_off_ratio[i, j, k, 2] = self.max_currents[i, j, k, 2] / self.min_currents[i, j, k, 1]
                        self.on_off_ratio[i, j, k, 3] = self.max_currents[i, j, k, 3] / self.min_currents[i, j, k, 1]

                        self.open_state_resistance[i, j, k, 0] = abs(self.sd_biases[k] / self.max_currents[i, j, k, 0])
                        self.open_state_resistance[i, j, k, 1] = abs(self.sd_biases[k] / self.max_currents[i, j, k, 1])
                        self.open_state_resistance[i, j, k, 2] = abs(self.sd_biases[k] / self.max_currents[i, j, k, 2])
                        self.open_state_resistance[i, j, k, 3] = abs(self.sd_biases[k] / self.max_currents[i, j, k, 3])

                        self.closed_state_resistance[i, j, k, 0] = abs(self.sd_biases[k] / self.min_currents[i, j, k, 0])
                        self.closed_state_resistance[i, j, k, 1] = abs(self.sd_biases[k] / self.min_currents[i, j, k, 1])

                        if np.mean(abs(self.max_currents[i, j, k, :])) < 5* 10 ** (-10):  # условие для непроводящих

                            self.on_off_ratio[i, j, k, :] = float('NaN')
                            self.open_state_resistance[i, j, k, :] = float('NaN')
                            self.closed_state_resistance[i, j, k, :] = float('NaN')
                            self. status_array[i, j] = 1

                        if np.mean(abs(gate_current)) > 1e-8:  # условие для пробитого гейта

                            self.on_off_ratio[i, j, k, :] = float('NaN')
                            self.open_state_resistance[i, j, k, :] = float('NaN')
                            self.status_array[i, j] = 2

                        if np.mean(abs(self.max_currents[i, j, k, :])) > 1e2 \
                            or np.mean(abs(self.min_currents[i, j, k, :])) > 1e2 \
                            or np.isnan(np.array(df)).any():  # Данные говно

                            self.on_off_ratio[i, j, k, :] = float('NaN')
                            self.open_state_resistance[i, j, k, :] = float('NaN')
                            self.status_array[i, j] = 3

                except FileNotFoundError:

                    self.status_array[i,j] = 3
                    pass

    def heatmaps(self):

        os.makedirs(self.filepath_kernel + r'\Onoff heatmaps', exist_ok=True)
        os.makedirs(self.filepath_kernel + r'\Ropen and Rclosed heatmaps', exist_ok=True)

        direction = ['forward', 'backward']

        self.label_arr = list(self.length_array[0, :] * 1e4)

        for item in self.label_arr:

            item = str(item) + 'um'

        for i in range(0, 4):

            if np.mean(self.on_off_ratio[:, :, :, i][~np.isnan(self.on_off_ratio[:, :, :, i])]) > 2:

                for k in range(0, self.num_sd_bias):

                    fig, ax = plt.subplots(figsize = (8, 8))

                    df = pd.DataFrame(self.status_array)
                    df.columns = range(1, self.columns + 1)
                    df.index = range(1, self.rows + 1)

                    sns.heatmap(df, linewidth = .3, ax = ax, cmap = self.status_cmap, vmin = 1, vmax = 3, cbar = False)

                    df = pd.DataFrame(self.on_off_ratio[:, :, k, i])
                    df.columns = range(1, self.columns + 1)
                    df.index = range(1, self.rows + 1)

                    sns.heatmap(df, linewidth = .5, annot=True,
                                fmt = ".1g", ax = ax,
                                vmin = 1, vmax = 10 ** 6, cbar=True,
                                norm = mpl.colors.LogNorm(), cmap = 'rainbow',
                                cbar_kws = {'ticks': [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]},
                                xticklabels=self.label_arr, yticklabels=False)

                    plt.title(self.chip_name + '\n' + self.measurement + '\n'
                              + 'On/off, bias = {} V'.format(self.sd_biases[k])
                              + '\n' + '{}'.format(self.branches_names[i]))

                    fig.legend(handles = self.patches, loc=8, ncol=3, fontsize='small')

                    plt.savefig((self.filepath_kernel + r'\Onoff heatmaps' + r'\On off ratio {} V {}.png'.format(
                                 self.sd_biases[k], self.branches_names[i])),
                                 dpi = 300, bbox_inches = 'tight', format = 'png')

                    plt.clf()
                    plt.close(fig)

        # здесь строится карта сопротивлений транзисторов в открытом состоянии
        for i in range(0, 4):

            if np.mean(self.on_off_ratio[:, :, :, i][~np.isnan(self.on_off_ratio[:, :, :, i])]) > 2:

                for k in range(0, self.num_sd_bias):

                    fig, ax = plt.subplots(figsize=(8, 8))

                    df = pd.DataFrame(self.status_array)
                    df.columns = range(1, self.columns + 1)
                    df.index = range(1, self.rows + 1)

                    sns.heatmap(df, linewidth=.3, ax=ax, cmap=self.status_cmap, vmin=1, vmax=3, cbar=False)

                    df = pd.DataFrame(self.open_state_resistance[:, :, k, i])
                    df.columns = range(1, self.columns + 1)
                    df.index = range(1, self.rows + 1)

                    sns.heatmap(df, linewidth=.5, annot=True,
                                fmt=".1g", ax=ax,
                                vmin=1e3, vmax=1e8, cbar=True,
                                norm=mpl.colors.LogNorm(), cmap='rainbow',
                                cbar_kws={'ticks': [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]},
                                xticklabels=self.label_arr, yticklabels=False)

                    plt.title(
                        self.chip_name + '\n' + self.measurement + '\n' + 'Ropen, bias = {} V'.format(self.sd_biases[k])
                        + '\n' + '{}'.format(self.branches_names[i]))

                    fig.legend(handles=self.patches, loc=8, ncol=3, fontsize='small')

                    plt.savefig((self.filepath_kernel + r'\Ropen and Rclosed heatmaps' + r'\Ropen {} V {}.png'.format(
                                self.sd_biases[k], self.branches_names[i])),
                                dpi=300, bbox_inches='tight', format='png')

                    plt.clf()
                    plt.close(fig)

        # здесь строится карта сопротивлений транзисторов в закрытом состоянии
        for i in range(0, 2):
            for k in range(0, self.num_sd_bias):

                fig, ax = plt.subplots(figsize=(8, 8))

                df = pd.DataFrame(self.status_array)
                df.columns = range(1, self.columns + 1)
                df.index = range(1, self.rows + 1)

                sns.heatmap(df, linewidth=.3, ax=ax, cmap=self.status_cmap, vmin=1, vmax=3, cbar=False)

                df = pd.DataFrame(self.closed_state_resistance[:, :, k, i])
                df.columns = range(1, self.columns + 1)
                df.index = range(1, self.rows + 1)

                sns.heatmap(df, linewidth=.3, annot=True, fmt=".2g", vmin=10 ** 6,
                            vmax=10 ** 11, ax=ax, cbar=True, norm=mpl.colors.LogNorm(), cmap='rainbow',
                            cbar_kws={'ticks': [1e6, 1e7, 1e8, 1e9, 1e10, 1e11]},
                            xticklabels=self.label_arr, yticklabels=False)

                plt.title(self.chip_name + '\n' + self.measurement + '\n'
                          + 'Rclosed, bias = {} V {}'.format(self.sd_biases[k], direction[i]))

                fig.legend(handles=self.patches, loc=8, ncol=3, fontsize='small')

                plt.savefig((self.filepath_kernel + r'\Ropen and Rclosed heatmaps' +
                             r'\Rclosed {} V {}.png'.format(self.sd_biases[k], direction[i])),
                            format='png', dpi=300, bbox_inches='tight')

                plt.clf()
                plt.close(fig)

    def plot_graphs(self):

        if self.measurement_type == 'Isd-Vg':

            os.makedirs(self.filepath_kernel + r'\Transfer curves log scale', exist_ok=True)
            os.makedirs(self.filepath_kernel + r'\Transfer curves', exist_ok=True)

        else:

            os.makedirs(self.filepath_kernel + r'\IV log scale', exist_ok=True)
            os.makedirs(self.filepath_kernel + r'\IV', exist_ok=True)

        locmax = mpl.ticker.SymmetricalLogLocator(base=10.0, linthresh=1e-13, subs=(0.1, 1, 10, 100,))
        locmin = mpl.ticker.SymmetricalLogLocator(base=10.0, linthresh=1e-13, subs=(0.2, 0.4, 0.6, 0.8, 1,))

        for i in range(0, self.rows):

            for j in range(0, self.columns):

                if np.isnan(self.status_array[i,j]) == True:

                    maxt = np.zeros((self.num_sd_bias))
                    mint = np.zeros((self.num_sd_bias))

                    df = pd.read_csv(self.filepath_kernel + r'\\data\\fet_{}_{}.csv'.format(i + 1, j + 1),
                                     engine='python',
                                     sep=',')

                    df.columns = np.arange(0, df.shape[1], 1)

                    fig, ax = plt.subplots(figsize=(6, 6))

                    for k in range(0, self.num_sd_bias):

                        m1 = int(df.shape[0] / self.num_sd_bias)
                        n1 = df.shape[1]
                        number_of_cycles = int((n1 - 2) / 2)
                        curr = np.zeros((m1))

                        for l in range(0, number_of_cycles):

                            curr += df[2+2*l].iloc[k*m1:(k+1)*m1]/number_of_cycles

                        plt.plot(df[1].iloc[0:m1], curr,
                                 label='{} V'.format(df[0].iloc[k*m1]))

                        maxt[k] = max(curr)
                        mint[k] = min(curr)

                    plt.title(self.chip_name + '\n' + self.measurement + '\n' + 'FET {}_{}'.format(i + 1, j + 1))
                    plt.legend(loc=0)

                    plt.yscale('symlog', linthreshy=10 ** (-13))

                    if self.measurement_type == 'Isd-Vg':

                        plt.xlabel('Vg, V', fontsize=14)

                    else:

                        plt.xlabel('Vsd, V', fontsize=14)

                    plt.ylabel('Isd, A', fontsize=14)

                    ax.yaxis.set_major_locator(locmax)
                    ax.yaxis.set_minor_locator(locmin)
                    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

                    ticks = np.array(list(ax.axes.get_yticks()))
                    ticks = ticks[(abs(ticks) > 5 * 1e-14) | (ticks == 0)]
                    ticks = ticks[ticks < max(maxt)]
                    ticks = ticks[ticks > min(mint)]
                    ax.axes.set_yticks(ticks)

                    plt.grid(b=True, which='major', axis='both')

                    if self.measurement_type == 'Isd-Vg':

                        plt.savefig((self.filepath_kernel
                                     + r'\Transfer curves log scale\fet_{}_{}.png').format(i + 1, j + 1),
                                    format='png',
                                    dpi=150,
                                    bbox_inches='tight')

                    else:

                        plt.savefig(
                            (self.filepath_kernel + r'\IV log scale\fet_{}_{}.png').format(i + 1, j + 1),
                            format='png',
                            dpi=150,
                            bbox_inches='tight')

                    plt.clf()
                    plt.close(fig)

        # этот цикл сохраняет графики всех транзисторов в обычном скейле

        for i in range(0, self.rows):

            for j in range(0, self.columns):

                if np.isnan(self.status_array[i,j]) == True:

                    df = pd.read_csv(self.filepath_kernel + r'\\data\\fet_{}_{}.csv'.format(i + 1, j + 1),
                                     engine='python',
                                     sep=',')

                    df.columns = np.arange(0, df.shape[1], 1)

                    fig, ax = plt.subplots(figsize=(6, 6))

                    for k in range(0, self.num_sd_bias):
                        m1 = int(df.shape[0] / self.num_sd_bias)
                        n1 = df.shape[1]
                        number_of_cycles = int((n1 - 2) / 2)
                        curr = np.zeros((m1))

                        for l in range(0, number_of_cycles):
                            curr = df[2 + 2 * l].iloc[k * m1:(k + 1) * m1] / number_of_cycles

                        plt.plot(df[1].iloc[0:m1], curr,
                                 label='{} V'.format(df[0].iloc[k*m1]))

                    plt.title(self.chip_name + '\n' + self.measurement + '\n' + 'FET {}_{}'.format(i + 1, j + 1))
                    plt.legend(loc=0)

                    if self.measurement_type == 'Isd-Vg':

                        plt.xlabel('Vg, V', fontsize=14)

                    else:

                        plt.xlabel('Vsd, V', fontsize=14)

                    plt.ylabel('Isd, A', fontsize=14)

                    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

                    plt.grid(b=True, which='major', axis='both')

                    if self.measurement_type == 'Isd-Vg':

                        plt.savefig((self.filepath_kernel + r'\Transfer curves\fet_{}_{}.png').format(i + 1, j + 1),
                                format='png',
                                dpi=150,
                                bbox_inches='tight')

                    else:

                        plt.savefig((self.filepath_kernel + r'\IV\fet_{}_{}.png').format(i + 1, j + 1),
                                    format='png',
                                    dpi=150,
                                    bbox_inches='tight')

                    plt.clf()
                    plt.close(fig)

    def mobility_calc(self):

        os.makedirs(self.filepath_kernel + r'\Mobility graphs', exist_ok=True)
        os.makedirs(self.filepath_kernel + r'\Mobility heatmaps', exist_ok=True)

        max_mobility = np.full((self.rows, self.columns, self.num_sd_bias), float('NaN'))

        C = 8.85 / 1e12 * 3.9 / (self.t_ox / 1e9) / 1e4

        for i in range(0, self.rows):

            for j in range(0, self.columns):

                if np.isnan(self.status_array[i,j]) == True:

                    df = pd.read_csv(self.filepath_kernel + r'\\data\\fet_{}_{}.csv'.format(i + 1, j + 1),
                                     engine='python',
                                     sep=',')

                    df.columns = np.arange(0, df.shape[1], 1)

                    m1 = int(df.shape[0] / self.num_sd_bias)
                    n1 = df.shape[1]
                    number_of_cycles = int((n1 - 2) / 2)

                    max_forward_p = int(0)
                    max_backward_p = int(m1 - 1)
                    max_forward_n = int((m1 - 2) / 2)
                    max_backward_n = int((m1 - 2) / 2 + 1)
                    min_forward = 0
                    min_backward = 0

                    fig, ax = plt.subplots(figsize=(6, 6))

                    for k in range(0, self.num_sd_bias):

                        curr = np.zeros((m1))

                        for l in range(0, number_of_cycles):

                            curr += np.array(df[2+2*l].iloc[k*m1:(k+1)*m1]/number_of_cycles)

                        curr_diff = -np.diff(np.delete(curr, max_backward_n), n=1) /\
                                    np.diff(np.delete(np.array(df[1].iloc[k*m1:(k+1)*m1]), max_backward_n), n=1)
                        curr_diff = self.length_array[i, j] / self.width / C * curr_diff / self.sd_biases[k]

                        max_mobility[i, j, k] = np.max(curr_diff)

                        if self.status_array[i, j] >= 1:

                            max_mobility[i, j, :] = float('NaN')

                        plt.plot(np.delete(np.array(df[1].iloc[k*m1:(k+1)*m1]), [max_forward_p, max_backward_n]),
                                 curr_diff,
                                 label='{} V'.format(self.sd_biases[k]))

                    plt.title(self.chip_name+ '\n' + self.measurement + '\n' + 'Mobility'
                              + '\n' + 'FET {}_{}'.format(j + 1, i + 1))

                    plt.legend(loc=0)

                    plt.xlabel('Vg, V', fontsize=14)
                    plt.ylabel('Mobility' + ' cm^2/(V*s)', fontsize=14)

                    plt.grid(b=True, axis='both', which='major')

                    plt.savefig((self.filepath_kernel + r'\Mobility graphs\fet_{}_{}.png').format(i + 1, j + 1),
                                format='png',
                                dpi=150,
                                bbox_inches='tight')

                    plt.clf()
                    plt.close(fig)

        for k in range(0, self.num_sd_bias):

            fig, ax = plt.subplots(figsize=(8, 8))

            df = pd.DataFrame(self.status_array)

            df.columns = range(1, self.columns + 1)
            df.index = range(1, self.rows + 1)

            sns.heatmap(df, linewidth=.3, ax=ax, cmap = self.status_cmap, vmin=1, vmax=3, cbar=False)

            df = pd.DataFrame(max_mobility[:, :, k])

            df.columns = range(1, self.columns + 1)
            df.index = range(1, self.rows + 1)

            sns.heatmap(df, linewidth=.3, annot=True, fmt=".2g", vmin=1e1,
                        vmax=1e3, ax=ax, cbar=True, norm=mpl.colors.LogNorm(), cmap='rainbow',
                        cbar_kws={'ticks': [1e1, 1e2, 1e3]},
                        xticklabels=self.label_arr, yticklabels=False)

            plt.title(self.chip_name + '\n' + self.measurement + '\n' + 'Max mobility'
                      + '\n' + 'Bias = {} V'.format(self.sd_biases[k]))

            fig.legend(handles=self.patches, loc=8, ncol=3, fontsize='small')

            plt.savefig((self.filepath_kernel + r'\Mobility heatmaps'
                         + r'\Max mobility {} V.png'.format(self.sd_biases[k])),
                        format='png', dpi=300, bbox_inches='tight')

            plt.clf()
            plt.close(fig)
