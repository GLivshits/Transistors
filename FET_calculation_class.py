import os
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

class FETs_calculation(object):

    def __init__(self, measurement_type, columns, rows, filepath_kernel,
                 chip_name, measurement, num_sd_bias):

        self.measurement_type = measurement_type
        self.columns = columns
        self.rows = rows
        self.filepath_kernel = filepath_kernel + r'\\' + chip_name + r'\\' + measurement
        self.chip_name = chip_name
        self.measurement = measurement
        self.num_sd_bias = num_sd_bias
        self.status_array = np.full((self.rows, self.columns), float('NaN'))

        self.patches = [mpl.patches.Patch(color='gray', label='Non-conductive'),
                        mpl.patches.Patch(color='black', label='Gate leakage'),
                        mpl.patches.Patch(color='rosybrown', label='No data')]

        self.status_cmap = mpl.colors.ListedColormap(['gray', 'black', 'rosybrown'])

    def params_calc(self):

        # инициализируем массивы параметров

        self.on_off_ratio = np.full([self.rows, self.columns, self.num_sd_bias, 4], float('NaN'))
        self.on_state_resistance = np.full([self.rows, self.columns, self.num_sd_bias, 4], float('NaN'))
        self.off_state_resistance = np.full([self.rows, self.columns, self.num_sd_bias, 2], float('NaN'))
        self.max_currents = np.zeros((self.rows, self.columns, self.num_sd_bias, 4))
        self.min_currents = np.zeros((self.rows, self.columns, self.num_sd_bias, 2))
        self.branches_names = ['p-type forward', 'n-type forward', 'p-type backward', 'n-type backward']
        self.sd_biases = np.zeros((self.num_sd_bias))

        # в этом цикле считаются параметры

        for i in range(self.rows):

            for j in range(self.columns):

                try:

                    df = pd.read_csv(self.filepath_kernel + r'\\data\\fet_{}_{}.csv'.format(i + 1, j + 1),
                                     engine='python',
                                     sep=',')
                    # эти переменные содержат информацию о числе измерений, количетсве циклов и т.п.
                    df.columns = np.arange(0, df.shape[1], 1)
                    m1 = int(np.rint(df.shape[0] / self.num_sd_bias))
                    n1 = df.shape[1]
                    number_of_cycles = int((n1 - 2) / 2)
                    max_forward_p = int(0)
                    max_backward_p = int(m1-1)
                    max_forward_n = int((m1-2)/2)
                    max_backward_n = int((m1-2)/2 + 1)
                    self.sd_biases = np.array(df[0].unique())

                    gate_current = np.zeros((m1 * self.num_sd_bias))
                    gate_current[:] = np.sum(np.array(df)[:, 3::2], axis = 1) / number_of_cycles

                    # в этом цикле усреднются токи по циклам для каждого байеса и достаются максимальные токи
                    for k in range(self.num_sd_bias):

                        single_transistor_currents = np.zeros((m1))
                        single_transistor_currents[:] = np.sum(np.array(df)[k*m1:(k+1)*m1, 2::2], axis = 1)
                        single_transistor_currents[:] /= number_of_cycles

                        single_transistor_currents[abs(single_transistor_currents) <= 1e-12] = 1e-12 * np.sign(self.sd_biases[k])

                        if df.iloc[0, 0] < 0 or df.iloc[int(m1/2), 0] > 0:

                            self.max_currents[i, j, k, :] = single_transistor_currents[[max_forward_p,
                                                                                        max_forward_n,
                                                                                        max_backward_p,
                                                                                        max_backward_n]]

                        else:

                            self.max_currents[i, j, k, :] = single_transistor_currents[[max_forward_n,
                                                                                        max_forward_p,
                                                                                        max_backward_n,
                                                                                        max_backward_p]]

                        # if abs(self.max_currents[i, j, k, 0]) < abs(self.max_currents[i, j, k, 1]):
                        #
                        #     u = self.max_currents[i, j, k, 0]
                        #     self.max_currents[i, j, k, 0] = self.max_currents[i, j, k, 1]
                        #     self.max_currents[i, j, k, 1] = u
                        #
                        #     u = self.max_currents[i, j, k, 2]
                        #     self.max_currents[i, j, k, 2] = self.max_currents[i, j, k, 3]
                        #     self.max_currents[i, j, k, 3] = u
                        # это минимальные токи
                        min_forward = np.amin(abs(single_transistor_currents[:max_forward_n]))
                        min_backward = np.amin(abs(single_transistor_currents[max_backward_n:]))
                        self.min_currents[i, j, k, :] = [min_forward * np.sign(self.sd_biases[k]),
                                                         min_backward * np.sign(self.sd_biases[k])]
                        # здесь задаются on/off, Ron и Roff
                        self.on_off_ratio[i, j, k, :] = self.max_currents[i, j, k, :] / [self.min_currents[i, j, k, 0],
                                                                                         self.min_currents[i, j, k, 0],
                                                                                         self.min_currents[i, j, k, 1],
                                                                                         self.min_currents[i, j, k, 1]]

                        self.on_state_resistance[i, j, k, :] = abs(self.sd_biases[k] / self.max_currents[i, j, k, :])

                        self.off_state_resistance[i, j, k, :] = abs(self.sd_biases[k] / self.min_currents[i, j, k, :])

                        if np.mean(abs(self.max_currents[i, j, k, :])) < 10 ** (-9):  # условие для непроводящих

                            self.on_off_ratio[i, j, k, :] = float('NaN')
                            self.on_state_resistance[i, j, k, :] = float('NaN')
                            self.off_state_resistance[i, j, k, :] = float('NaN')
                            self. status_array[i, j] = 1

                        if np.max(abs(gate_current)) > 1e-8:  # условие для пробитого гейта

                            self.on_off_ratio[i, j, k, :] = float('NaN')
                            self.on_state_resistance[i, j, k, :] = float('NaN')
                            self.status_array[i, j] = 2

                        if np.mean(abs(self.max_currents[i, j, k, :])) > 1e2 \
                            or np.mean(abs(self.min_currents[i, j, k, :])) > 1e2 \
                            or np.isnan(np.array(df)).any():  # Данные говно

                            self.on_off_ratio[i, j, k, :] = float('NaN')
                            self.on_state_resistance[i, j, k, :] = float('NaN')
                            self.status_array[i, j] = 3

                except FileNotFoundError:

                    self.status_array[i,j] = 3
                    pass

    def heatmaps(self):

        os.makedirs(self.filepath_kernel + r'\Onoff heatmaps', exist_ok=True)
        os.makedirs(self.filepath_kernel + r'\Ropen and Rclosed heatmaps', exist_ok=True)

        direction = ['forward', 'backward']

        for i in range(4):

            if np.mean(self.on_off_ratio[:, :, :, i][~np.isnan(self.on_off_ratio[:, :, :, i])]) > 2:

                for k in range(0, self.num_sd_bias):

                    fig, ax = plt.subplots(figsize = (8, 8))

                    sns.heatmap(self.status_array, linewidth = .3, ax = ax,
                                cmap = self.status_cmap, vmin = 1, vmax = 3, cbar = False,
                                xticklabels = np.arange(1, self.status_array.shape[1] + 1, 1),
                                yticklabels = np.arange(1, self.status_array.shape[0] + 1, 1))
                    sns.heatmap(self.on_off_ratio[:, :, k, i], linewidth = .5, ax = ax,
                                vmin = 1, vmax = 10 ** 6, cbar=True,
                                norm = mpl.colors.LogNorm(), cmap = 'rainbow',
                                cbar_kws = {'ticks': [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]},
                                xticklabels=np.arange(1, self.status_array.shape[1] + 1, 1),
                                yticklabels=np.arange(1, self.status_array.shape[0] + 1, 1))

                    plt.title(self.chip_name + '\n' + self.measurement + '\n'
                              + '$I_{ON}/I_{OFF}$' + ', bias = {} V'.format(self.sd_biases[k])
                              + '\n' + '{}'.format(self.branches_names[i]))
                    fig.legend(handles = self.patches, loc = 8, ncol = 3, fontsize = 'small')
                    plt.savefig((self.filepath_kernel + r'\Onoff heatmaps' + r'\On off ratio {} V {}.png'.format(
                                 self.sd_biases[k], self.branches_names[i])),
                                 dpi = 300, bbox_inches = 'tight', format = 'png')
                    plt.clf()
                    plt.close(fig)

                    fig, ax = plt.subplots(figsize=(8, 8))
                    sns.heatmap(self.status_array, linewidth=.3, ax=ax,
                                cmap=self.status_cmap, vmin=1, vmax=3, cbar=False,
                                xticklabels=np.arange(1, self.status_array.shape[1] + 1, 1),
                                yticklabels=np.arange(1, self.status_array.shape[0] + 1, 1))
                    sns.heatmap(self.on_state_resistance[:, :, k, i], linewidth=.5, ax=ax,
                                vmin=1e3, vmax=1e8, cbar=True,
                                norm=mpl.colors.LogNorm(), cmap='rainbow',
                                cbar_kws={'ticks': [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]},
                                xticklabels=np.arange(1, self.status_array.shape[1] + 1, 1),
                                yticklabels=np.arange(1, self.status_array.shape[0] + 1, 1))
                    plt.title(
                        self.chip_name + '\n' + self.measurement + '\n' + '$R_{ON}$,' + ' bias = {} V'.format(self.sd_biases[k])
                        + '\n' + '{}'.format(self.branches_names[i]))
                    fig.legend(handles=self.patches, loc=8, ncol=3, fontsize='small')
                    plt.savefig((self.filepath_kernel + r'\Ropen and Rclosed heatmaps' + r'\Ropen {} V {}.png'.format(
                        self.sd_biases[k], self.branches_names[i])),
                                dpi=300, bbox_inches='tight', format='png')
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

        for i in range(self.rows):

            for j in range(self.columns):

                if np.isnan(self.status_array[i,j]) == True:

                    maxt = np.zeros((self.num_sd_bias))
                    mint = np.zeros((self.num_sd_bias))

                    df = pd.read_csv(self.filepath_kernel + r'\\data\\fet_{}_{}.csv'.format(i + 1, j + 1),
                                     engine='python',
                                     sep=',')

                    df.columns = np.arange(0, df.shape[1], 1)
                    m1 = int(np.rint(df.shape[0] / self.num_sd_bias))
                    n1 = df.shape[1]
                    number_of_cycles = int((n1 - 2) / 2)

                    fig, ax = plt.subplots(figsize=(6, 6))

                    for k in range(self.num_sd_bias):

                        curr = np.zeros((m1))
                        curr[:] = np.sum(np.array(df)[k * m1:(k + 1) * m1, 2::2], axis=1)
                        curr[:] /= number_of_cycles

                        plt.plot(df[1].iloc[0:m1], curr*np.sign(self.sd_biases[k]),
                                 label='{} V'.format(df[0].iloc[k*m1]))

                        maxt[k] = max(curr*np.sign(self.sd_biases[k]))
                        mint[k] = min(curr*np.sign(self.sd_biases[k]))

                    plt.title(self.chip_name + '\n' + self.measurement + '\n' + 'FET {}_{}'.format(i + 1, j + 1))
                    plt.legend(loc=0)

                    plt.yscale('symlog', linthreshy=10 ** (-13))

                    if self.measurement_type == 'Isd-Vg':

                        plt.xlabel('$V_{G}$, V', fontsize=14)

                    else:

                        plt.xlabel('$V_{SD}$, V', fontsize=14)

                    plt.ylabel('$I_{SD}$, A', fontsize=14)

                    ax.yaxis.set_major_locator(locmax)
                    ax.yaxis.set_minor_locator(locmin)
                    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

                    ticks = np.array(list(ax.axes.get_yticks()))
                    ticks = ticks[(abs(ticks) > 5 * 1e-14) | (ticks == 0)]
                    ticks = ticks[ticks < max(maxt)]
                    ticks = ticks[ticks > min(mint)]
                    ax.axes.set_yticks(ticks)

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

                    fig, ax = plt.subplots(figsize=(6, 6))

                    for k in range(self.num_sd_bias):
                        curr = np.zeros((m1))
                        curr[:] = np.sum(np.array(df)[k * m1:(k + 1) * m1, 2::2], axis=1)
                        curr[:] /= number_of_cycles

                        plt.plot(df[1].iloc[0:m1], curr * np.sign(self.sd_biases[k]),
                                 label='{} V'.format(df[0].iloc[k * m1]))

                    plt.title(self.chip_name + '\n' + self.measurement + '\n' + 'FET {}_{}'.format(i + 1, j + 1))
                    plt.legend(loc=0)

                    if self.measurement_type == 'Isd-Vg':

                        plt.xlabel('$V_{G}$, V', fontsize=14)

                    else:

                        plt.xlabel('$V_{SD}$, V', fontsize=14)

                    plt.ylabel('$I_{SD}$, A', fontsize=14)

                    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

                    # plt.grid(b=True, which='major', axis='both')

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

    def mobility_calc(self, length_array, width, C, plot_graphs = True, plot_heatmaps = True):

        os.makedirs(self.filepath_kernel + r'\Mobility graphs', exist_ok=True)
        os.makedirs(self.filepath_kernel + r'\Mobility heatmaps', exist_ok=True)

        self.max_mobility = np.full((self.rows, self.columns, self.num_sd_bias), float('NaN'))
        self.length_array = length_array
        self.width = width

        for i in range(self.rows):

            for j in range(self.columns):

                if np.isnan(self.status_array[i,j]) == True:

                    df = pd.read_csv(self.filepath_kernel + r'\\data\\fet_{}_{}.csv'.format(i + 1, j + 1),
                                     engine='python',
                                     sep=',')


                    df.columns = np.arange(0, df.shape[1], 1)
                    m1 = int(np.rint(df.shape[0] / self.num_sd_bias))
                    n1 = df.shape[1]
                    number_of_cycles = int((n1 - 2) / 2)
                    max_forward_p = int(0)
                    max_backward_p = int(m1 - 1)
                    max_forward_n = int((m1 - 2) / 2)
                    max_backward_n = int((m1 - 2) / 2 + 1)
                    self.sd_biases = np.array(df[0].unique())
                    curr = np.zeros((m1, self.num_sd_bias))
                    curr_diff = np.zeros((m1 - 2, self.num_sd_bias))

                    for k in range(self.num_sd_bias):

                        curr[:, k] = np.sum(np.array(df)[k*m1:(k+1)*m1, 2::2], axis = 1)
                        curr[:, k] /= number_of_cycles

                        curr_diff[:, k] = -np.diff(np.delete(curr[:, k], max_backward_n), n=1) /\
                                    np.diff(np.delete(np.array(df[1].iloc[k*m1:(k+1)*m1]), max_backward_n), n=1)
                        # curr_diff[:, k] = self.length_array[i, j] / self.width / self.capacitance * curr_diff[:, k] / self.sd_biases[k]

                        self.max_mobility[i, j, k] = np.max(curr_diff)

                        if self.status_array[i, j] >= 1:

                            self.max_mobility[i, j, :] = float('NaN')

                    if plot_graphs:

                        fig, ax = plt.subplots(figsize=(6, 6))

                        for k in range(self.num_sd_bias):

                            plt.plot(np.delete(np.array(df[1].iloc[k*m1:(k+1)*m1]), [max_forward_p, max_backward_n]),
                                 curr_diff[:, k] * 1e6,
                                 label='{} V'.format(self.sd_biases[k]))

                        plt.title(self.chip_name+ '\n' + self.measurement + '\n' + 'Transconductance'
                                  + '\n' + 'FET {}_{}'.format(i + 1, j + 1))

                        plt.legend(loc=0)

                        plt.xlabel('$V_{G}$, V', fontsize=14)
                        plt.ylabel('Transconductance, ' + '\u03bcA/V', fontsize=14)

                        # plt.grid(b=True, axis='both', which='major')

                        plt.savefig((self.filepath_kernel + r'\Mobility graphs\fet_{}_{}.png').format(i + 1, j + 1),
                                    format='png',
                                    dpi=150,
                                    bbox_inches='tight')

                        plt.clf()
                        plt.close(fig)

        if plot_heatmaps:

            for k in range(0, self.num_sd_bias):

                fig, ax = plt.subplots(figsize=(8, 8))

                df = pd.DataFrame(self.status_array)

                df.columns = range(1, self.columns + 1)
                df.index = range(1, self.rows + 1)

                sns.heatmap(df, linewidth=.3, ax=ax, cmap = self.status_cmap, vmin=1, vmax=3, cbar=False)

                df = pd.DataFrame(self.max_mobility[:, :, k])

                df.columns = range(1, self.columns + 1)
                df.index = range(1, self.rows + 1)

                sns.heatmap(df, linewidth=.3, vmin=1e0, annot = True, fmt = '.1g',
                            vmax=1e3, ax=ax, cbar=True, norm=mpl.colors.LogNorm(), cmap='rainbow',
                            cbar_kws={'ticks': [1e1, 1e2, 1e3]})

                plt.title(self.chip_name + '\n' + self.measurement + '\n' + 'Max mobility'
                          + '\n' + 'Bias = {} V'.format(self.sd_biases[k]))

                fig.legend(handles=self.patches, loc=8, ncol=3, fontsize='small')

                plt.savefig((self.filepath_kernel + r'\Mobility heatmaps'
                             + r'\Max mobility {} V.png'.format(self.sd_biases[k])),
                            format='png', dpi=300, bbox_inches='tight')

                plt.clf()
                plt.close(fig)

    def lch_dependence(self):

        sns.set(style="ticks", rc={"lines.linewidth": 1})

        if np.isnan(self.status_array).any() == True:

            for z in range(self.num_sd_bias):

                fig, ax = plt.subplots(figsize=(6, 6))

                df = pd.DataFrame(np.transpose([np.ndarray.flatten((self.open_state_resistance[:, :, z, 0])),
                                                np.ndarray.flatten((self.closed_state_resistance[:, :, z, 0])),
                                                np.ndarray.flatten((self.on_off_ratio[:, :, z, 0])),
                                                np.ndarray.flatten(self.max_mobility[:, :, z]),
                                                np.rint(np.ndarray.flatten(self.length_array[:, :] * 1e4))]))
                df.dropna(inplace = True, how = 'any')
                df.columns = ('R_open', 'R_closed', 'On/off', 'Max_mob', 'Lch')

                if (self.on_off_ratio == float('nan')).all() == False:

                    # sns.scatterplot(x='R_open', y='R_closed', hue = 'Lch', data=df)

                    # data['R open'] = data['R open']/data['Lch']
                    # data['R closed'] = data['R closed']/data['Lch']

                    g = sns.pairplot(df, hue='Lch', diag_kind='hist', palette='rainbow',
                                     vars=['R_open', 'R_closed', 'On/off', 'Max_mob'],
                                     plot_kws={'size': 15, 'alpha': 0.85}, diag_kws={'alpha': 0.4, 'log': True,
                                                                                     'bins': np.logspace(np.log10(1e-3),
                                                                                                         np.log10(1e12),
                                                                                                         80)})

                    k = ['R_open', 'R_closed', 'On/off', 'Max_mob']

                    for i in range(0, g.axes.shape[0]):

                        for j in range(0, g.axes.shape[1]):

                            if i != j:

                                g.axes[i, j].set_xscale('log')
                                g.axes[i, j].set_yscale('log')

                                g.axes[i, j].set_xlim([10 ** np.floor(np.log10(np.min(df[k[j]]))),
                                                       10 ** np.ceil(np.log10(np.max(df[k[j]])))])
                                g.axes[i, j].set_ylim([10 ** np.floor(np.log10(np.min(df[k[i]]))),
                                                       10 ** np.ceil(np.log10(np.max(df[k[i]])))])

                            else:

                                g.axes[i, j].set_xscale('log')
                                g.axes[i, j].set_xlim([10 ** np.floor(np.log10(np.min(df[k[j]]))),
                                                       10 ** np.ceil(np.log10(np.max(df[k[j]])))])

                    # plt.savefig('SuperPuper.png', dpi=600, format='png')

                    # sns.stripplot(x='Lch', y='R_open', data=df, jitter=False, color='orange')

                    # plt.title(self.chip_name + '\n' + self.measurement + '\n' + 'On/off ratio(Lch)' + '\n' + 'Bias = {} V'.format(
                    #     self.sd_biases[z]))

                    # plt.xlabel('Lch, um', fontsize=14)
                    # plt.ylabel('Log10 on/off ratio', fontsize=14)

                    fig.tight_layout(pad=1)

                    plt.savefig(
                        (self.filepath_kernel + r'\Pairplot {}V.png'.format(self.sd_biases[z])),
                        format='png', dpi=300)

                    plt.clf()
                    plt.close(fig)
                    #
                    # sns.stripplot(x='Lch', y='R_closed', data=df, jitter=False, color='orange')
                    #
                    # plt.title(self.chip_name + '\n' + self.measurement + '\n' + 'On/off ratio(Lch)' + '\n' + 'Bias = {} V'.format(
                    #     self.sd_biases[z]))
                    #
                    # # plt.xlabel('Lch, um', fontsize=14)
                    # # plt.ylabel('Log10 on/off ratio', fontsize=14)
                    #
                    # fig.tight_layout(pad=1)
                    #
                    # plt.savefig(
                    #     (self.filepath_kernel + r'\R_closed(Lch) {} V.png'.format(self.sd_biases[z])),
                    #     format='png', dpi=150)
                    #
                    # plt.clf()
                    # plt.close(fig)

    def export_data(self):

        status = np.array(self.status_array, dtype='str')

        for i in range(self.rows):
            for j in range(self.columns):

                if self.status_array[i, j] == 1:
                    status[i, j] = 'Non-conductive'

                if self.status_array[i, j] == 2:
                    status[i, j] = 'Gate leakage'

                if self.status_array[i, j] == 3:
                    status[i, j] = 'Bad/no data'

        for i in range(self.rows):
            for j in range(self.columns):

                if status[i, j] == 'nan':

                    status[i, j] = 'Operable'

        for z in range(0, self.num_sd_bias):

            df = pd.DataFrame(np.array([
                              np.array(np.repeat(self.chip_name, self.rows * self.columns)),
                              np.array(np.repeat(self.measurement, self.rows * self.columns)),
                              np.ndarray.flatten(status),
                              np.repeat(self.sd_biases[z], self.rows * self.columns),
                              np.rint(np.ndarray.flatten(self.length_array * 10 ** 4)),
                              np.rint(np.ndarray.flatten(self.on_state_resistance[:, :, z, 0])),
                              np.rint(np.ndarray.flatten(self.off_state_resistance[:, :, z, 0])),
                              np.rint(np.ndarray.flatten(self.on_off_ratio[:, :, z, 0])),
                              np.round(np.ndarray.flatten(self.max_mobility[:, :, z]), 3)]).T)

            df.columns = ['Chip name', 'Measurement', 'Status', 'Bias',
                          'Lch', 'R open', 'R closed', 'On/off', 'Max mobility']

            pd.DataFrame.to_csv(df, path_or_buf = self.filepath_kernel + r'\Results {} V.csv'.format(self.sd_biases[z]),
                                sep = ',', header = True, index = False)
