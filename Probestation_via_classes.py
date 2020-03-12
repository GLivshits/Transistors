import os
import serial
import pyvisa
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import time


class Probestation_FETs(object):

    def __init__(self, measurement_type):

        self.measurement_type = measurement_type

    def turn_on_keysight(self):

        rm = pyvisa.ResourceManager(r'C:\\Windows\\System32\\visa64.dll')
        self.my_inst = rm.open_resource(r'GPIB0::17::INSTR')
        del self.my_inst.timeout

        self.my_inst.query('*IDN?')

        self.my_inst.write('*RST')

        self.my_inst.write('*SRE 59')

        self.my_inst.write('FMT 12,1')

        self.my_inst.write('CN 101')  # enables channel 101
        self.my_inst.write('CN 201')  # enables channel 201

        self.my_inst.write('MM 2,101,201')  # enables staircase sweep for channels 101 and 201

        self.my_inst.write('CMM 101,1')  # sets current measurement (1) for channel 101
        self.my_inst.write('CMM 201,1')  # sets current measurement(1) for channel 201

        self.my_inst.write('RI 101,0')  # sets auto-ranging (0) for channel 101
        self.my_inst.write('RI 201,0')  # sets auto-ranging (0) for channel 201

        self.my_inst.write('AAD 101,1;AZ 0;AIT 1, 2, 10')  # sets high resolution (1) for channel 101
        self.my_inst.write('AAD 201,1;AZ 0;AIT 1, 2, 10')  # sets high resolution (1) for channel 201

        self.my_inst.write('AZ 0')  # ADC zero function disabled (0)

        self.my_inst.write('AIT 1, 2, 10')

        self.my_inst.write('WAT 1,2,0.1')
        self.my_inst.write('WAT 2,6,0.1')

        self.my_inst.write('FL 1,101')  # connect (1) to channel 101
        self.my_inst.write('FL 1,201')  # connect (1) to channel 201

        self.my_inst.write('SSR 101,1')  # turns on (1) resistor on channel 101
        self.my_inst.write('SSR 201,0')  # turns off (0) resistor on channel 201

        self.my_inst.write('WT 1,0.1,0.1,0,0')
        self.error_check()

    def turn_on_drive(self):

        self.drive = serial.Serial('COM3', baudrate=9600, write_timeout=2, timeout=2, rtscts=True)

    def error_check(self):

        k = self.my_inst.query('ERRX?')
        if k.startswith('+0') == False:
            print(k)

    def measure(self, Vs1, Vs2, num_Vs, Vc, num_cycles):

        self.data = np.zeros((num_Vs * len(Vc) * 2, 2 + 2 * num_cycles))

        self.test_data = np.zeros((22*len(Vc)))

        self.num_cycles = num_cycles

        locmax = mpl.ticker.SymmetricalLogLocator(base=10.0, linthresh=1e-13, subs=(0.1, 1, 10, 100,))
        locmin = mpl.ticker.SymmetricalLogLocator(base=10.0, linthresh=1e-13, subs=(0.2, 0.4, 0.6, 0.8, 1,))

        fig, ax = plt.subplots(1, 2)


        for i in range(0, len(Vc)):

            status = ''

            self.my_inst.write('DV 201,0,{},0.1,0,0'.format(Vc[i]))
            self.error_check()

            self.my_inst.write('WV 101,3,0,{},{},11,0.1,0.1'.format(Vs1, Vs2))
            self.error_check()

            self.my_inst.write('WM 1,1')
            self.error_check()

            self.my_inst.write('BC')

            self.my_inst.write('XE')

            self.my_inst.query('*OPC?')

            self.my_inst.query('NUB?')

            self.error_check()

            data_raw = self.my_inst.read_ascii_values()

            for k in range(0, 22):
                self.test_data[2 * i * num_Vs + k, 0] = Vc[i]
                self.test_data[2 * i * num_Vs + k, 1] = float(data_raw[3 * k + 2])
                self.test_data[2 * i * num_Vs + k, 2] = float(data_raw[3 * k + 1])
                self.test_data[2 * i * num_Vs + k, 3] = float(data_raw[3 * k])

            if abs(np.mean(self.test_data[:, 2])) < 1e-11:

                status = 'Non-conductive'

            if abs(np.mean(self.test_data[:, 3])) > 1e-8:

                status = 'Gate leak'

            if abs(np.mean(self.test_data[:])) > 1e3:

                status = 'Too high current'

            if status != '':

                for j in range(0, num_cycles):

                    self.my_inst.write('BC')

                    self.my_inst.write('WV 101,3,0,{},{},{},0.1,0.1'.format(Vs1, Vs2, num_Vs))
                    self.error_check()

                    self.my_inst.write('XE')

                    self.my_inst.query('*OPC?')

                    self.my_inst.query('NUB?')
                    self.error_check()

                    data_raw = self.my_inst.read_ascii_values()

                    for k in range(0, 2 * num_Vs):

                        self.data[2 * i * num_Vs + k, 0] = Vc[i]
                        self.data[2 * i * num_Vs + k, 1] = float(data_raw[3 * k + 2])
                        self.data[2 * i * num_Vs + k, 2 + 2 * j] = float(data_raw[3 * k + 1])
                        self.data[2 * i * num_Vs + k, 3 + 2 * j] = float(data_raw[3 * k])

                    # max_Isd = np.zeros((j + 1))
                    # min_Isd = np.zeros((j + 1))
                    # max_Ig = np.zeros((j + 1))
                    # min_Ig = np.zeros((j + 1))
                    #
                    # plt.clf()
                    #
                    # for k in range(0, j+1):
                    #
                    #     ax[0].plot(self.data[2*i*num_Vs:2*(i+1)*num_Vs, 1],
                    #                self.data[2*i*num_Vs:2*(i+1)*num_Vs, 2 + 2 * k] * np.sign(Vc[i]))
                    #
                    #     max_Isd[k] = np.max(self.data[2*i*num_Vs:2*(i+1)*num_Vs, 2 + 2 * k] * np.sign(Vc[i]))
                    #     min_Isd[k] = np.min(self.data[2*i*num_Vs:2*(i+1)*num_Vs, 2 + 2 * k] * np.sign(Vc[i]))
                    #
                    #     ax[1].plot(self.data[2*i * num_Vs:2*(i + 1) * num_Vs, 1],
                    #                self.data[2*i * num_Vs:2*(i + 1) * num_Vs, 3 + 2 * k])
                    #
                    #     max_Ig[k] = np.max(self.data[2*i * num_Vs:2*(i + 1) * num_Vs, 3 + 2 * k])
                    #     min_Ig[k] = np.min(self.data[2*i * num_Vs:2*(i + 1) * num_Vs, 3 + 2 * k])
                    #
                    # ax[1].suptitle('Vsd = {} V'.format(Vc[i]))
                    #
                    # ax[0].set_xlabel('Vg')
                    # ax[0].set_ylabel('Isd*sgn(Isd)')
                    #
                    # ax[1].set_xlabel('Vg')
                    # ax[1].set_ylabel('Ig')
                    #
                    # ax[0].yaxis.set_major_locator(locmax)
                    # ax[0].yaxis.set_minor_locator(locmin)
                    # ax[0].yaxis.set_minor_formatter(mpl.ticker.LogFormatter())
                    #
                    # ticks_Isd = np.array(list(ax[0].axes.get_yticks()))
                    # ticks_Isd = ticks_Isd[(abs(ticks_Isd) > 5 * 1e-14) | (ticks_Isd == 0)]
                    # ticks_Isd = ticks_Isd[ticks_Isd < max(max_Isd)]
                    # ticks_Isd = ticks_Isd[ticks_Isd > min(min_Isd)]
                    # ax[0].axes.set_yticks(ticks_Isd)
                    #
                    # ax[1].yaxis.set_major_locator(locmax)
                    # ax[1].yaxis.set_minor_locator(locmin)
                    # ax[1].yaxis.set_minor_formatter(mpl.ticker.LogFormatter())
                    #
                    # ticks_Ig = np.array(list(ax[1].axes.get_yticks()))
                    # ticks_Ig = ticks_Ig[(abs(ticks_Ig) > 5 * 1e-14) | (ticks_Ig == 0)]
                    # ticks_Ig = ticks_Ig[ticks_Ig < max(max_Ig)]
                    # ticks_Ig = ticks_Ig[ticks_Ig > min(min_Ig)]
                    # ax[1].axes.set_yticks(ticks_Ig)
                    #
                    # plt.show()

                    # Ask for any error

                    self.error_check()

        if np.equal(self.data, 0).all():

            self.data = self.test_data

    def write_to_csv(self, filepath_kernel, current_x, current_y):

        filepath = filepath_kernel + r'\\data'
        os.makedirs(filepath, exist_ok=True)

        df = pd.DataFrame(self.data)
        header = []

        if self.measurement_type == 'Isd-Vg':

            header.append('Vsd')
            header.append('Vg')

            for i in range(0, self.num_cycles):
                header.append('Isd%s' % (i + 1))
                header.append('Ig%s' % (i + 1))

        else:

            header.append('Vg')
            header.append('Vsd')

            for i in range(0, self.num_cycles):
                header.append('Ig%s' % (i + 1))
                header.append('Isd%s' % (i + 1))

        pd.DataFrame.to_csv(df, path_or_buf=filepath + r'\\fet_%s_%s.csv' % (current_x, current_y),
                            index = False, header = header)

    def drive_commands_move(self, command_array):

        k = self.drive.read_all()

        # if k != b'':
        #
        #     print('Initial: {}'.format(k))

        time.sleep(0.05)

        for item in command_array:

            self.drive.write((item + '\r').encode())
            print('Send: {}'.format((item + '\r').encode()))
            time.sleep(0.05)

            k = self.drive.read_all()

            # if k != b'':
            #
            #     print('Recieve: {}'.format(k))

            time.sleep(0.05)

            while item.encode() not in k:

                k = self.drive.read_all()

                # if k != b'':
                #
                #     print('Recieve: {}'.format(k))

                time.sleep(0.05)

        while 'e'.encode() not in k:

            k = self.drive.read_all()

            # if k != b'':
            #
            #     print('Final recieve: {}'.format(k))

            time.sleep(0.05)

    def move_xyz(self, x_dist, y_dist, cols, rows, current_x, current_y):

        if current_x < cols:

            arr = ['2', '3', '0', '0', '1600']
            self.drive_commands_move(arr)

            arr = ['2', '1', '0', '0']
            arr.append(str(int(3.2 * x_dist)))
            self.drive_commands_move(arr)

            arr = ['2', '3', '1', '0', '1600']
            self.drive_commands_move(arr)

        else:

            if current_y < rows:

                arr = ['2', '3', '0', '0', '1600']
                self.drive_commands_move(arr)

                arr = ['2', '1', '1', '0']
                arr.append(str(int(3.2*x_dist*(cols-1))))
                self.drive_commands_move(arr)

                arr = ['2', '2', '0', '0']
                arr.append(str(int(3.2*y_dist)))
                self.drive_commands_move(arr)

                arr = ['2', '3', '1', '0', '1600']
                self.drive_commands_move(arr)

    def move_to_zero(self):

        arr = ['4', '3']
        self.drive_commands_move(arr)

        arr = ['4', '2']
        self.drive_commands_move(arr)

        arr = ['4', '1']
        self.drive_commands_move(arr)

        self.drive.close()
        self.drive.__exit__





