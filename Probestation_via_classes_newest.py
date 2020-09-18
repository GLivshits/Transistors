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

        # These 3 rows choose appropriate visa library and opens instrument by its name
        rm = pyvisa.ResourceManager(r'C:\\Windows\\System32\\visa64.dll')
        self.my_inst = rm.open_resource(r'GPIB0::17::INSTR')
        del self.my_inst.timeout

        # The *IDN? query command returns the instrument model number and the firmware
        # revision number.
        self.my_inst.query('*IDN?')

        # The *RST command resets the B1500 to the initial settings. This command does not
        # clear the program memory and the self-calibration data
        self.my_inst.write('*RST')

        # The *SRE command enables the specified bits of the status byte register for SRQ
        # (service requests), and masks (disables) the bits that are not specified. Using 59 is always ok.
        self.my_inst.write('*SRE 59')

        # This command clears the B1500 output data buffer, and specifies the data output format.
        self.my_inst.write('FMT 12,1')

        # This command enables specified channels (SMU 1 = 101, SMU 2 = 201)
        self.my_inst.write('CN 101')
        self.my_inst.write('CN 201')

        # The MM command specifies the measurement mode and the channels used for measurements
        self.my_inst.write('MM 2,101,201')  # enables staircase sweep for channels 101 and 201

        # The CMM command sets the SMU measurement operation mode. 1 for current, 2 for voltage.
        self.my_inst.write('CMM 101,1')  # sets current measurement (1) for channel 101
        self.my_inst.write('CMM 201,1')  # sets current measurement(1) for channel 201

        # The RI command specifies the current measurement range or ranging type. In the
        # initial setting, the auto ranging is set.
        self.my_inst.write('RI 101,0')  # sets auto-ranging (0) for channel 101
        self.my_inst.write('RI 201,0')  # sets auto-ranging (0) for channel 201

        # AAD: This command is used to specify the A/D converter (ADC) type. 0 for high speed, 1 for high resolution.
        # AZ: This command is used to enable or disable the ADC zero function that is the
        # function to cancel offset of the high-resolution A/D converter
        # AIT: This command is used to set the integration time or the number of averaging
        # samples of the A/D converter (ADC) for each ADC type.
        self.my_inst.write('AAD 101,1;AZ 0;AIT 1, 2, 10')  # sets high resolution (1) for channel 101
        self.my_inst.write('AAD 201,1;AZ 0;AIT 1, 2, 10')  # sets high resolution (1) for channel 201

        # This command sets the source wait time and the measurement wait time.
        self.my_inst.write('WAT 1,6,0')
        self.my_inst.write('WAT 2,6,0')

        # This command sets the connection mode of a SMU filter for each channel.
        # A filter is mounted on the SMU. It assures clean source output with no spikes or
        # overshooting.
        self.my_inst.write('FL 1,101')  # enables filter on SMU 1
        self.my_inst.write('FL 1,201')  # enables filter on SMU 2

        # This command sets the connection mode of a SMU series resistor (approx. 1 MΩ)
        # for each channel.
        self.my_inst.write('SSR 101,1')  # turns on (1) resistor on channel 101
        self.my_inst.write('SSR 201,0')  # turns off (0) resistor on channel 201

        # The WT command sets the hold time, delay time, and step delay time for the
        # staircase sweep or multi channel sweep measurement.
        self.my_inst.write('WT 0.2,0.1,0,0,0.01')
        self.error_check()

    def turn_on_drive(self):

        # This command enables Om-Nom driver.
        self.drive = serial.Serial('COM3', baudrate=9600, write_timeout=2, timeout=2, rtscts=True)

    def error_check(self):

        # The whole function is used to ask instrument for an error and returns an error only if it is present
        k = self.my_inst.query('ERRX?')
        if k.startswith('+0') == False:
            print(k)

    def measure(self, Vs1, Vs2, num_Vs, Vc, num_cycles):

        self.data = np.zeros((num_Vs * len(Vc) * 2, 2 + 2 * num_cycles)) # array of measured values

        self.test_data = np.zeros((22*len(Vc), 4)) # array of tested values

        self.num_cycles = num_cycles

        # locmax = mpl.ticker.SymmetricalLogLocator(base=10.0, linthresh=1e-13, subs=(0.1, 1, 10, 100,))
        # locmin = mpl.ticker.SymmetricalLogLocator(base=10.0, linthresh=1e-13, subs=(0.2, 0.4, 0.6, 0.8, 1,))

        # fig, ax = plt.subplots(1, 2)

        for i in range(0, len(Vc)):

            # First of all, fast measurement is conducted. If device is non-conductive or damaged - program
            # will skip it.
            print('Testing {} bias'.format(i+1))

            status = ''

            # The DV command forces DC voltage from the specified SMU.
            self.my_inst.write('DV 201,0,{},0.1,0,0'.format(Vc[i]))
            self.error_check()

            # The WV command specifies the staircase sweep voltage source and its parameters.
            self.my_inst.write('WV 101,3,0,{},{},11,0.1,0.1'.format(Vs1, Vs2))
            self.error_check()

            # The WM command enables or disables the automatic abort function for the staircase
            # sweep sources and the pulsed sweep source.
            self.my_inst.write('WM 1,1')
            self.error_check()

            # The BC command clears the output data buffer that stores measurement data and query command response data.
            self.my_inst.write('BC')

            # This command starts measurement.
            self.my_inst.write('XE')

            # The *OPC? command monitors the pending operations, and places ASCII character
            # 1 into the output queue when all pending operations are completed.
            self.my_inst.query('*OPC?')

            # The NUB? query command checks the number of measurement data in the output
            # data buffer, and stores the results in the output data buffer (query buffer).
            self.my_inst.query('NUB?')
            self.error_check()

            # This command reads the instrument data buffer.
            data_raw = self.my_inst.read_ascii_values()

            self.test_data[22 * i: 22 * (i + 1), 0] = Vc[i]
            self.test_data[22 * i: 22 * (i + 1), 1] = float(data_raw[2::3])
            self.test_data[22 * i: 22 * (i + 1), 2] = float(data_raw[1::3])
            self.test_data[22 * i: 22 * (i + 1), 3] = float(data_raw[0::3])

            # if maximum SD current less than 1 nA - device is considered as non-conductive.
            if (np.amax(abs(self.test_data[22 * i: 22 * (i + 1), 2]))) < 1e-9:

                status = 'Non-conductive'
                print('Non-conductive')

            # if gate current is greater than 1 nA - device is considered as gate-damaged.
            if (np.mean(abs(self.test_data[22 * i: 22 * (i + 1), 3]))) > 1e-9:

                status = 'Gate leak'
                print('Gate leak')

            # If just shitty measurement, which caused +227 error - returns this.
            if (np.mean(abs(self.test_data[:]))) > 1e3:

                status = 'Too high current'
                print('Too high current')

            # if status is empty, it means that test is successful. Full measurement is conducted.
            if status == '':

                for j in range(0, num_cycles):

                    print('Everything ok! Now {} cycle ongoing'.format(j+1))

                    self.my_inst.write('BC')

                    self.my_inst.write('WV 101,3,0,{},{},{},0.1,0.1'.format(Vs1, Vs2, num_Vs))
                    self.error_check()

                    self.my_inst.write('XE')

                    self.my_inst.query('*OPC?')

                    self.my_inst.query('NUB?')
                    self.error_check()

                    data_raw = self.my_inst.read_ascii_values()

                    self.data[2 * num_Vs * i: 2 * num_Vs * (i + 1), 0] = Vc[i]
                    self.data[2 * num_Vs * i: 2 * num_Vs * (i + 1), 1] = float(data_raw[2::3])
                    self.data[2 * num_Vs * i: 2 * num_Vs * (i + 1), 2 + 2 * j] = float(data_raw[1::3])
                    self.data[2 * num_Vs * i: 2 * num_Vs * (i + 1), 3 + 2 * j] = float(data_raw[0::3])

                    print('On current = {}'.format(self.data[2 * i * num_Vs, 2 + 2*j]))
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

        # Since the data array was created all zeros, if test is failed, it will remain fully zero.
        # This returns the test values.
        if np.equal(self.data, 0).all():

            self.data = self.test_data

    def write_to_csv(self, filepath_kernel, current_x, current_y):

        # This function creates .csv file out of data array.
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

        pd.DataFrame.to_csv(df, path_or_buf=filepath + r'\\fet_%s_%s.csv' % (current_y, current_x),
                            index = False, header = header)

    def drive_commands_move(self, command_array):

        k = self.drive.read_all()

        # if k != b'':
        #
        #     print('Initial: {}'.format(k))

        time.sleep(0.05)

        for item in command_array:

            self.drive.write((item + '\r').encode())
            # print('Send: {}'.format((item + '\r').encode()))
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

        while 'e'.encode not in self.drive.read_all():

            self.drive.read_all()

        self.drive.close()
        self.drive.__exit__





