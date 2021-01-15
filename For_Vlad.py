import serial
import time


class Shit(object):

    def __init__(self, com_name, baudrate=9600, write_timeout=2, timeout=2, rtscts=True):

        self.drive = serial.Serial(com_name, baudrate=baudrate,
                                   write_timeout=write_timeout, timeout=timeout, rtscts=rtscts)
        time.sleep(1)

        self.drive.read_all()

    def __serial_commands(self, command_array):

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

    def light(self, state: str):

        assert state in ['ON', 'OFF'], '"state" argument can take only the following values: "ON" and "OFF"'
        _map = {'ON': '40', 'OFF': '41'}
        _command = [_map[state]]
        self.__serial_commands(_command)
        print('Light is turned {}'.format(state))
        time.sleep(0.1)

    def valves_1_and_2(self, state: str):

        assert state in ['ON', 'OFF'], '"state" argument can take only the following values: "ON" and "OFF"'
        _map = {'ON': '20', 'OFF': '21'}
        _command = [_map[state]]
        self.__serial_commands(_command)
        print('Valves 1 and 2 are turned {}'.format(state))
        time.sleep(0.1)

    def valve_3(self, state: str):

        assert state in ['ON', 'OFF'], '"state" argument can take only the following values: "ON" and "OFF"'
        _map = {'ON': '30', 'OFF': '31'}
        _command = [_map[state]]
        self.__serial_commands(_command)
        print('Valve 3 is turned {}'.format(state))
        time.sleep(0.1)

    def pump(self, state: str):

        assert state in ['ON', 'OFF'], '"state" argument can take only the following values: "ON" and "OFF"'
        _map = {'ON': '10', 'OFF': '11'}
        _command = [_map[state]]
        self.__serial_commands(_command)
        print('Pump is turned {}'.format(state))
        time.sleep(0.1)

    def __del__(self):

        self.drive.read_all()
        self.drive.close()
        self.drive.__exit__

COM_name = ''
base = Shit(com_name=COM_name)


def sequential():
    # Первый этап
    base.valve_3('OFF')
    base.valves_1_and_2('ON') # открываю 1 и 2 клапаны
    base.pump('ON')

    # Жду 10 мин
    time.sleep(600)

    # Второй этап
    base.valves_1_and_2('OFF')
    base.valve_3('ON')

    # Жду 10 мин
    time.sleep(600)

    # Третий этап
    base.pump('OFF')

    # Жду 40 мин
    time.sleep(2400)


sequential()

if isinstance(base, Shit):
    del base
