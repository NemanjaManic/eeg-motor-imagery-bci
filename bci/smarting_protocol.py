"""SMARTING serial protocol: device commands, packet parsing, and ADC-to-microvolt conversion."""

import time

import numpy as np
import serial

from bci.config import (
    BAUD_RATE,
    C3_BYTE_OFFSET,
    C4_BYTE_OFFSET,
    COM_PORT,
    GAIN,
    PACKET_SIZE,
    VOLTAGE_REFERENCE,
)

PACKET_START_BYTE = ord(">")
PACKET_END_BYTE = ord("<")
CHECKSUM_BYTE_RANGE = (1, 81)
CHECKSUM_BYTE_INDEX = 81

START_ACQUISITION_COMMAND = b">ON<"
STOP_ACQUISITION_COMMAND = b">OFF<"
SET_SAMPLING_RATE_160HZ_COMMAND = b">160<"
NORMAL_MODE_COMMAND = b">NORMAL<"


def select_channels_command(ch_a, ch_b, ch_c):
    """Build the command that activates channel groups `ch_a`/`ch_b`/`ch_c` (0xFF = all)."""
    return bytearray([ord(">"), ord("S"), ord("C"), ord(";"), ch_a, ch_b, ch_c, ord("<")])


def bytes_to_microvolts(channel_bytes):
    """Convert a 3-byte big-endian two's-complement ADC sample to microvolts."""
    raw_value = (int(channel_bytes[0]) << 16) + (int(channel_bytes[1]) << 8) + int(channel_bytes[2])
    if raw_value > 0x007FFFFF:
        raw_value -= 0x01000000
    scale_factor = (VOLTAGE_REFERENCE / (2 ** 23 - 1)) / GAIN
    return raw_value * scale_factor * 1e6


class SmartingClient:
    """Serial connection using the SMARTING protocol.

    Developed and tested against the SMARTING simulator; real SMARTING
    hardware should be compatible since it speaks the same wire protocol,
    but that has not been verified.
    """

    def __init__(self, port=COM_PORT, baud_rate=BAUD_RATE, packet_size=PACKET_SIZE, read_timeout=0.1):
        self.serial_port = serial.Serial(
            port=port,
            baudrate=baud_rate,
            bytesize=serial.EIGHTBITS,
            stopbits=serial.STOPBITS_ONE,
            xonxoff=False,
            rtscts=True,
            dsrdtr=False,
            timeout=read_timeout,
        )
        self.serial_port.set_buffer_size(rx_size=10000, tx_size=10000)
        self.packet_size = packet_size
        self._buffer = np.array([], dtype=np.uint8)

    def send_command(self, command):
        time.sleep(1)
        if not self.serial_port.is_open:
            print("Error: could not send command, port is not open.")
            return
        self.serial_port.write(command)
        print(f"Sent: {command}")

    def start_acquisition(self):
        self.send_command(START_ACQUISITION_COMMAND)
        self.send_command(select_channels_command(0xFF, 0xFF, 0xFF))
        self.send_command(SET_SAMPLING_RATE_160HZ_COMMAND)
        self.send_command(NORMAL_MODE_COMMAND)

    def stop_acquisition(self):
        self.send_command(STOP_ACQUISITION_COMMAND)

    def read_c3_c4_samples(self):
        """Poll the port and return newly parsed (c3_uV, c4_uV) samples from complete, checksum-valid packets."""
        samples = []
        if not (self.serial_port and self.serial_port.is_open):
            return samples

        bytes_waiting = self.serial_port.in_waiting
        if bytes_waiting > 0:
            new_data = self.serial_port.read(bytes_waiting)
            new_data = np.frombuffer(new_data, dtype=np.uint8)
            self._buffer = np.append(self._buffer, new_data)

        while len(self._buffer) >= self.packet_size:
            if self._buffer[0] == PACKET_START_BYTE and self._buffer[self.packet_size - 1] == PACKET_END_BYTE:
                packet = self._buffer[: self.packet_size]
                checksum = 0
                for j in range(*CHECKSUM_BYTE_RANGE):
                    checksum ^= packet[j]
                if checksum == packet[CHECKSUM_BYTE_INDEX]:
                    self._buffer = self._buffer[self.packet_size:]
                    c3 = bytes_to_microvolts(packet[C3_BYTE_OFFSET:C3_BYTE_OFFSET + 3])
                    c4 = bytes_to_microvolts(packet[C4_BYTE_OFFSET:C4_BYTE_OFFSET + 3])
                    samples.append((c3, c4))
                else:
                    self._buffer = self._buffer[1:]
            else:
                self._buffer = self._buffer[1:]

        return samples

    def close(self):
        if not self.serial_port.is_open:
            print("Cannot close: COM port is not open.")
            return
        if self.serial_port.in_waiting > 0:
            response = self.serial_port.read(self.serial_port.in_waiting)
            if len(response) >= 4:
                print("Last 4 bytes:", response[-4:].decode("ascii", errors="ignore"))
        self.serial_port.reset_input_buffer()
        self.serial_port.reset_output_buffer()
        self.serial_port.close()
        print("Connection closed.")
