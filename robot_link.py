"""
robot_link.py — transport to the quadruped, over WiFi (TCP) or USB (serial).

Both robot.py and voice_control.py use connect() from here. Default is
wireless: it resolves the robot via mDNS at sesame-robot.local:8888 (so the
robot can run on battery, untethered). If that fails it falls back to a USB
serial port. Force either with connect(prefer="net"|"serial").

A Link exposes:
  .send(command)          send one command line (thread-safe)
  .read_reply(timeout)    read whatever the robot echoes back (str)
  .drain()                discard any pending input (non-blocking)
  .close()
  .describe()             human label of the active transport
"""

import glob
import socket
import threading
import time

DEFAULT_HOST = "sesame-robot.local"
TCP_PORT = 8888
BAUD = 115200

PORT_GLOBS = [
    "/dev/cu.usbmodem*",
    "/dev/cu.wchusbserial*",
    "/dev/cu.usbserial*",
    "/dev/cu.SLAB_USBtoUART*",
]


def find_port():
    for pattern in PORT_GLOBS:
        m = sorted(glob.glob(pattern))
        if m:
            return m[0]
    return None


class TcpLink:
    def __init__(self, host=DEFAULT_HOST, port=TCP_PORT, timeout=5.0):
        self.host, self.port = host, port
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(0.2)
        self._lock = threading.Lock()
        self._buf = b""  # persistent receive buffer for readline
        time.sleep(0.2)
        self.drain()  # discard the connect banner

    def send(self, command):
        data = (command.strip() + "\n").encode()
        with self._lock:
            self.sock.sendall(data)

    def read_reply(self, timeout=0.4):
        out, deadline = [], time.time() + timeout
        while time.time() < deadline:
            try:
                chunk = self.sock.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            out.append(chunk.decode(errors="replace"))
            deadline = time.time() + 0.15
        return "".join(out).strip()

    def readline(self, timeout=0.02):
        """Return one complete line from a persistent buffer.
        Never drops events when two arrive in the same TCP segment."""
        if b"\n" not in self._buf:
            self.sock.settimeout(timeout)
            try:
                chunk = self.sock.recv(1024)
                if chunk:
                    self._buf += chunk
            except socket.timeout:
                pass
            finally:
                self.sock.settimeout(0.2)
        if b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            return line.decode(errors="replace").strip()
        return ""

    def drain(self):
        try:
            self.sock.setblocking(False)
            while True:
                if not self.sock.recv(4096):
                    break
        except (BlockingIOError, OSError):
            pass
        finally:
            self.sock.setblocking(True)
            self.sock.settimeout(0.2)

    def describe(self):
        return f"WiFi {self.host}:{self.port}"

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


class SerialLink:
    def __init__(self, port):
        import serial
        self.port = port
        # Open without asserting DTR/RTS to avoid resetting the ESP32
        self.ser = serial.Serial()
        self.ser.port    = port
        self.ser.baudrate = BAUD
        self.ser.timeout = 0.2
        self.ser.dtr = False
        self.ser.rts = False
        self.ser.open()
        self._lock = threading.Lock()
        time.sleep(0.5)
        self.ser.reset_input_buffer()

    def send(self, command):
        data = (command.strip() + "\n").encode()
        with self._lock:
            self.ser.write(data)
            self.ser.flush()

    def read_reply(self, timeout=0.4):
        out, deadline = [], time.time() + timeout
        while time.time() < deadline:
            line = self.ser.readline().decode(errors="replace").strip()
            if line:
                out.append(line)
                deadline = time.time() + 0.15
        return "\n".join(out)

    def readline(self, timeout=0.02):
        """Read one newline-terminated line. Returns '' on timeout."""
        self.ser.timeout = timeout
        return self.ser.readline().decode(errors="replace").strip()

    def drain(self):
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass

    def describe(self):
        return f"USB serial {self.port}"

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


def connect(prefer="net", host=DEFAULT_HOST, tcp_port=TCP_PORT, serial_port=None,
            verbose=True):
    """Open a Link. prefer='net' tries WiFi first then serial; 'serial' the
    reverse; 'net-only'/'serial-only' disable the fallback."""
    errors = []

    def try_net():
        try:
            return TcpLink(host, tcp_port)
        except OSError as e:
            errors.append(f"WiFi {host}:{tcp_port} -> {e}")
            return None

    def try_serial():
        port = serial_port or find_port()
        if not port:
            errors.append("USB serial -> no port found")
            return None
        try:
            return SerialLink(port)
        except Exception as e:
            errors.append(f"USB serial {port} -> {e}")
            return None

    order = {
        "net": [try_net, try_serial],
        "serial": [try_serial, try_net],
        "net-only": [try_net],
        "serial-only": [try_serial],
    }[prefer]

    for attempt in order:
        link = attempt()
        if link:
            if verbose:
                print(f"Connected via {link.describe()}")
            return link

    raise ConnectionError("Could not reach the robot:\n  " + "\n  ".join(errors))
