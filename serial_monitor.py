#!/usr/bin/env python3
"""
WiFi serial monitor for Sesame robot.

Connects to the robot's debug log server (TCP port 8890) and prints all
output that would normally appear in the Arduino Serial Monitor.

Usage:
  python software/serial_monitor.py              # auto-discovers quadruped.local
  python software/serial_monitor.py 192.168.1.X  # explicit IP

The robot firmware must be running (flashed with wifi_log.h support).
Reconnects automatically if the connection drops.
"""

import socket
import sys
import time

HOST = sys.argv[1] if len(sys.argv) > 1 else "quadruped.local"
PORT = 8890
RECONNECT_DELAY = 3  # seconds between reconnect attempts


def connect() -> socket.socket:
    while True:
        try:
            s = socket.create_connection((HOST, PORT), timeout=10)
            s.settimeout(None)  # blocking reads after connect
            print(f"\033[32m[connected to {HOST}:{PORT}]\033[0m", flush=True)
            return s
        except (OSError, socket.timeout) as e:
            print(f"\033[33m[{HOST}:{PORT} unreachable: {e} — retrying in {RECONNECT_DELAY}s]\033[0m",
                  flush=True)
            time.sleep(RECONNECT_DELAY)


def main():
    print(f"Sesame serial monitor → {HOST}:{PORT}  (Ctrl-C to quit)")
    while True:
        s = connect()
        try:
            buf = b""
            while True:
                chunk = s.recv(1024)
                if not chunk:
                    print("\033[33m[disconnected]\033[0m", flush=True)
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    print(line.decode(errors="replace").rstrip("\r"), flush=True)
        except (OSError, ConnectionResetError) as e:
            print(f"\033[33m[{e}]\033[0m", flush=True)
        finally:
            s.close()
        time.sleep(RECONNECT_DELAY)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye")
