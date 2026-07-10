#!/usr/bin/env python3
"""
imu_receiver.py — print IMU events from the robot over TCP.

Connects to sesame-robot.local:8888 and listens for JSON lines pushed by the
robot when a physical event is detected. All other lines (command echoes,
status) are ignored.

USAGE
  python3 imu_receiver.py
  python3 imu_receiver.py --host 192.168.1.42
  python3 imu_receiver.py --net          # WiFi only
  python3 imu_receiver.py --serial       # USB serial fallback
"""

import argparse
import json
import sys
import time

import robot_link


def main():
    ap = argparse.ArgumentParser(description="Print IMU events from the robot.")
    ap.add_argument("--host",   help="robot address (default: sesame-robot.local)")
    ap.add_argument("--serial", action="store_true", help="force USB serial")
    ap.add_argument("--net",    action="store_true", help="force WiFi")
    args = ap.parse_args()

    if args.serial and args.net:
        sys.exit("Pick one of --serial / --net.")
    prefer = "serial-only" if args.serial else "net-only" if args.net else "net"

    try:
        link = robot_link.connect(
            prefer=prefer,
            host=args.host or robot_link.DEFAULT_HOST,
        )
    except ConnectionError as e:
        sys.exit(str(e))

    print(f"Connected via {link.describe()}. Listening for IMU events…")
    print("(Ctrl-C to quit)\n")

    try:
        while True:
            line = link.readline(timeout=0.1)
            if not line:
                continue
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") != "imu_event":
                continue

            event = data.get("event", "?")
            accel = data.get("accel", 0.0)
            pitch = data.get("pitch", 0.0)
            roll  = data.get("roll",  0.0)

            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] {event:<10}  accel={accel:.2f}g  pitch={pitch:+.1f}°  roll={roll:+.1f}°")

    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        link.close()


if __name__ == "__main__":
    main()
