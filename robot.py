#!/usr/bin/env python3
"""
robot.py — serial/WiFi bridge to the quadruped (sesame-robot-sense).

Sends commands to the robot over WiFi (default, so it can run on battery) or
USB serial. Useful for manual testing and scripted choreography.

USAGE
  python3 robot.py "forward"            # send one command, print the reply
  python3 robot.py --serial "stop"      # force USB serial instead of WiFi
  python3 robot.py --host 192.168.1.42 "pose"   # connect by IP
  python3 robot.py --seq <<'EOF'        # timed choreography: one step per line
  stand
  wait 0.5
  forward
  wait 3
  stop
  EOF

A line in --seq mode is either `wait <seconds>` (sleep) or any firmware
command (sent verbatim). Blank lines and `# comments` are ignored.

Transport: defaults to WiFi (quadruped.local:8888 via mDNS) and falls back to
USB serial. Use --serial / --net to force one, --host to override the address.

Firmware command surface (see docs/commands.md):
  neutral | all <a> | servo <id> <a> | nudge <id> <d>
  hips <a> | knees <a> | stance <hip> <knee>
  stand | rest | stop
  forward | backward | left | right | gait <leftScale> <rightScale>
  wave | dance | swim | point | bow | pushup | cute | freaky | worm | shake | shrug | dead | crab
  trim <id> <v> | rev <id> | save | load | clear | map | pose | dump | help
"""

import argparse
import sys
import time

import robot_link


def run_sequence(link, lines):
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.lower().startswith("wait "):
            try:
                secs = float(line.split(None, 1)[1])
            except (IndexError, ValueError):
                print(f"! bad wait: {line}", file=sys.stderr)
                continue
            time.sleep(secs)
            continue
        link.send(line)
        reply = link.read_reply()
        print(f"> {line}" + (f"\n{reply}" if reply else ""))


def main():
    ap = argparse.ArgumentParser(description="Bridge to the quadruped (WiFi/serial).")
    ap.add_argument("command", nargs="?", help="single firmware command to send")
    ap.add_argument("--seq", action="store_true", help="read a timed sequence from stdin")
    ap.add_argument("--host", help="robot address (default: quadruped.local)")
    ap.add_argument("--serial", action="store_true", help="force USB serial")
    ap.add_argument("--net", action="store_true", help="force WiFi (no serial fallback)")
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

    try:
        if args.seq:
            run_sequence(link, sys.stdin.readlines())
        elif args.command:
            link.send(args.command)
            reply = link.read_reply()
            if reply:
                print(reply)
        else:
            sys.exit("Nothing to do. Pass a command, or --seq.")
    finally:
        link.close()


if __name__ == "__main__":
    main()
