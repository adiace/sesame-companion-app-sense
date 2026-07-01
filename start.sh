#!/usr/bin/env zsh
# Full-stack launcher for the Sesame companion app.
#
# Opens serial_monitor.py in a separate Terminal window (robot debug log),
# then launches the companion GUI in this window.
#
# Usage:
#   ./start.sh                     # auto-discovers quadruped.local
#   ./start.sh 192.168.68.100      # explicit robot IP
set -e
cd "$(dirname "$0")"

VENV=".venv"
if [ ! -d "$VENV" ]; then
    echo "venv not found — run ./run.sh first to set it up."
    exit 1
fi

ROBOT_IP="${1:-}"

# Open serial monitor in a new Terminal window
MONITOR_CMD="source '$PWD/$VENV/bin/activate' && python3 '$PWD/serial_monitor.py' ${ROBOT_IP}"
osascript -e "tell application \"Terminal\"
  do script \"$MONITOR_CMD\"
end tell"

# Launch the companion GUI in this window
exec "$VENV/bin/python3" sesame_gui.py
