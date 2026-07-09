#!/usr/bin/env bash
# Launch the Sesame companion app.
# Creates venv on first run; always syncs deps from requirements.txt.
set -e
cd "$(dirname "$0")"

VENV=".venv"

if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"

# portaudio is required for PyAudio (macOS only, safe to skip if already installed)
if ! brew list portaudio &>/dev/null 2>&1; then
    echo "Installing portaudio (required for PyAudio)..."
    brew install portaudio
fi

# Always sync deps so new packages (opencv, Pillow, etc.) are picked up automatically
pip install -q -r requirements.txt

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo ""
        echo "Created .env from .env.example — edit it to set SESAME_ROBOT_IP and LOCAL_LLM_MODEL."
        echo ""
    fi
fi

# Start Ollama in the background if not already running
if ! pgrep -x ollama > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &>/dev/null &
    sleep 2
fi

exec python3 sesame_gui.py "$@"
