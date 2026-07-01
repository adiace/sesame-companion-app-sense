#!/usr/bin/env bash
# Launch the Sesame companion app (creates venv + installs deps on first run).
set -e
cd "$(dirname "$0")"

VENV=".venv"

if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
    # portaudio is required for PyAudio
    if ! brew list portaudio &>/dev/null 2>&1; then
        echo "Installing portaudio (required for PyAudio)..."
        brew install portaudio
    fi
    echo "Installing Python deps..."
    pip install faster-whisper SpeechRecognition PyAudio pyttsx3 requests numpy python-dotenv pyserial
else
    source "$VENV/bin/activate"
fi

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo ""
        echo "Created .env from .env.example — edit it to set SESAME_ROBOT_IP and LOCAL_LLM_MODEL."
        echo ""
    fi
fi

exec python3 sesame_gui.py "$@"
