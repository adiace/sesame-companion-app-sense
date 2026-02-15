# Sesame Robot Companion App

AI-powered natural language interface for the Sesame robot using Google Gemini and PyTTSx3.

## Setup

### Requirements
- Python 3.7+
- Sesame robot on same network
- Google Gemini API key from [ai.google.dev](https://aistudio.google.com/app/apikey)

### Installation

```powershell
pip install -r requirements.txt
```

### Configuration

Option 1 - Environment variables:
```powershell
$env:SESAME_ROBOT_IP="192.168.1.100"
$env:GEMINI_API_KEY="your_key_here"
```

Option 2 - `.env` file:
```
SESAME_ROBOT_IP=192.168.1.100
GEMINI_API_KEY=your_key_here
VOICE_ENABLED=true
TTS_ENGINE=pyttsx3
WAKE_WORD=hey sesame
WAKE_WORD_MODE=false
```

## Usage

### CLI Mode
```powershell
python sesame_companion.py
```

### GUI Mode
```powershell
python sesame_gui.py
```

## Features

### Commands
- **Movement**: walk, dance, wave, swim, pushup, bow, shake, etc.
- **Control**: idle, stop, rest
- **Status**: Check robot state

### Faces
- **Conversational**: happy, sad, angry, excited, sleepy, love, confused, surprised
- **Action-specific**: Auto-selected during movements

### Voice Control
- Speech recognition via Google Speech API
- TTS engines: pyttsx3 (local) or Gemini (cloud)
- Wake word support: "hey sesame" (configurable)
- Audio-synced talking animation

### AI Personality
- Small robot with limited intelligence
- Randomly sarcastic/mean responses (~90%)
- Self-aware of being artificial (~20%)
- Short, simple responses with comedic timing

## Architecture

### Components

**VoiceInterface**
- Speech recognition (Google Speech API)
- TTS with pyttsx3 or Gemini
- Audio-level based mouth animation (pyaudio)
- Wake word detection

**SesameRobotController**
- HTTP API client for robot control
- Handles face-only updates (idle command)
- Command validation and error handling

**GeminiInterface**
- Natural language processing
- Command/face extraction from user input
- Personality and response generation

**SesameCompanionApp**
- Orchestrates all components
- Interactive CLI loop
- Command routing and execution

### Face Animation System

The robot has two types of faces:
- **Base faces**: Emotion with mouth closed (e.g., `happy`)
- **Talk variants**: Same emotion with mouth open (e.g., `talk_happy`)

During TTS, the system switches between base and talk variants based on audio levels detected via pyaudio. Falls back to time-based animation if audio monitoring unavailable.

### API Protocol

**Command endpoint**: `POST /api/command`
```json
{"command": "walk"}
{"face": "happy"}
{"command": "wave", "face": "excited"}
```

**Status endpoint**: `GET /api/status`
```json
{
  "currentCommand": "idle",
  "currentFace": "happy",
  "networkConnected": true
}
```

## Finding Robot IP

1. Connect robot via USB
2. Open serial monitor at 115200 baud
3. Look for: `Connected to network! IP: X.X.X.X`

## Troubleshooting

**pyaudio not available**
- Audio-synced animation will fall back to time-based
- Install: `pip install pyaudio`
- Windows: May need unofficial wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)

**Robot connection failed**
- Verify robot is on and connected to network
- Check IP address is correct
- Ensure same network/subnet
- Test with `curl http://ROBOT_IP/api/status`

**TTS not working**
- pyttsx3: Check system TTS engines installed
- Gemini TTS: Requires valid API key and `pygame` package

**Wake word not detecting**
- Adjust microphone input level
- Reduce ambient noise
- Speak clearly near microphone
