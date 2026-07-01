# Sesame Companion App

Desktop companion for the Sesame quadruped robot. Runs entirely on your laptop — local STT, local LLM, macOS TTS — with a Tkinter GUI for chat, quick commands, and voice control.

---

## Requirements

- macOS (TTS uses the built-in `say` command)
- Python 3.10+
- [Ollama](https://ollama.com) running locally with a model pulled
- Sesame robot on the same network (`quadruped.local` or a known IP)

---

## Setup

```bash
# 1. Install Ollama and pull a model
brew install ollama
ollama serve          # keep running in background
ollama pull llama3.2

# 2. Clone and launch
git clone https://github.com/adiace/sesame-companion-app-sense.git
cd sesame-companion-app-sense
./run.sh              # creates venv, installs deps, launches GUI
```

On first run, `run.sh` creates `.env` from the template. Set `SESAME_ROBOT_IP` to the robot's address.

---

## Configuration

Copy `.env.example` to `.env` and edit:

```
SESAME_ROBOT_IP=quadruped.local   # or an IP like 192.168.1.42
LOCAL_LLM_URL=http://localhost:11434/v1
LOCAL_LLM_MODEL=llama3.2
SESAME_LOCAL=true
TTS_ENGINE=pyttsx3                # 'pyttsx3' uses macOS say; 'gemini' requires GEMINI_API_KEY
WAKE_WORD=hey sesame
WAKE_WORD_MODE=false
```

To test without a robot: `SESAME_ROBOT_IP=mock`

---

## Usage

### GUI

```bash
./run.sh
# or:
source .venv/bin/activate
python3 sesame_gui.py
```

**Chat input** — type a message and press Enter. The LLM interprets it and sends the matching command to the robot. Prefix with `/` to bypass the LLM and send a raw command directly:

```
/nudge R1 -10
/face happy
/walk
```

**Quick action buttons** — send common poses (walk, rest, dance, box, stand, stop, …) without going through the LLM.

**Voice mode** — click the mic button to speak. Click again to cancel. Transcribed by `faster_whisper` locally, then sent to the LLM.

**Robot voice (on-device wake word)** — the robot's own mic detects "Hi ESP", records a clip, and sends it to the companion app on port 8889. The app transcribes, queries the LLM, speaks the response with `say`, and sends the WAV back so the robot plays it on its speaker — all automatically, no button press needed.

---

## Voice pipeline

```
Robot mic detects "Hi ESP"
    → records 4s PCM
    → streams to companion app :8889

Companion app (RobotVoiceReceiver):
    → faster_whisper STT (local, base model)
    → Ollama LLM → {command, face, response}
    → macOS say + afconvert → 16kHz WAV
    → WAV sent back to robot (plays on MAX98357A speaker)
    → command sent to robot via TCP :8888
    → conversation appears in GUI chat
```

---

## Architecture

| Component | Role |
|---|---|
| `sesame_companion.py` | Core library — LLM, STT, TTS, robot TCP controller, robot voice receiver |
| `sesame_gui.py` | Tkinter GUI — chat, quick actions, mic button, settings |
| `robot_link.py` | Low-level TCP + serial transport |
| `robot.py` | CLI bridge — send single commands or timed sequences |
| `imu_receiver.py` | Print IMU events from the robot in real time |
| `serial_monitor.py` | Stream the robot's debug log (TCP port 8890) |
| `moves.json` | Named action library |

### Communication channels

| Channel | Direction | Purpose |
|---|---|---|
| TCP :8888 | laptop → robot | Commands and face changes (persistent connection, ~5ms latency) |
| HTTP GET /api/status | laptop → robot | Poll current face and command |
| TCP :8889 | robot → laptop | Robot streams PCM after wake word |
| TCP :8890 | robot → laptop | Robot pushes debug log (serial monitor) |

---

## LLM fallbacks

If the LLM returns no command for a clearly physical request (e.g., "go into box mode"), the app falls back to a keyword scan of the transcript before giving up. If the LLM returns no response text, Sesame stays in character with a random short line.

---

## Troubleshooting

**Robot not found** — check `SESAME_ROBOT_IP` in `.env`. Try `ping quadruped.local`.

**Port 8889 in use** — kill the old process: `lsof -ti:8889 | xargs kill`

**Whisper slow on first run** — downloads the `base` model (~145 MB) once, then caches it.

**No audio from robot speaker** — the MAX98357A needs 3.3V power (not 5V) and a clean supply. Battery power works best. See [wiring](../sesame-robot-sense/docs/wiring.md).

**LLM not responding** — confirm `ollama serve` is running and the model name in `.env` matches a pulled model (`ollama list`).
