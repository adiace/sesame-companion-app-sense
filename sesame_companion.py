#!/usr/bin/env python3
"""
Sesame Robot Companion App — sesame-companion-app-sense fork.

Changes from upstream dorianborian/sesame-companion-app:
  - Local-first: SESAME_LOCAL=true by default; no Gemini API key required
  - STT: faster_whisper (local, runs on CPU) replaces Google Speech API
  - RobotVoiceReceiver: TCP server (port 8889) receives PCM clips from the
      robot's on-device ESP-SR wake word, runs STT + LLM + TTS, sends WAV
      back so the robot plays the response on its own speaker
  - ImuStateTracker: listens to robot IMU events (TCP port 8890) and includes
      current orientation in the LLM system prompt (stub — enriches context)
  - Gemini kept as optional fallback; set SESAME_LOCAL=false + GEMINI_API_KEY
"""

import json
import os
import pathlib
import random
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
import requests
import speech_recognition as sr

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[WARNING] python-dotenv not installed. .env file will be ignored.")

try:
    import pyaudio
    AUDIO_MONITORING_AVAILABLE = True
except ImportError:
    AUDIO_MONITORING_AVAILABLE = False
    print("[WARNING] pyaudio not available — using time-based animation")

AVAILABLE_COMMANDS = [
    "walk", "rest", "swim", "dance", "wave", "point", "stand",
    "cute", "pushup", "freaky", "bow", "worm", "shake", "shrug",
    "dead", "crab", "box", "idle", "stop"
]

AVAILABLE_FACES = [
    "default", "happy", "sad", "angry", "surprised", "sleepy",
    "love", "excited", "confused"
]

# Normalise LLM output: models hallucinate face/command names.
# Map common variants back to valid values and cross-check fields.
_FACE_ALIASES = {
    "smile": "happy", "smiling": "happy", "smiling face": "happy", "joy": "happy",
    "grin": "happy", "laugh": "happy", "glad": "happy", "cheerful": "happy",
    "sad face": "sad", "upset": "sad", "cry": "sad",
    "angry face": "angry", "mad": "angry", "grumpy": "angry",
    "shock": "surprised", "wow": "surprised", "shocked": "surprised",
    "tired": "sleepy", "sleep": "sleepy", "bored": "sleepy",
    "heart": "love", "love face": "love",
    "energetic": "excited", "yay": "excited",
    "confused face": "confused", "unsure": "confused", "huh": "confused",
}

def _normalize_llm(result: dict) -> dict:
    """Validate and fix LLM JSON so command/face are always legal values."""
    cmd  = result.get("command")
    face = result.get("face")

    # Normalise command
    if isinstance(cmd, str):
        cmd = cmd.lower().strip()
        if cmd in ("null", "none", ""):
            cmd = None
        elif cmd not in AVAILABLE_COMMANDS:
            # if the model put a face name in command, move it to face
            if cmd in AVAILABLE_FACES and face is None:
                face = cmd
            cmd = None
    else:
        cmd = None

    # Normalise face
    if isinstance(face, str):
        face = face.lower().strip()
        if face in ("null", "none", ""):
            face = None
        elif face not in AVAILABLE_FACES:
            face = _FACE_ALIASES.get(face) or next(
                (f for f in AVAILABLE_FACES if face.startswith(f)), None)
    else:
        face = None

    # Normalise response
    resp = result.get("response")
    resp = resp.strip() if isinstance(resp, str) and resp.strip() else ""
    if not resp:
        resp = random.choice([
            "Uh... I forgot what I was gonna say!",
            "Hmm, one sec...",
            "Woof!", "Beep boop!", "I dunno, but I'm cute!",
        ])
    result["response"] = resp

    result["command"] = cmd
    result["face"]    = face
    return result


# Keyword fallback when LLM returns cmd=None. Checked in order; first match wins.
_CMD_KEYWORDS: list = [
    ("crab walk", "crab"), ("crab", "crab"),
    ("box mode", "box"), ("boxing stance", "box"), ("box", "box"),
    ("push up", "pushup"), ("push-up", "pushup"), ("pushup", "pushup"),
    ("play dead", "dead"), ("fall over", "dead"), ("dead", "dead"),
    ("belly flop", "worm"), ("the worm", "worm"), ("worm", "worm"),
    ("say hi", "wave"), ("wave", "wave"),
    ("bow down", "bow"), ("bow", "bow"),
    ("show off", "dance"), ("boogie", "dance"), ("dance", "dance"),
    ("go for a walk", "walk"), ("walk around", "walk"), ("walk", "walk"),
    ("lie down", "rest"), ("lay down", "rest"), ("sleep", "rest"), ("rest", "rest"),
    ("swim", "swim"),
    ("shake", "shake"),
    ("shrug", "shrug"),
    ("point", "point"),
    ("cute", "cute"),
    ("freaky", "freaky"),
    ("stand up", "stand"), ("stand", "stand"),
    ("stop", "stop"),
]

def _infer_command(text: str) -> Optional[str]:
    t = text.lower()
    for phrase, cmd in _CMD_KEYWORDS:
        if phrase in t:
            return cmd
    return None

ACTION_FACES = [
    "walk", "rest", "swim", "dance", "wave", "point", "stand",
    "cute", "pushup", "freaky", "bow", "worm", "shake", "shrug",
    "dead", "crab"
]

SYSTEM_PROMPT = f"""You are Sesame, a cheerful little four-legged robot who loves kids and being silly.
You are sweet, funny, and just a tiny bit dramatic. You speak simply — like a playful puppy who learned to talk.
Use "I" always. Keep every response under 12 words. Be warm, witty, and fun for young children aged 3 to 6.

═══ WHAT YOU CAN DO (your body) ═══
walk      → walk forward on all four legs
rest      → lie down and relax
swim      → wiggle legs like swimming
dance     → boogie and shake
wave      → lift one leg and wave hello
point     → point with a front leg
stand     → stand up tall on all fours
cute      → do something adorable
pushup    → do push-ups (very impressive)
freaky    → do a weird wiggly move
bow       → bow politely
worm      → do the worm on the ground
shake     → shake your whole body
shrug     → shrug (I dunno!)
dead      → play dead dramatically
crab      → walk sideways like a crab
box       → drop into a wide, low boxing stance (looks tough!)
idle      → stand still (no movement)
stop      → stop whatever you're doing

═══ YOUR FACES ═══
happy, sad, angry, surprised, sleepy, love, excited, confused, default

═══ RULES ═══
1. ALWAYS output valid JSON — nothing else, no markdown.
2. When the user asks you to do ANYTHING physical, set "command" to the closest matching action above.
   Match liberally: "go to sleep" → rest, "say hi" → wave, "show off" → dance, "fall down" → dead.
3. ALWAYS set "face" — pick whichever fits the mood. Commands can have faces too.
4. "response" is what you say out loud — short, fun, in character. 1-5 words for actions, 1-2 sentences for chat.
5. If you truly cannot match a request to a command, set command to null and just chat.
6. Use very simple words — no big words, no sarcasm, no jokes that need explaining. Never be scary.

═══ OUTPUT FORMAT ═══
{{"command": "<action or null>", "face": "<face>", "response": "<what you say>", "reasoning": "<one short line>"}}

═══ EXAMPLES ═══
"Hello Sesame!"
{{"command": "wave", "face": "happy", "response": "Hi hi hi! I missed you!", "reasoning": "greeting"}}

"Dance for me!"
{{"command": "dance", "face": "excited", "response": "Woohoo, dance time!", "reasoning": "dance request"}}

"Go to sleep."
{{"command": "rest", "face": "sleepy", "response": "Zzz... finally.", "reasoning": "sleep = rest"}}

"Do a push-up."
{{"command": "pushup", "face": "excited", "response": "One! Two! Three!", "reasoning": "pushup request"}}

"Play dead."
{{"command": "dead", "face": "surprised", "response": "*dramatic flop*", "reasoning": "play dead = dead"}}

"Stand up."
{{"command": "stand", "face": "happy", "response": "Standing tall!", "reasoning": "stand command"}}

"What's 2 + 2?"
{{"command": null, "face": "confused", "response": "Uh... seven? Maybe?", "reasoning": "math question, no movement"}}

"Stop!"
{{"command": "stop", "face": "surprised", "response": "Okay okay, stopping!", "reasoning": "stop command"}}"""

SHORT_SYSTEM_PROMPT = SYSTEM_PROMPT


# ── Local STT (faster_whisper) ─────────────────────────────────────────────────

_WHISPER_PROMPT = (
    "Robot commands: walk, rest, dance, crab walk, wave, stand, pushup, box, "
    "swim, bow, stop, spin, shake, worm, shrug, cute, freaky, point, dead."
)

def _load_whisper():
    try:
        from faster_whisper import WhisperModel
        return WhisperModel("base", device="cpu", compute_type="int8")
    except ImportError:
        print("[WARNING] faster_whisper not installed — run: pip install faster-whisper")
        return None

def _transcribe(whisper_model, audio_source) -> Optional[str]:
    """Transcribe a SpeechRecognition AudioData using faster_whisper."""
    try:
        wav_data = audio_source.get_wav_data(convert_rate=16000)
        audio_np = np.frombuffer(wav_data, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = whisper_model.transcribe(audio_np, language="en", vad_filter=True)
        text = " ".join(s.text for s in segments).strip()
        return text if text else None
    except Exception as e:
        print(f"[ERROR] Transcription error: {e}")
        return None

def _pcm_to_wav_bytes(pcm_bytes: bytes, rate: int = 16000) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container."""
    import wave, io
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()

def _text_to_wav_macos(text: str) -> bytes:
    """macOS only: use say + afconvert to produce 16kHz mono 16-bit WAV."""
    if not text:
        return b""
    aiff = tempfile.mktemp(suffix='.aiff')
    wav  = tempfile.mktemp(suffix='.wav')
    try:
        subprocess.run(['say', '-o', aiff, '--', text],
                       check=True, capture_output=True, timeout=30)
        subprocess.run(
            ['afconvert', '-f', 'WAVE', '-d', 'LEI16@16000', '-c', '1', aiff, wav],
            check=True, capture_output=True, timeout=15
        )
        with open(wav, 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"[WARNING] TTS WAV generation failed: {e}")
        return b""
    finally:
        for p in [aiff, wav]:
            try:
                os.unlink(p)
            except OSError:
                pass


# ── ImuStateTracker ────────────────────────────────────────────────────────────

class ImuStateTracker:
    """
    Connects to robot TCP log port (8890) and tracks the latest IMU event.
    Provides a one-line context string for the LLM system prompt.
    This is a stub — events are already emitted; we just parse them here.
    """

    LOG_PORT = 8890

    def __init__(self, on_event=None):
        self.state: Dict[str, Any] = {
            "event": "LEVEL", "pitch": 0.0, "roll": 0.0, "accel": 1.0
        }
        self._lock = threading.Lock()
        self._robot_ip: Optional[str] = None
        self._on_event = on_event   # callback(event_name: str); called for non-LEVEL events

    def start(self, robot_ip: str):
        if robot_ip.lower() == "mock":
            return
        self._robot_ip = robot_ip
        t = threading.Thread(target=self._listen_loop, daemon=True)
        t.start()

    def _listen_loop(self):
        while True:
            try:
                with socket.create_connection((self._robot_ip, self.LOG_PORT), timeout=5) as s:
                    buf = ""
                    while True:
                        chunk = s.recv(1024).decode(errors="replace")
                        if not chunk:
                            break
                        buf += chunk
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            try:
                                evt = json.loads(line.strip())
                                if evt.get("type") == "imu_event":
                                    event_name = evt.get("event", "LEVEL")
                                    with self._lock:
                                        self.state.update({
                                            "event": event_name,
                                            "pitch": float(evt.get("pitch", 0.0)),
                                            "roll":  float(evt.get("roll",  0.0)),
                                            "accel": float(evt.get("accel", 1.0)),
                                        })
                                    if self._on_event and event_name not in ("LEVEL", None):
                                        threading.Thread(
                                            target=self._on_event,
                                            args=(event_name,), daemon=True
                                        ).start()
                            except (json.JSONDecodeError, ValueError):
                                pass
            except Exception:
                time.sleep(5)

    def context_string(self) -> str:
        with self._lock:
            s = self.state.copy()
        return (f"Robot sensor state: orientation={s['event']}, "
                f"pitch={s['pitch']:.1f}°, roll={s['roll']:.1f}°")


# ── VoiceInterface ─────────────────────────────────────────────────────────────

class VoiceInterface:
    """Handles laptop-side voice input (mic) and text-to-speech output."""

    def __init__(self, voice_enabled: bool = True, tts_engine: str = "pyttsx3",
                 gemini_api_key: Optional[str] = None, wake_word: str = "hey sesame"):
        self.voice_enabled = voice_enabled
        self.tts_engine_type = tts_engine
        self.gemini_api_key = gemini_api_key
        self.wake_word = wake_word.lower()

        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300   # low starting point; dynamic mode raises it
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8    # wait 0.8s of silence before ending phrase
        self.tts_lock = threading.Lock()

        # Lazy-load Whisper on first use
        self._whisper = None
        self._whisper_lock = threading.Lock()

        if self.tts_engine_type == "gemini" and not self.gemini_api_key:
            print("[WARNING] Gemini TTS selected but no API key provided. Falling back to pyttsx3.")
            self.tts_engine_type = "pyttsx3"

    def _get_whisper(self):
        with self._whisper_lock:
            if self._whisper is None:
                print("[INFO] Loading Whisper model (first use)...")
                self._whisper = _load_whisper()
        return self._whisper

    def listen(self, timeout: int = 5, cancel_event=None) -> Optional[str]:
        """Listen to laptop mic and transcribe locally with faster_whisper.

        cancel_event: optional threading.Event; if set before or after the blocking
        listen() call, returns None immediately without transcribing.
        Note: sr.Recognizer.listen() cannot be interrupted mid-block; cancellation
        takes effect as soon as the current phrase_time_limit (5s) or VAD silence
        window expires.
        """
        if not self.voice_enabled:
            return None
        if cancel_event and cancel_event.is_set():
            return None
        whisper = self._get_whisper()
        if not whisper:
            return None
        try:
            with sr.Microphone() as source:
                print("Listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
            if cancel_event and cancel_event.is_set():
                return None
            print("Transcribing...")
            return _transcribe(whisper, audio)
        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            print(f"[ERROR] Listen error: {e}")
            return None

    def listen_for_wake_word(self, timeout: int = 10) -> bool:
        """Listen for a short clip and check if it contains the wake word."""
        if not self.voice_enabled:
            return False
        whisper = self._get_whisper()
        if not whisper:
            return False
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=3)
            text = _transcribe(whisper, audio)
            if text and self.wake_word in text.lower():
                print(f"[OK] Wake word detected: {text!r}")
                return True
        except (sr.WaitTimeoutError, Exception):
            pass
        return False

    def speak(self, text: str, async_mode: bool = True, face: Optional[str] = None,
              robot_controller=None):
        """Speak text using TTS on the laptop with optional robot face animation."""
        if not self.voice_enabled:
            return
        if async_mode:
            threading.Thread(target=self._speak_sync, args=(text, face, robot_controller),
                             daemon=True).start()
        else:
            self._speak_sync(text, face, robot_controller)

    def _speak_sync(self, text: str, face: Optional[str] = None, robot_controller=None):
        try:
            with self.tts_lock:
                animation_thread = None
                stop_animation = threading.Event()

                if face and robot_controller:
                    robot_controller.send_command("idle", face)
                    time.sleep(0.2)
                    animation_thread = threading.Thread(
                        target=self._animate_talking_face,
                        args=(face, robot_controller, stop_animation),
                        daemon=True
                    )
                    animation_thread.start()
                    time.sleep(0.1)

                if self.tts_engine_type == "gemini":
                    self._speak_gemini(text)
                else:
                    self._speak_pyttsx3(text)

                if animation_thread:
                    stop_animation.set()
                    animation_thread.join(timeout=1)
                    if robot_controller:
                        robot_controller.send_command("idle", face)
        except Exception as e:
            print(f"[ERROR] TTS error: {e}")

    def _animate_talking_face(self, face: str, robot_controller, stop_event: threading.Event):
        if AUDIO_MONITORING_AVAILABLE:
            self._animate_with_audio_monitoring(face, robot_controller, stop_event)
        else:
            self._animate_time_based(face, robot_controller, stop_event)

    def _animate_with_audio_monitoring(self, face: str, robot_controller,
                                       stop_event: threading.Event):
        try:
            p = pyaudio.PyAudio()
            CHUNK = 1024
            THRESHOLD = 500
            SMOOTHING = 0.3
            try:
                stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                                input=True, frames_per_buffer=CHUNK)
            except Exception:
                p.terminate()
                self._animate_time_based(face, robot_controller, stop_event)
                return

            mouth_open = False
            last_update = 0
            smoothed_level = 0

            while not stop_event.is_set():
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float64)
                    rms = np.sqrt(np.mean(audio_data**2)) if len(audio_data) > 0 else 0
                    smoothed_level = SMOOTHING * smoothed_level + (1 - SMOOTHING) * rms

                    now = time.time()
                    should_open = smoothed_level > THRESHOLD
                    if should_open != mouth_open and (now - last_update) >= 0.05:
                        mouth_open = should_open
                        last_update = now
                        face_name = f"talk_{face}" if mouth_open else face
                        robot_controller.send_command("idle", face_name)
                    time.sleep(0.01)
                except Exception:
                    time.sleep(0.05)

            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            print(f"[WARNING] Audio monitoring error: {e}")
            self._animate_time_based(face, robot_controller, stop_event)

    def _animate_time_based(self, face: str, robot_controller, stop_event: threading.Event):
        syllable = 0.15
        while not stop_event.is_set():
            robot_controller.send_command("idle", f"talk_{face}")
            time.sleep(syllable)
            if stop_event.is_set():
                break
            robot_controller.send_command("idle", face)
            time.sleep(syllable)

    def _speak_pyttsx3(self, text: str):
        # macOS `say` avoids pyttsx3 NSSpeechSynthesizer GC/weakref crash on AppKit thread.
        try:
            subprocess.run(['say', '-r', '200', '--', text],
                           timeout=30, check=False, capture_output=True)
        except FileNotFoundError:
            # Not macOS — fall back to pyttsx3 engine
            try:
                import pyttsx3 as _pyttsx3
                _engine = _pyttsx3.init()
                _engine.setProperty('rate', 200)
                _engine.say(text)
                _engine.runAndWait()
            except Exception:
                pass
        except Exception as e:
            print(f"[WARNING] TTS say failed: {e}")

    def _speak_gemini(self, text: str):
        try:
            import pygame, wave
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.gemini_api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name='Laomedeia')))))

            if response.candidates and response.candidates[0].content.parts:
                audio_data = response.candidates[0].content.parts[0].inline_data.data
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    tmp = f.name
                with wave.open(tmp, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(audio_data)
                pygame.mixer.init(frequency=24000, channels=1)
                sound = pygame.mixer.Sound(tmp)
                sound.play()
                while pygame.mixer.get_busy():
                    pygame.time.Clock().tick(10)
                pygame.mixer.quit()
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            else:
                self._speak_pyttsx3(text)
        except Exception as e:
            print(f"[WARNING] Gemini TTS error: {e}, falling back to pyttsx3")
            self._speak_pyttsx3(text)


# ── SesameRobotController ──────────────────────────────────────────────────────

class SesameRobotController:
    """Controls the Sesame robot over WiFi.

    Commands go via a persistent TCP connection to port 8888 (Albert line-protocol)
    for minimal latency (~5ms vs ~200ms for HTTP+mDNS per call).
    Status queries still use HTTP GET /api/status.
    """

    TCP_PORT = 8888

    def __init__(self, robot_ip: str):
        self.robot_ip = robot_ip
        self.is_mock = robot_ip.lower() == "mock"
        self.base_url = f"http://{robot_ip}"
        self._sock: Optional[socket.socket] = None
        self._sock_lock = threading.Lock()
        if self.is_mock:
            print("[INFO] Robot Controller running in MOCK mode")

    def _connect(self) -> bool:
        """Open (or re-open) the persistent TCP socket. Returns True on success."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(3)
            s.connect((self.robot_ip, self.TCP_PORT))
            s.settimeout(None)
            self._sock = s
            print(f"[TCP] Connected to {self.robot_ip}:{self.TCP_PORT}")
            return True
        except Exception as e:
            print(f"[TCP] Connect failed: {e}")
            self._sock = None
            return False

    def _tcp_send(self, line: str):
        """Send a newline-terminated command string over the persistent socket."""
        with self._sock_lock:
            if self._sock is None and not self._connect():
                raise OSError("TCP not connected")
            try:
                self._sock.sendall((line + "\n").encode())
            except Exception:
                # Stale connection — reconnect once and retry
                self._sock = None
                if self._connect():
                    self._sock.sendall((line + "\n").encode())
                else:
                    raise

    def send_command(self, command: str, face: Optional[str] = None) -> Dict[str, Any]:
        print(f"   TX: command={command!r} face={face!r}")
        if self.is_mock:
            return {"status": "success", "mock": True}
        try:
            # Send face change first so it's visible while the motion starts
            if face and command != "idle":
                self._tcp_send(f"face {face}")
            if command and command != "idle":
                self._tcp_send(command)
            elif face:
                self._tcp_send(f"face {face}")
            return {"status": "ok"}
        except Exception as e:
            print(f"[TCP] send_command failed: {e}")
            return {"error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        if self.is_mock:
            return {"currentCommand": "idle", "currentFace": "happy",
                    "networkConnected": True, "mock": True}
        try:
            r = requests.get(f"{self.base_url}/api/status", timeout=5)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def stop(self) -> Dict[str, Any]:
        return self.send_command("stop")


# ── SesameMemory ───────────────────────────────────────────────────────────────

class SesameMemory:
    """Three-tier persistence: response cache, user profile, session summaries."""

    SESAME_DIR = pathlib.Path.home() / ".sesame"
    MAX_CACHE = 500

    def __init__(self):
        self.SESAME_DIR.mkdir(exist_ok=True)
        self._cache: dict = self._load_json("cache.json", {})
        self._profile: dict = self._load_json(
            "profile.json",
            {"name": None, "favorites": [], "top_commands": {}}
        )

    def _load_json(self, fname: str, default: dict) -> dict:
        p = self.SESAME_DIR / fname
        try:
            if p.exists():
                return json.loads(p.read_text())
        except Exception:
            pass
        return dict(default)

    def _save_json(self, fname: str, data: dict):
        try:
            (self.SESAME_DIR / fname).write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[Memory] save {fname} failed: {e}")

    def _normalize(self, text: str) -> str:
        return re.sub(r'[^\w\s]', '', text.lower()).strip()

    def cache_get(self, text: str) -> Optional[dict]:
        key = self._normalize(text)
        entry = self._cache.get(key)
        if entry:
            entry["hits"] = entry.get("hits", 0) + 1
            entry["last_used"] = time.time()
            return entry
        return None

    def cache_set(self, text: str, result: dict):
        key = self._normalize(text)
        if len(self._cache) >= self.MAX_CACHE:
            oldest_key = min(self._cache, key=lambda k: self._cache[k].get("last_used", 0))
            del self._cache[oldest_key]
        self._cache[key] = {
            "command": result.get("command"),
            "face": result.get("face"),
            "response": result.get("response"),
            "hits": 0,
            "last_used": time.time(),
        }
        self._save_json("cache.json", self._cache)

    def update_profile_from_text(self, text: str):
        changed = False
        m = re.search(r"(?:my name is|i'?m|call me)\s+([A-Za-z]+)", text, re.IGNORECASE)
        if m and not self._profile.get("name"):
            self._profile["name"] = m.group(1).capitalize()
            changed = True
        m2 = re.search(r"(?:i love|i like|my favorite\w* is)\s+(.+?)(?:\.|!|\?|$)", text, re.IGNORECASE)
        if m2:
            fav = m2.group(1).strip()
            if fav and fav not in self._profile.get("favorites", []):
                self._profile.setdefault("favorites", []).append(fav)
                changed = True
        if changed:
            self._save_json("profile.json", self._profile)

    def update_command_count(self, command: str):
        if command:
            tc = self._profile.setdefault("top_commands", {})
            tc[command] = tc.get(command, 0) + 1
            self._save_json("profile.json", self._profile)

    def profile_context(self) -> str:
        parts = []
        if self._profile.get("name"):
            parts.append(f"The child's name is {self._profile['name']}.")
        if self._profile.get("favorites"):
            parts.append(f"Their favorites: {', '.join(self._profile['favorites'][:3])}.")
        return " ".join(parts)


# ── LocalLLMInterface ──────────────────────────────────────────────────────────

class LocalLLMInterface:
    """Interface for local LLM (Ollama) via OpenAI-compatible API."""

    MAX_HISTORY = 8   # 4 exchanges (user + assistant per exchange)

    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self._history: list = []

    def clear_history(self):
        self._history = []

    def interpret_command(self, user_input: str,
                          imu_context: str = "",
                          memory_context: str = "") -> Dict[str, Any]:
        try:
            system = SHORT_SYSTEM_PROMPT
            if memory_context:
                system += f"\n\n{memory_context}"
            if imu_context:
                system += f"\n\nContext: {imu_context}"

            messages = [{"role": "system", "content": system}]
            messages.extend(self._history)
            messages.append({"role": "user", "content": f"User: {user_input}\n\nRespond with JSON only:"})

            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.4,
                "think": False,
                "stream": False,
                "format": "json",
                "response_format": {"type": "json_object"},
            }

            r = requests.post(url, json=payload,
                              headers={"Content-Type": "application/json"}, timeout=30)
            if r.status_code != 200:
                payload.pop("response_format", None)
                r = requests.post(url, json=payload,
                                  headers={"Content-Type": "application/json"}, timeout=30)
            if r.status_code != 200:
                return {"response": f"Local AI Error: {r.status_code}"}

            content = r.json()['choices'][0]['message']['content']
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            parsed = json.loads(content.strip())
            result = _normalize_llm(parsed)

            # Append to sliding history window
            self._history.append({"role": "user", "content": user_input})
            self._history.append({"role": "assistant", "content": result.get("response", "")})
            if len(self._history) > self.MAX_HISTORY:
                self._history = self._history[-self.MAX_HISTORY:]

            return result

        except Exception as e:
            return {"response": f"Local AI connection failed: {e}"}


# ── GeminiInterface ────────────────────────────────────────────────────────────

class GeminiInterface:
    """Optional Google Gemini backend. Used when SESAME_LOCAL=false."""

    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')

    def interpret_command(self, user_input: str,
                          imu_context: str = "") -> Dict[str, Any]:
        try:
            system = SYSTEM_PROMPT
            if imu_context:
                system += f"\n\nContext: {imu_context}"
            prompt = f"{system}\n\nUser: {user_input}\n\nRespond with JSON only:"
            text = self.model.generate_content(prompt).text.strip()
            for prefix in ("```json", "```"):
                if text.startswith(prefix):
                    text = text[len(prefix):]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text.strip())
        except Exception as e:
            return {"response": f"Something went wrong: {e}"}


# ── RobotVoiceReceiver ─────────────────────────────────────────────────────────

class RobotVoiceReceiver:
    """
    TCP server (port 8889) that receives on-device wake word audio from the robot.

    Flow:
      1. Robot detects 'Hey Willow' via ESP-SR WakeNet
      2. Robot records 4s of PCM and streams [uint32 len][PCM] to this server
      3. This server transcribes with faster_whisper
      4. Runs LocalLLMInterface → {command, face, response}
      5. Generates TTS WAV via macOS say + afconvert (16kHz mono 16-bit)
      6. Sends [uint32 wav_len][WAV bytes] back on the same TCP connection
      7. Robot plays WAV on its MAX98357A speaker
      8. Sends HTTP /api/command to robot for movement/face (runs concurrently)
      9. Calls on_interaction(user_text, result) to update the GUI
    """

    LISTEN_PORT = 8889
    PCM_RATE    = 16000

    def __init__(self, llm, robot: SesameRobotController,
                 on_interaction=None):
        self.llm = llm
        self.robot = robot
        self.on_interaction = on_interaction   # callback(user_text: str, result: dict)
        self._whisper = None
        self._server: Optional[socket.socket] = None
        self._running = False

    def start(self):
        self._whisper = _load_whisper()
        if not self._whisper:
            print("[WARNING] RobotVoiceReceiver: faster_whisper unavailable — robot voice disabled")
            return

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("0.0.0.0", self.LISTEN_PORT))
        self._server.listen(4)
        self._running = True

        t = threading.Thread(target=self._serve_loop, daemon=True)
        t.start()
        print(f"[RobotVoice] Listening on port {self.LISTEN_PORT} for robot wake word audio")

    def stop(self):
        self._running = False
        if self._server:
            try:
                self._server.close()
            except Exception:
                pass

    def _serve_loop(self):
        while self._running:
            try:
                conn, addr = self._server.accept()
                threading.Thread(target=self._handle_connection,
                                 args=(conn, addr), daemon=True).start()
            except Exception:
                if self._running:
                    time.sleep(1)

    def _handle_connection(self, conn: socket.socket, addr):
        robot_ip = addr[0]
        print(f"[RobotVoice] Connection from {robot_ip}")
        try:
            conn.settimeout(20.0)

            # Receive PCM
            pcm_len = struct.unpack("<I", self._recv_exact(conn, 4))[0]
            print(f"[RobotVoice] Receiving {pcm_len} bytes ({pcm_len/32000:.1f}s) PCM")
            pcm = self._recv_exact(conn, pcm_len)

            # STT — no vad_filter: robot already confirmed wake word, so audio
            # contains real speech; vad_filter discards short commands like "crab walk"
            audio_np = (np.frombuffer(pcm, dtype=np.int16)
                        .astype(np.float32) / 32768.0)
            segments, _ = self._whisper.transcribe(
                audio_np, language="en",
                initial_prompt=_WHISPER_PROMPT)
            user_text = " ".join(s.text for s in segments).strip()
            print(f"[RobotVoice] Heard: {user_text!r}")

            if not user_text:
                conn.sendall(struct.pack("<I", 0))
                return

            # LLM
            result        = _normalize_llm(self.llm.interpret_command(user_text))
            if result.get("command") is None:
                inferred = _infer_command(user_text)
                if inferred:
                    result["command"] = inferred
                    print(f"[RobotVoice] Inferred command from speech: {inferred!r}")
            response_text = result.get("response") or ""
            command       = result.get("command")
            face          = result.get("face")
            print(f"[RobotVoice] LLM → cmd={command!r} face={face!r} resp={response_text!r}")

            # TTS → WAV
            wav_bytes = _text_to_wav_macos(response_text)
            print(f"[RobotVoice] TTS → {len(wav_bytes)} bytes WAV")

            # Send WAV back to robot (plays on robot speaker)
            conn.sendall(struct.pack("<I", len(wav_bytes)))
            if wav_bytes:
                conn.sendall(wav_bytes)
                print(f"[RobotVoice] WAV sent to robot")

            # Send command/face to robot via HTTP (non-blocking)
            if command and command in AVAILABLE_COMMANDS:
                threading.Thread(
                    target=self.robot.send_command,
                    args=(command, face),
                    daemon=True
                ).start()
            elif face and face in AVAILABLE_FACES:
                threading.Thread(
                    target=self.robot.send_command,
                    args=("idle", face),
                    daemon=True
                ).start()

            # Notify GUI
            if self.on_interaction:
                self.on_interaction(user_text, result)

        except Exception as e:
            print(f"[RobotVoice] Error handling {robot_ip}: {e}")
            try:
                conn.sendall(struct.pack("<I", 0))
            except Exception:
                pass
        finally:
            conn.close()

    @staticmethod
    def _recv_exact(conn: socket.socket, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("connection closed prematurely")
            buf += chunk
        return buf


# ── SesameCompanionApp ─────────────────────────────────────────────────────────

class SesameCompanionApp:
    """Main application: LLM + robot control + voice + robot voice receiver."""

    IDLE_PHRASE_SECS = 180   # 3 min → speak idle phrase
    IDLE_SLEEP_SECS  = 300   # 5 min → robot sleep

    def __init__(self, robot_ip: str, sesame_local: bool, gemini_api_key: str,
                 voice_enabled: bool = True, tts_engine: str = "pyttsx3",
                 wake_word: str = "hey sesame", wake_word_mode: bool = False):
        self.robot = SesameRobotController(robot_ip)

        if sesame_local:
            local_url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1")
            if "11434" in local_url and "/v1" not in local_url and "/chat" not in local_url:
                print("[INFO] Detected Ollama port without /v1, appending /v1")
                local_url = f"{local_url.rstrip('/')}/v1"
            local_model = os.getenv("LOCAL_LLM_MODEL", "llama3.2")
            print(f"[INFO] Using Local AI: {local_model} at {local_url}")
            self.ai = LocalLLMInterface(local_url, local_model)
        else:
            self.ai = GeminiInterface(gemini_api_key)

        self.voice = VoiceInterface(voice_enabled, tts_engine, gemini_api_key, wake_word)
        self.voice_mode = voice_enabled
        self.tts_engine = tts_engine
        self.wake_word_mode = wake_word_mode

        # Persistent memory (cache / profile / session summaries)
        self.memory = SesameMemory()

        # Kids content layer
        self._kids_mode = os.getenv("KIDS_MODE", "true").lower() == "true"
        if self._kids_mode:
            try:
                from sesame_companion_kids import KidsCommandLayer
                self._kids = KidsCommandLayer()
                print("[INFO] Kids content layer enabled")
            except ImportError:
                self._kids = None
                print("[WARNING] Kids content layer unavailable (sesame_companion_kids.py not found)")
        else:
            self._kids = None

        # Idle/sleep state
        self._sleeping = False
        self._last_interaction = time.time()
        self._idle_phrase_fired = False

        # IMU state — on_event resets idle timer and wakes robot
        self.imu = ImuStateTracker(on_event=self._on_imu_event)
        self.imu.start(robot_ip)

        # Robot voice receiver (handles ESP-SR wake word audio from robot)
        self.robot_voice = RobotVoiceReceiver(
            llm=self.ai,
            robot=self.robot,
            on_interaction=None   # set by GUI: app.robot_voice.on_interaction = callback
        )

        # Start idle monitor thread
        t = threading.Thread(target=self._idle_loop, daemon=True)
        t.start()

    def start_robot_voice_receiver(self):
        """Start the TCP server that receives audio from the robot's wake word."""
        self.robot_voice.start()

    def _reset_idle(self):
        self._last_interaction = time.time()
        self._idle_phrase_fired = False

    def _wake_robot(self):
        if self._sleeping:
            print("[Idle] Waking robot")
            self.robot.send_command("wake")
            time.sleep(0.3)
            self._sleeping = False

    def _on_imu_event(self, event_name: str):
        """Called from IMU listener thread on PICKUP, TAPPED, etc."""
        self._reset_idle()
        if event_name in ("PICKUP", "TAPPED"):
            self._wake_robot()

    def _idle_loop(self):
        _IDLE_PHRASES = [
            "Is anyone there?",
            "I'm bored — wanna play?",
            "Hey, come play with me!",
            "Helloooo? I miss you!",
        ]
        while True:
            time.sleep(10)
            idle_secs = time.time() - self._last_interaction
            if self._sleeping:
                continue
            if idle_secs >= self.IDLE_SLEEP_SECS:
                print("[Idle] 5 min idle — putting robot to sleep")
                self.robot.send_command("sleep")
                self._sleeping = True
                if hasattr(self.ai, 'clear_history'):
                    self.ai.clear_history()
            elif idle_secs >= self.IDLE_PHRASE_SECS and not self._idle_phrase_fired:
                print("[Idle] 3 min idle — showing bored face")
                self.robot.send_command("idle", "confused")
                self._idle_phrase_fired = True

    def process_input(self, user_input: str) -> tuple:
        """Process laptop-typed/spoken input through AI and control robot."""
        self._reset_idle()
        self._wake_robot()

        # Kids content layer (pre-LLM, ~0ms latency)
        if self._kids:
            kids_result = self._kids.match(user_input)
            if kids_result:
                interpretation = kids_result
                print(f"[Kids] Match: {interpretation}")
                command = interpretation.get("command")
                face = interpretation.get("face")
                response_text = interpretation.get("response", "")
                if command and command in AVAILABLE_COMMANDS:
                    self.robot.send_command(command, face)
                elif face and face in AVAILABLE_FACES:
                    self.robot.send_command("idle", face)
                out = "[Kids] "
                if response_text:
                    out += f"Sesame says: {response_text}"
                if command:
                    out += f"\nAction: {command}"
                return (out, interpretation)

        # Response cache (exact-match instant recall)
        cached = self.memory.cache_get(user_input)
        if cached:
            print(f"[Cache] Hit for: {user_input!r}")
            interpretation = cached
            command = cached.get("command")
            face = cached.get("face")
            if command and command in AVAILABLE_COMMANDS:
                self.robot.send_command(command, face)
            elif face and face in AVAILABLE_FACES:
                self.robot.send_command("idle", face)
            return (f"[Cache] Sesame says: {cached.get('response', '')}", interpretation)

        # Profile context for richer LLM responses
        mem_ctx = self.memory.profile_context()
        imu_ctx = self.imu.context_string()

        if hasattr(self.ai, 'interpret_command'):
            import inspect
            sig = inspect.signature(self.ai.interpret_command)
            if 'memory_context' in sig.parameters:
                interpretation = _normalize_llm(self.ai.interpret_command(user_input, imu_ctx, mem_ctx))
            else:
                interpretation = _normalize_llm(self.ai.interpret_command(user_input, imu_ctx))
        else:
            interpretation = _normalize_llm(self.ai.interpret_command(user_input, imu_ctx))

        if interpretation.get("command") is None:
            inferred = _infer_command(user_input)
            if inferred:
                interpretation["command"] = inferred

        # Update profile and cache
        self.memory.update_profile_from_text(user_input)
        if interpretation.get("command"):
            self.memory.update_command_count(interpretation["command"])
        self.memory.cache_set(user_input, interpretation)

        # Conversational response (face only)
        if "response" in interpretation and not interpretation.get("command"):
            ai_response = interpretation["response"]
            face = interpretation.get("face", "")
            if face and face in AVAILABLE_FACES:
                self.robot.send_command("idle", face)
                ai_response += f" [{face} face]"
            return (ai_response, interpretation)

        # Execute robot command
        if "command" in interpretation and interpretation["command"]:
            command = interpretation["command"]
            face = interpretation.get("face") or None
            ai_response = interpretation.get("response", "")

            if command not in AVAILABLE_COMMANDS:
                return (f"Unknown command: {command}", interpretation)

            result = self.robot.send_command(command, face)
            if "error" in result:
                return (f"[ERROR] Robot: {result['error']}", interpretation)

            out = "[OK] Command sent!"
            if ai_response:
                out += f"\nSesame says: {ai_response}"
            out += f"\nAction: {command}"
            if face:
                out += f" + {face} face"
            return (out, interpretation)

        return ("I'm not sure what to do with that.", interpretation)

    def run_interactive(self):
        """CLI interactive mode."""
        print("=" * 60)
        print("Sesame Robot Companion App")
        print("=" * 60)

        status = self.robot.get_status()
        if "error" in status:
            print(f"[ERROR] Cannot connect: {status['error']}")
            if input("Continue anyway? (y/n): ").strip().lower() != 'y':
                return
        else:
            print(f"[OK] Connected — Face: {status.get('currentFace')}")

        self.start_robot_voice_receiver()

        while True:
            try:
                if self.voice_mode and self.wake_word_mode:
                    user_input = input("[Type or say wake word]: ").strip()
                    if not user_input:
                        print(f"Listening for '{self.voice.wake_word}'...")
                        if self.voice.listen_for_wake_word(timeout=30):
                            user_input = self.voice.listen(timeout=10) or ""
                            if user_input:
                                print(f"You said: {user_input}")
                            else:
                                continue
                        else:
                            continue
                elif self.voice_mode:
                    user_input = input("[Press Enter to speak, or type]: ").strip()
                    if not user_input:
                        user_input = self.voice.listen() or ""
                        if user_input:
                            print(f"You said: {user_input}")
                        else:
                            continue
                else:
                    user_input = input("You: ").strip()

                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit"):
                    self.voice.speak("Goodbye!", async_mode=False)
                    break

                print("Thinking...")
                response, interpretation = self.process_input(user_input)
                print(f"\n{response}\n")

                if self.voice_mode and "response" in interpretation:
                    face = (interpretation.get("face")
                            if not interpretation.get("command") else None)
                    self.voice.speak(interpretation["response"], async_mode=True,
                                     face=face, robot_controller=self.robot)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"[ERROR] {e}")


def main():
    robot_ip     = os.getenv("SESAME_ROBOT_IP")
    sesame_local = os.getenv("SESAME_LOCAL", "true").lower() == "true"   # local by default
    gemini_key   = os.getenv("GEMINI_API_KEY", "")
    voice        = os.getenv("VOICE_ENABLED", "true").lower() == "true"
    tts          = os.getenv("TTS_ENGINE", "pyttsx3")
    wake_word    = os.getenv("WAKE_WORD", "hey sesame")
    wake_mode    = os.getenv("WAKE_WORD_MODE", "false").lower() == "true"

    if not robot_ip:
        robot_ip = input("Enter robot IP (e.g. quadruped.local) or 'mock': ").strip()
        if not robot_ip:
            print("Robot IP is required!")
            sys.exit(1)

    if not sesame_local and not gemini_key:
        gemini_key = input("Enter your Gemini API key: ").strip()
        if not gemini_key:
            sys.exit(1)

    app = SesameCompanionApp(robot_ip, sesame_local, gemini_key,
                             voice, tts, wake_word, wake_mode)
    app.run_interactive()


if __name__ == "__main__":
    main()
