#!/usr/bin/env python3
"""
sesame_vision_content.py — game state machines for vision-initiated games.

Each class manages the sequence of announcements and robot commands for one game.
The companion app creates the appropriate content object when a game starts and
calls tick() each time the camera loop fires.  The content object emits
(robot_command, speak_text) tuples; the caller is responsible for executing them.
"""

import time
import random
from typing import Optional, Tuple

Command = Optional[str]   # robot command string or None
Text    = Optional[str]   # text to speak or None
Step    = Tuple[Command, Text]


# ─────────────────────────────────────────────────────────────────────────────
# HideAndSeekContent
# Robot closes eyes → counts to 5 → starts searching (rotate slowly)
# Vision layer calls seek_found() when the target is centred; game ends.
# ─────────────────────────────────────────────────────────────────────────────

class HideAndSeekContent:
    STATES = ["eyes_closed", "counting", "searching", "found"]

    def __init__(self, robot, voice):
        self.robot = robot
        self.voice = voice
        self._state = "eyes_closed"
        self._started = time.time()
        self._count = 0

    def start(self):
        self.robot.send_command("idle", "sleepy")
        self.voice.speak("Close your eyes! I'm counting to five!", async_mode=True,
                         robot_controller=self.robot)
        self._state   = "counting"
        self._started = time.time()
        self._count   = 0

    def tick(self) -> Step:
        now = time.time()
        elapsed = now - self._started

        if self._state == "counting":
            count_step = int(elapsed)
            if count_step != self._count and count_step <= 5:
                self._count = count_step
                self.voice.speak(str(count_step), async_mode=True,
                                 robot_controller=self.robot)
            if elapsed >= 6:
                self._state   = "searching"
                self._started = now
                self.voice.speak("Ready or not, here I come!", async_mode=True,
                                 robot_controller=self.robot)
                self.robot.send_command("idle", "excited")
                return "left", None
            return None, None

        if self._state == "searching":
            if elapsed > 30:
                self._state = "found"
                return "stop", None
            return "left", None

        return None, None

    def seek_found(self):
        self._state = "found"
        self.robot.send_command("wave")
        self.voice.speak("Found you! I win!", async_mode=True,
                         robot_controller=self.robot)


# ─────────────────────────────────────────────────────────────────────────────
# RedLightGreenLightContent
# Robot alternates green (dance) and red (freeze) phases; CV checks for motion.
# ─────────────────────────────────────────────────────────────────────────────

class RedLightGreenLightContent:
    def __init__(self, robot, voice):
        self.robot  = robot
        self.voice  = voice
        self._phase = "green"
        self._phase_start = time.time()
        self._green_secs  = 3.5
        self._red_secs    = 2.5
        self._rounds      = 0
        self._max_rounds  = 5

    def start(self):
        self.voice.speak("Green light! Go go go!", async_mode=True,
                         robot_controller=self.robot)
        self.robot.send_command("dance")
        self._phase       = "green"
        self._phase_start = time.time()

    def tick(self, child_moving: bool) -> Step:
        elapsed = time.time() - self._phase_start

        if self._phase == "green":
            if elapsed >= self._green_secs:
                self._phase       = "red"
                self._phase_start = time.time()
                self.voice.speak("Red light! Freeze!", async_mode=True,
                                 robot_controller=self.robot)
                self.robot.send_command("stop")
                self.robot.send_command("idle", "excited")
                return "freeze_check", None
            return None, None

        if self._phase == "red":
            if child_moving:
                self.robot.send_command("idle", "surprised")
                self.voice.speak("You moved! You're out!", async_mode=True,
                                 robot_controller=self.robot)
                self._phase = "done"
                return "stop", None
            if elapsed >= self._red_secs:
                self._rounds += 1
                if self._rounds >= self._max_rounds:
                    self.voice.speak("You made it! You win!", async_mode=True,
                                     robot_controller=self.robot)
                    self.robot.send_command("wave")
                    self._phase = "done"
                    return "stop", None
                self._phase       = "green"
                self._phase_start = time.time()
                self.voice.speak("Green light!", async_mode=True,
                                 robot_controller=self.robot)
                self.robot.send_command("dance")

        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# FreezeDanceContent
# Music plays (dance), robot calls freeze, CV checks child stopped.
# ─────────────────────────────────────────────────────────────────────────────

class FreezeDanceContent:
    _DANCE_PHRASES = [
        "Boogie time! Dance!",
        "Move those legs!",
        "Shake it shake it!",
    ]
    _FREEZE_PHRASES = ["Freeze!", "Stop! Don't move!", "Freeze dance!"]

    def __init__(self, robot, voice):
        self.robot        = robot
        self.voice        = voice
        self._phase       = "dance"
        self._phase_start = time.time()
        self._dance_secs  = random.uniform(3.0, 5.5)
        self._freeze_secs = 2.0
        self._rounds      = 0

    def start(self):
        phrase = random.choice(self._DANCE_PHRASES)
        self.voice.speak(phrase, async_mode=True, robot_controller=self.robot)
        self.robot.send_command("dance")
        self._phase       = "dance"
        self._phase_start = time.time()

    def tick(self, child_moving: bool) -> Step:
        elapsed = time.time() - self._phase_start

        if self._phase == "dance":
            if elapsed >= self._dance_secs:
                self._phase       = "freeze"
                self._phase_start = time.time()
                self._dance_secs  = random.uniform(3.0, 5.5)
                self.robot.send_command("stop")
                self.robot.send_command("idle", "excited")
                self.voice.speak(random.choice(self._FREEZE_PHRASES),
                                 async_mode=True, robot_controller=self.robot)
            return None, None

        if self._phase == "freeze":
            if child_moving:
                self.robot.send_command("idle", "surprised")
                self.voice.speak("You moved! You're out!", async_mode=True,
                                 robot_controller=self.robot)
                self._phase = "done"
                return "stop", None
            if elapsed >= self._freeze_secs:
                self._rounds     += 1
                self._phase       = "dance"
                self._phase_start = time.time()
                if self._rounds >= 4:
                    self.voice.speak("Amazing dancer! You win!", async_mode=True,
                                     robot_controller=self.robot)
                    self.robot.send_command("wave")
                    self._phase = "done"
                    return "stop", None
                phrase = random.choice(self._DANCE_PHRASES)
                self.voice.speak(phrase, async_mode=True, robot_controller=self.robot)
                self.robot.send_command("dance")

        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# ColorTeacherContent
# Sequential scavenger hunt: find blue → find yellow → find red → done
# ─────────────────────────────────────────────────────────────────────────────

_COLOUR_SEQUENCE = ["red", "blue", "green", "yellow", "orange"]

class ColorTeacherContent:
    def __init__(self, robot, voice, vision_receiver):
        self.robot    = robot
        self.voice    = voice
        self.vision   = vision_receiver
        self._seq     = list(_COLOUR_SEQUENCE)
        self._idx     = 0
        self._waiting = False

    def start(self):
        self._idx     = 0
        self._waiting = False
        self._ask_next()

    def _ask_next(self):
        if self._idx >= len(self._seq):
            self.voice.speak("You found them all! Great job!", async_mode=True,
                             robot_controller=self.robot)
            self.robot.send_command("wave")
            self.vision.set_passive()
            return
        colour = self._seq[self._idx]
        self.voice.speak(f"Find something {colour}! Show it to me!",
                         async_mode=True, robot_controller=self.robot)
        self.robot.send_command("idle", "excited")
        self.vision.set_active("find", colour)
        self._waiting = True

    def on_found(self):
        colour = self._seq[self._idx]
        self.voice.speak(f"Yes! That's {colour}! Great!", async_mode=True,
                         robot_controller=self.robot)
        self.robot.send_command("wave")
        self._idx     += 1
        self._waiting  = False
        time.sleep(1.5)
        self._ask_next()


# ─────────────────────────────────────────────────────────────────────────────
# ShowAndTellContent
# Capture a single JPEG, describe it via vision LLM (future) or colour summary
# ─────────────────────────────────────────────────────────────────────────────

_SHOW_RESPONSES = [
    "Oh wow, I see something colourful!",
    "That looks really cool!",
    "I see lots of shapes!",
    "That is so interesting!",
    "Wow, show me more!",
]

class ShowAndTellContent:
    def __init__(self, robot, voice):
        self.robot = robot
        self.voice = voice

    def on_frame(self, jpeg: bytes):
        """Called with the first frame received in show_tell mode."""
        # Future: send jpeg to vision LLM here.
        # For now, pick a random enthusiastic response.
        response = random.choice(_SHOW_RESPONSES)
        self.voice.speak(response, async_mode=True, robot_controller=self.robot)
        self.robot.send_command("idle", "love")
