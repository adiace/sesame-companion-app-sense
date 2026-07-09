#!/usr/bin/env python3
"""
sesame_vision.py — camera vision processing for the sesame-robot-sense companion app.

Two modes:
  passive (1fps)  — always-on background; auto-fires reactions (wave back, peek-a-boo, face close)
  active  (3fps)  — voice-initiated; tracks objects/people; sends movement commands

Architecture:
  RobotVisionReceiver  listens on TCP :8891 for JPEG frames from the robot,
                       runs VisionProcessor on each frame,
                       sends movement / reaction commands back on the same connection.
  VisionCommandLayer   pre-LLM regex matcher; called from process_input() before kids layer.
"""

import os
import re
import time
import struct
import socket
import threading
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# ── OpenCV import (optional at module load — fail gracefully if not installed) ──
try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("[Vision] opencv-python not installed — vision features disabled")


# ─────────────────────────────────────────────────────────────────────────────
# VisionProcessor — all OpenCV logic; stateless per-frame calls
# ─────────────────────────────────────────────────────────────────────────────

class VisionProcessor:
    """CPU-only OpenCV processing: colour detection, face/person, motion, fingers."""

    # HSV ranges for common toy colours. Red wraps the hue axis → two ranges.
    COLOR_RANGES = {
        "red":    [(0, 120, 70, 10, 255, 255), (170, 120, 70, 180, 255, 255)],
        "blue":   [(100, 150, 50, 130, 255, 255)],
        "green":  [(40, 70, 50, 80, 255, 255)],
        "yellow": [(20, 100, 100, 35, 255, 255)],
        "orange": [(10, 100, 100, 20, 255, 255)],
        "pink":   [(140, 50, 100, 170, 255, 255)],
        "purple": [(125, 50, 50, 145, 255, 255)],
    }

    def __init__(self):
        if _CV2_AVAILABLE:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            self._bg_sub = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=40)
        self._dark_counter = 0          # consecutive dark frames (peek-a-boo)
        self._was_dark = False

    # ── helpers ───────────────────────────────────────────────────────────────

    def _jpeg_to_bgr(self, jpeg: bytes) -> Optional["np.ndarray"]:
        if not _CV2_AVAILABLE:
            return None
        arr = np.frombuffer(jpeg, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame

    def _color_mask(self, hsv: "np.ndarray", color: str) -> "np.ndarray":
        ranges = self.COLOR_RANGES.get(color, self.COLOR_RANGES["red"])
        mask = None
        for r in ranges:
            lo = np.array([r[0], r[1], r[2]])
            hi = np.array([r[3], r[4], r[5]])
            m = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # ── active tracking ───────────────────────────────────────────────────────

    def find_color_object(self, jpeg: bytes, color: str) -> dict:
        """Largest colour blob → normalised cx (-1..1), cy (-1..1), area (0..1)."""
        frame = self._jpeg_to_bgr(jpeg)
        if frame is None:
            return {"found": False, "cx": 0, "cy": 0, "area": 0, "annotated": jpeg}

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self._color_mask(hsv, color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = {"found": False, "cx": 0.0, "cy": 0.0, "area": 0.0, "annotated": jpeg}

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area_px = cv2.contourArea(largest)
            if area_px > 300:
                x, y, bw, bh = cv2.boundingRect(largest)
                cx_px = x + bw // 2
                cy_px = y + bh // 2
                result["found"] = True
                result["cx"]    = (cx_px / w) * 2 - 1   # -1 (left) .. +1 (right)
                result["cy"]    = (cy_px / h) * 2 - 1   # -1 (top)  .. +1 (bottom)
                result["area"]  = area_px / (w * h)

                # Annotate frame
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                label = f"{color} ({result['area']:.2f})"
                cv2.putText(frame, label, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                result["annotated"] = bytes(enc)

        return result

    def find_person(self, jpeg: bytes) -> dict:
        """Motion blob (MOG2) or face cascade to find and track a person."""
        frame = self._jpeg_to_bgr(jpeg)
        if frame is None:
            return {"found": False, "cx": 0.0, "area": 0.0, "annotated": jpeg}

        h, w = frame.shape[:2]
        result = {"found": False, "cx": 0.0, "area": 0.0, "annotated": jpeg}

        # Try face cascade first (more reliable centre-of-person)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces):
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            cx_px = fx + fw // 2
            result["found"] = True
            result["cx"]    = (cx_px / w) * 2 - 1
            result["area"]  = (fw * fh) / (w * h)
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 165, 0), 2)
            cv2.putText(frame, "person", (fx, fy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            result["annotated"] = bytes(enc)
        else:
            # Fall back to motion blob
            fg = self._bg_sub.apply(frame)
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 1000:
                    x, y, bw, bh = cv2.boundingRect(largest)
                    cx_px = x + bw // 2
                    result["found"] = True
                    result["cx"]    = (cx_px / w) * 2 - 1
                    result["area"]  = cv2.contourArea(largest) / (w * h)
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 200, 255), 2)
                    _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    result["annotated"] = bytes(enc)

        return result

    def count_fingers(self, jpeg: bytes) -> int:
        """Estimate finger count from skin-coloured convex hull defects."""
        frame = self._jpeg_to_bgr(jpeg)
        if frame is None:
            return 0

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Skin tone range (works under typical indoor lighting)
        skin_lo = np.array([0,  20,  70])
        skin_hi = np.array([20, 180, 255])
        mask = cv2.inRange(hsv, skin_lo, skin_hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0

        hand = max(contours, key=cv2.contourArea)
        if cv2.contourArea(hand) < 3000:
            return 0

        hull = cv2.convexHull(hand, returnPoints=False)
        if hull is None or len(hull) < 3:
            return 0

        try:
            defects = cv2.convexityDefects(hand, hull)
        except cv2.error:
            return 0

        if defects is None:
            return 0

        fingers = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            depth = d / 256.0
            if depth > 10:
                fingers += 1

        return min(fingers + 1, 5)   # defects = gaps between fingers; add 1 for thumb side

    def detection_to_command(self, det: dict, mode: str) -> str:
        """Convert detection result to a robot movement command."""
        if not det["found"]:
            return "left"   # rotate to search

        cx   = det["cx"]
        area = det["area"]

        dead_zone = 0.18
        if mode == "find":
            target_area = 0.25   # stop when object fills ~25% of frame
        else:
            target_area = 0.15   # follow/chase: keep a bit of distance

        if area >= target_area:
            return "stop"   # close enough
        if cx < -dead_zone:
            return "left"
        if cx > dead_zone:
            return "right"
        return "forward"

    # ── passive detection ─────────────────────────────────────────────────────

    def check_peekaboo(self, jpeg: bytes) -> bool:
        """Returns True on the frame where lens goes from dark back to bright."""
        frame = self._jpeg_to_bgr(jpeg)
        if frame is None:
            return False

        brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        if brightness < 30:
            self._dark_counter += 1
            self._was_dark = self._dark_counter >= 2   # must be dark for ≥2 frames (~2s)
        else:
            if self._was_dark and brightness > 60:
                self._dark_counter = 0
                self._was_dark = False
                return True
            self._dark_counter = 0

        return False

    def check_wave(self, jpeg: bytes, prev_jpeg: Optional[bytes]) -> bool:
        """Large motion blob in upper half of frame — wave detection."""
        if prev_jpeg is None:
            return False
        frame = self._jpeg_to_bgr(jpeg)
        prev  = self._jpeg_to_bgr(prev_jpeg)
        if frame is None or prev is None:
            return False

        # Skip if lens is covered or scene is very dark (check both frames to catch transition)
        brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        prev_brightness = float(np.mean(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)))
        if brightness < 40 or prev_brightness < 40:
            return False

        h, w = frame.shape[:2]
        # Only look at upper half of frame (hands raised)
        diff = cv2.absdiff(frame[:h//2], prev[:h//2])
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        changed_pixels = np.count_nonzero(thresh)
        total_pixels   = (h // 2) * w
        return (changed_pixels / total_pixels) > 0.20   # >20% upper-half motion

    def check_face_close(self, jpeg: bytes) -> bool:
        """Face fills >25% of frame — child is up close."""
        frame = self._jpeg_to_bgr(jpeg)
        if frame is None:
            return False

        h, w = frame.shape[:2]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, 1.2, 4, minSize=(50, 50))
        if not len(faces):
            return False

        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_area_ratio = (fw * fh) / (w * h)
        return face_area_ratio > 0.10   # ~arm's length distance fills ~10-15% of QVGA frame

    def check_motion(self, jpeg: bytes, prev_jpeg: Optional[bytes]) -> bool:
        """Overall motion >8% of frame — used by freeze-dance / red-light referee."""
        if prev_jpeg is None:
            return False
        frame = self._jpeg_to_bgr(jpeg)
        prev  = self._jpeg_to_bgr(prev_jpeg)
        if frame is None or prev is None:
            return False

        diff  = cv2.absdiff(frame, prev)
        gray  = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        ratio = np.count_nonzero(th) / float(gray.size)
        return ratio > 0.08

    def annotate_passive(self, jpeg: bytes, reaction: str) -> bytes:
        """Draw a small reaction label in the corner of the frame for the GUI."""
        frame = self._jpeg_to_bgr(jpeg)
        if frame is None:
            return jpeg
        icons = {"peekaboo": "BOO!", "wave": "WAVE!", "face_close": "CLOSE!"}
        label = icons.get(reaction, reaction.upper())
        cv2.putText(frame, label, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return bytes(enc)


# ─────────────────────────────────────────────────────────────────────────────
# RobotVisionReceiver — TCP server on port 8891
# ─────────────────────────────────────────────────────────────────────────────

class RobotVisionReceiver:
    """Listens for JPEG frames from the robot, runs CV, sends commands back."""

    LISTEN_PORT = 8891

    PASSIVE_COOLDOWN = {
        "peekaboo":  15,
        "wave":      20,   # wave animation takes ~3s on Core 1; long cooldown prevents camera starvation
        "face_close": 12,
    }

    def __init__(self, robot, on_reaction: Optional[Callable] = None,
                 on_frame: Optional[Callable] = None):
        """
        robot       — SesameRobotController
        on_reaction — callback(reaction_name: str) for passive events
        on_frame    — callback(annotated_jpeg: bytes) for GUI display
        """
        self.robot       = robot
        self.on_reaction = on_reaction
        self.on_frame    = on_frame

        self._mode          = "passive"
        self._active_target = None   # (game, target) e.g. ("find", "red") | ("follow", None)
        self._last_reaction: dict[str, float] = {}
        self._prev_frame:    Optional[bytes]  = None
        self._active_timeout = int(os.environ.get("VISION_TIMEOUT", "30"))
        self._active_started   = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._last_frame_time: float = 0.0  # 0 = no frame ever received

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        t = threading.Thread(target=self._listen_loop, daemon=True)
        t.start()
        logger.info("[Vision] Receiver started on port %d", self.LISTEN_PORT)

    def stop(self):
        self._running = False

    def set_active(self, game: str, target: Optional[str] = None):
        with self._lock:
            self._mode           = "active"
            self._active_target  = (game, target)
            self._active_started = time.time()
        logger.info("[Vision] Active mode: %s target=%s", game, target)

    def set_passive(self):
        with self._lock:
            self._mode          = "passive"
            self._active_target = None
        logger.info("[Vision] Passive mode")

    @property
    def has_camera(self) -> bool:
        """True if a camera frame was received in the last 30 seconds."""
        return self._last_frame_time > 0 and (time.time() - self._last_frame_time) < 30.0

    def status(self) -> str:
        """Returns 'PASSIVE', 'ACTIVE — <game>', or 'No camera'."""
        if not self.has_camera:
            return "No camera"
        with self._lock:
            if self._mode == "active" and self._active_target:
                game, target = self._active_target
                label = f"{game} {target}" if target else game
                return f"ACTIVE — {label}"
            return "PASSIVE"

    # ── internal ─────────────────────────────────────────────────────────────

    def _listen_loop(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", self.LISTEN_PORT))
        srv.listen(1)
        srv.settimeout(2.0)
        logger.info("[Vision] Listening on :%d", self.LISTEN_PORT)

        while self._running:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except Exception as e:
                logger.error("[Vision] Accept error: %s", e)
                continue
            logger.info("[Vision] Robot connected from %s", addr)
            # Handle on a separate thread so accept() is always available for reconnects
            def _serve(c):
                try:
                    self._handle_connection(c)
                except Exception as e:
                    logger.error("[Vision] Connection error: %s", e)
                finally:
                    try:
                        c.close()
                    except Exception:
                        pass
                    logger.info("[Vision] Robot disconnected")
            threading.Thread(target=_serve, args=(conn,), daemon=True).start()
            logger.info("[Vision] Robot disconnected")

        srv.close()

    def _recv_exact(self, conn: socket.socket, n: int) -> Optional[bytes]:
        buf = b""
        conn.settimeout(15.0)  # robot may pause frames during servo animations (Core 1 shared)
        while len(buf) < n:
            try:
                chunk = conn.recv(n - len(buf))
            except (socket.timeout, OSError):
                return None
            if not chunk:
                return None
            buf += chunk
        return buf

    def _handle_connection(self, conn: socket.socket):
        proc = VisionProcessor()
        latest_frame: list = [None]   # [annotated_jpeg] — written by cv thread, read by on_frame
        latest_lock = threading.Lock()
        face_frame_counter = [0]       # run face detection every N frames (it's slow)

        def _fast_passive_check(jpeg):
            """Peekaboo + wave — runs inline on the receive thread (very fast, no face detection)."""
            now = time.time()
            reaction = None
            if proc.check_peekaboo(jpeg) and self._cooldown_ok("peekaboo", now):
                reaction = "peekaboo"
            elif proc.check_wave(jpeg, self._prev_frame) and self._cooldown_ok("wave", now):
                reaction = "wave"
            if reaction:
                self._last_reaction[reaction] = now
                if self.on_reaction:
                    threading.Thread(target=self.on_reaction, args=(reaction,), daemon=True).start()
                return proc.annotate_passive(jpeg, reaction)
            return jpeg

        def _slow_cv_worker(jpeg, mode, target):
            """Face detection + active CV — runs on a background thread (can be slow)."""
            if mode == "passive":
                now = time.time()
                if proc.check_face_close(jpeg) and self._cooldown_ok("face_close", now):
                    self._last_reaction["face_close"] = now
                    if self.on_reaction:
                        threading.Thread(target=self.on_reaction, args=("face_close",),
                                         daemon=True).start()
                    annotated = proc.annotate_passive(jpeg, "face_close")
                    with latest_lock:
                        latest_frame[0] = annotated
            else:
                _, annotated = self._active_check(proc, jpeg, target)
                with latest_lock:
                    latest_frame[0] = annotated

        slow_thread: list = [None]

        while True:
            # Read [uint32 len][JPEG]
            len_bytes = self._recv_exact(conn, 4)
            if not len_bytes:
                break
            jpeg_len = struct.unpack("<I", len_bytes)[0]
            if jpeg_len == 0 or jpeg_len > 200_000:
                break
            jpeg = self._recv_exact(conn, jpeg_len)
            if not jpeg:
                break

            self._last_frame_time = time.time()  # camera is alive

            with self._lock:
                mode   = self._mode
                target = self._active_target

            # Active timeout
            if mode == "active":
                elapsed = time.time() - self._active_started
                if elapsed > self._active_timeout:
                    logger.info("[Vision] Active timeout after %.0fs", elapsed)
                    self.set_passive()
                    mode   = "passive"
                    target = None
                    self.robot.send_command("stop")

            # Reply immediately — robot never waits on CV
            try:
                ack = b"ok"
                conn.sendall(struct.pack("<I", len(ack)) + ack)
            except Exception:
                break

            # Fast checks (peekaboo + wave) run inline — each takes <5ms
            annotated = _fast_passive_check(jpeg) if mode == "passive" else jpeg

            # Merge slow CV result if available
            with latest_lock:
                if latest_frame[0] is not None:
                    annotated = latest_frame[0]
                    latest_frame[0] = None

            if self.on_frame:
                try:
                    self.on_frame(annotated)
                except Exception:
                    pass

            self._prev_frame = jpeg

            # Slow CV (face detection / active tracking) — one thread at a time, every 5 frames
            face_frame_counter[0] += 1
            if face_frame_counter[0] % 5 == 0:
                if slow_thread[0] is None or not slow_thread[0].is_alive():
                    t = threading.Thread(target=_slow_cv_worker, args=(jpeg, mode, target),
                                         daemon=True)
                    t.start()
                    slow_thread[0] = t

    def _passive_check(self, proc: VisionProcessor, jpeg: bytes):
        now = time.time()
        reaction = None

        if proc.check_peekaboo(jpeg) and self._cooldown_ok("peekaboo", now):
            reaction = "peekaboo"
        elif proc.check_wave(jpeg, self._prev_frame) and self._cooldown_ok("wave", now):
            reaction = "wave"
        elif proc.check_face_close(jpeg) and self._cooldown_ok("face_close", now):
            reaction = "face_close"

        if reaction:
            self._last_reaction[reaction] = now
            if self.on_reaction:
                # Fire in background so the "ok" reply to the robot isn't delayed
                threading.Thread(target=self.on_reaction, args=(reaction,),
                                 daemon=True).start()
            annotated = proc.annotate_passive(jpeg, reaction)
        else:
            annotated = jpeg

        return "ok", annotated

    def _active_check(self, proc: VisionProcessor, jpeg: bytes, target):
        if target is None:
            return "ok", jpeg

        game, tgt = target

        if game in ("find", "chase"):
            det = proc.find_color_object(jpeg, tgt or "red")
            cmd = proc.detection_to_command(det, game)
            if det["found"] and cmd == "stop" and game == "find":
                # Found! — notify, then return to passive
                self.set_passive()
                if self.on_reaction:
                    try:
                        self.on_reaction("found")
                    except Exception:
                        pass
            return cmd, det.get("annotated", jpeg)

        if game == "follow":
            det = proc.find_person(jpeg)
            cmd = proc.detection_to_command(det, "follow")
            return cmd, det.get("annotated", jpeg)

        if game == "freeze_check":
            moving = proc.check_motion(jpeg, self._prev_frame)
            return ("caught" if moving else "ok"), jpeg

        if game == "fingers":
            n = proc.count_fingers(jpeg)
            return f"fingers:{n}", jpeg

        return "ok", jpeg

    def _cooldown_ok(self, reaction: str, now: float) -> bool:
        last = self._last_reaction.get(reaction, 0)
        return (now - last) >= self.PASSIVE_COOLDOWN.get(reaction, 5)


# ─────────────────────────────────────────────────────────────────────────────
# VisionCommandLayer — pre-LLM voice matching
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = r"(?P<color>red|blue|green|yellow|orange|pink|purple)"

_VISION_PATTERNS = [
    # (regex, game, target_group_or_literal, response_template)
    (rf"\bfollow\b.*\bme\b",                           "follow",        None,      "Okay, I'll follow you!"),
    (rf"\bhide.*seek\b|\bseek.*hide\b",                "hide_seek",     None,      "Ready or not, here I come!"),
    (rf"\bfind\b.*{_COLORS}",                          "find",          "color",   "Looking for the {color} one!"),
    (rf"\bfind\b.*(ball|lego|toy|block)",               "find",          "red",     "Looking for it!"),
    (rf"\bchase\b.*{_COLORS}",                         "chase",         "color",   "Chasing the {color} one!"),
    (rf"\bchase\b.*(ball|lego|toy)",                   "chase",         "red",     "Chasing it!"),
    (rf"\bred light.*green light\b|\bgreen light\b",   "red_light",     None,      "I'm the stoplight — ready!"),
    (rf"\bfreeze dance\b",                             "freeze_dance",  None,      "Dance party! Go!"),
    (rf"\bbring me.*{_COLORS}",                        "bring",         "color",   "Show me something {color}!"),
    (rf"\bhow many fingers\b",                         "fingers",       None,      "Hold up your fingers!"),
    (rf"\bshow.*tell\b|\bwhat.*see\b|\bwhat.*that\b",  "show_tell",     None,      "Let me look!"),
    (rf"\bcolou?r.*teach|\bteach.*colou?r",            "color_teacher", None,      "Let's learn colours!"),
    (rf"\bmake me laugh\b|\bsilly face\b",             "make_laugh",    None,      "Show me your funniest face!"),
    (rf"\bstaring contest\b",                          "stare",         None,      "Don't blink! Go!"),
    (rf"\bvision\s+stop\b|\bstop\s+camera\b|\bstop\s+vision\b", "stop", None,     "Okay, closing my eyes."),
]

_COMPILED = [(re.compile(pat, re.IGNORECASE), game, tgt_key, resp)
             for pat, game, tgt_key, resp in _VISION_PATTERNS]


class VisionCommandLayer:
    """Match voice input to vision game modes before hitting the LLM."""

    def match(self, text: str) -> Optional[dict]:
        """
        Returns dict with keys: game, target, response, command
        or None if no vision match.
        """
        for pattern, game, tgt_key, resp_template in _COMPILED:
            m = pattern.search(text)
            if not m:
                continue

            # Resolve target
            if tgt_key == "color":
                target = m.group("color")
            elif tgt_key is not None:
                target = tgt_key
            else:
                target = None

            response = resp_template.format(color=target) if target and "{color}" in resp_template else resp_template

            return {
                "game":     game,
                "target":   target,
                "response": response,
                "command":  "vision",
                "face":     "excited",
            }
        return None
