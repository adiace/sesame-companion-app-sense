#!/usr/bin/env python3
"""
Sesame Robot Companion App GUI
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
import queue
from datetime import datetime
from sesame_companion import (SesameCompanionApp, SesameRobotController,
                              AVAILABLE_COMMANDS, AVAILABLE_FACES)

try:
    from PIL import Image, ImageTk
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import cv2
    _CV2_GUI_AVAILABLE = True
except ImportError:
    _CV2_GUI_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[WARNING] python-dotenv not installed. .env file will be ignored.")

class SesameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sesame Robot Companion")
        self.root.geometry("1080x780")
        self.root.minsize(900, 680)
        
        self.robot_ip = os.getenv("SESAME_ROBOT_IP", "192.168.1.1")
        self.sesame_local = os.getenv("SESAME_LOCAL", "false").lower() == "true"
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.tts_engine = tk.StringVar(value=os.getenv("TTS_ENGINE", "pyttsx3"))
        self.voice_enabled = tk.BooleanVar(value=True)
        self.wake_word_mode = tk.BooleanVar(value=os.getenv("WAKE_WORD_MODE", "false").lower() == "true")
        self.wake_word = os.getenv("WAKE_WORD", "hey sesame")
        
        self.is_listening = False
        self.is_speaking = False
        self.app: SesameCompanionApp = None   # set by init_app thread; guard all access
        self.message_queue = queue.Queue()

        # Camera panel state
        self._latest_frame: bytes = b""
        self._frame_lock = threading.Lock()
        self._cam_photo = None   # keep reference so Tkinter doesn't GC it
        self._pending_pil = None     # PIL Image decoded on bg thread, converted on main thread
        
        # Theme
        self.bg_color = "#1e1e1e"
        self.secondary_bg = "#2d2d2d"
        self.accent_color = "#ff8c42"
        self.text_color = "#e0e0e0"
        self.success_color = "#4caf50"
        self.error_color = "#f44336"
        
        self.btn_bg      = "#444444"
        self.btn_hover   = "#ff8c42"
        self.btn_fg      = "#ffffff"

        self.setup_ui()
        self.start_backend()
        self.process_queue()

    def _make_btn(self, parent, text, command, bg=None, fg=None,
                  font=("Segoe UI", 11, "bold"), padx=12, pady=8, fill=False):
        """macOS-safe button: tk.Label with click + hover bindings."""
        bg  = bg  or self.btn_bg
        fg  = fg  or self.btn_fg
        frm = tk.Frame(parent, bg=bg, cursor="hand2")
        lbl = tk.Label(frm, text=text, bg=bg, fg=fg, font=font,
                       padx=padx, pady=pady, cursor="hand2")
        lbl.pack(fill=tk.BOTH, expand=True)
        hover = self.btn_hover
        def _enter(e): lbl.config(bg=hover); frm.config(bg=hover)
        def _leave(e): lbl.config(bg=bg);    frm.config(bg=bg)
        def _click(e): command()
        for w in (frm, lbl):
            w.bind("<Enter>",    _enter)
            w.bind("<Leave>",    _leave)
            w.bind("<Button-1>", _click)
        return frm
        
    def setup_ui(self):
        """Setup the user interface"""
        self.root.configure(bg=self.bg_color)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background=self.bg_color)
        style.configure("TLabel", background=self.bg_color, foreground=self.text_color, font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 12, "bold"))
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.create_status_bar(main_frame)
        
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Left column — quick action buttons
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.create_quick_actions(left_frame)

        # Right column — camera on top, status below. Width fixed by camera canvas (320px).
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.create_camera_panel(right_frame)
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        self.create_status_panel(right_frame)

        # Center column — chat (expands to fill remaining space)
        center_frame = ttk.Frame(content_frame)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.create_chat_area(center_frame)
        
    def create_status_bar(self, parent):
        """Create the top title bar."""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(status_frame, text="SESAME DESKTOP INTERFACE",
                  font=("Segoe UI", 20, "bold")).pack(side=tk.LEFT)
        # robot_status_label kept as invisible stub — referenced by process_queue
        self.robot_status_label = ttk.Label(status_frame, text="")
        self.connection_label   = ttk.Label(status_frame, text="")   # lives in right panel now
        
    def create_quick_actions(self, parent):
        """D-pad movement, stop, poses, motor control, refresh."""

        # ── D-pad ─────────────────────────────────────────────────────────────
        ttk.Label(parent, text="Movement", font=("Segoe UI", 11, "bold")).pack(pady=(0, 6))

        dpad = tk.Frame(parent, bg=self.bg_color)
        dpad.pack()

        def _dpad_btn(text, row, col, direction):
            frm = tk.Frame(dpad, bg=self.btn_bg, cursor="hand2")
            frm.grid(row=row, column=col, padx=3, pady=3)
            lbl = tk.Label(frm, text=text, bg=self.btn_bg, fg=self.btn_fg,
                           font=("Segoe UI", 16), width=3, pady=6, cursor="hand2")
            lbl.pack()
            def _press(e):  self._dpad_press(direction)
            def _release(e): self._dpad_release()
            def _enter(e):  lbl.config(bg=self.btn_hover); frm.config(bg=self.btn_hover)
            def _leave(e):  lbl.config(bg=self.btn_bg);   frm.config(bg=self.btn_bg)
            for w in (frm, lbl):
                w.bind("<ButtonPress-1>",   _press)
                w.bind("<ButtonRelease-1>", _release)
                w.bind("<Enter>", _enter)
                w.bind("<Leave>", _leave)

        _dpad_btn("▲", 0, 1, "forward")
        _dpad_btn("◄", 1, 0, "left")
        _dpad_btn("▼", 1, 1, "backward")
        _dpad_btn("►", 1, 2, "right")

        stop_btn = self._make_btn(parent, "■  STOP", lambda: self.send_quick_command("stop"),
                                  bg="#e63946", font=("Segoe UI", 10, "bold"), pady=6)
        stop_btn.pack(fill=tk.X, pady=(8, 0))

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # ── Poses ─────────────────────────────────────────────────────────────
        ttk.Label(parent, text="Poses", font=("Segoe UI", 11, "bold")).pack(pady=(0, 6))

        poses = [
            ("Wave",  "wave"),  ("Dance", "dance"), ("Swim",   "swim"),
            ("Stand", "stand"), ("Rest",  "rest"),  ("Pushup", "pushup"),
            ("Bow",   "bow"),   ("Cute",  "cute"),  ("Crab",   "crab"),
            ("Box",   "box"),   ("Shrug", "shrug"), ("Worm",   "worm"),
        ]
        pose_grid = tk.Frame(parent, bg=self.bg_color)
        pose_grid.pack(fill=tk.X)
        for idx, (text, cmd) in enumerate(poses):
            row, col = divmod(idx, 2)
            btn = self._make_btn(pose_grid, text,
                                 lambda c=cmd: self.send_quick_command(c),
                                 font=("Segoe UI", 9, "bold"), padx=6, pady=5)
            btn.grid(row=row, column=col, padx=2, pady=2, sticky=tk.EW)
        pose_grid.columnconfigure(0, weight=1)
        pose_grid.columnconfigure(1, weight=1)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # ── Motor control + refresh ───────────────────────────────────────────
        motor_btn = self._make_btn(parent, "Motor Control", self.open_motor_control,
                                   font=("Segoe UI", 9, "bold"), pady=5)
        motor_btn.pack(fill=tk.X, pady=(0, 4))

        refresh_btn = self._make_btn(parent, "Refresh Status", self.refresh_status,
                                     font=("Segoe UI", 9, "bold"), pady=5)
        refresh_btn.pack(fill=tk.X)
        
    def create_chat_area(self, parent):
        """Create the main chat interface"""
        chat_frame = ttk.Frame(parent)
        chat_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(chat_frame, text="Conversation", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg=self.secondary_bg,
            fg=self.text_color,
            insertbackground=self.text_color,
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)
        
        self.chat_display.tag_config("user", foreground="#64b5f6", font=("Consolas", 10, "bold"))
        self.chat_display.tag_config("sesame", foreground=self.accent_color, font=("Consolas", 10, "bold"))
        self.chat_display.tag_config("system", foreground="#aaaaaa", font=("Consolas", 9, "italic"))
        self.chat_display.tag_config("error", foreground=self.error_color)
        self.chat_display.tag_config("success", foreground=self.success_color)
        
        input_frame = ttk.Frame(parent)
        input_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.input_entry = tk.Entry(
            input_frame,
            font=("Segoe UI", 11),
            bg=self.secondary_bg,
            fg=self.text_color,
            insertbackground=self.text_color,
            relief=tk.FLAT
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 5))
        self.input_entry.bind("<Return>", lambda e: self.send_message())
        
        self.mic_button = self._make_btn(input_frame, "MIC", self.toggle_listening,
                                         bg=self.accent_color, font=("Segoe UI", 10, "bold"),
                                         padx=8, pady=4)
        self.mic_button._label = self.mic_button.winfo_children()[0]
        self.mic_button.pack(side=tk.LEFT, padx=(0, 5))
        
        send_button = self._make_btn(input_frame, "Send", self.send_message,
                                     bg=self.accent_color, font=("Segoe UI", 10, "bold"),
                                     padx=15, pady=4)
        send_button.pack(side=tk.LEFT)
        
    def create_camera_panel(self, parent):
        """Live camera feed panel with passive/active status badge."""
        cam_outer = tk.Frame(parent, bg=self.bg_color)
        cam_outer.pack(fill=tk.X, pady=(0, 10))

        # Status badge row
        badge_row = tk.Frame(cam_outer, bg=self.bg_color)
        badge_row.pack(fill=tk.X)

        self._cam_dot = tk.Label(badge_row, text="●", font=("Segoe UI", 10),
                                 fg="#888888", bg=self.bg_color)
        self._cam_dot.pack(side=tk.LEFT)

        self._cam_status = tk.Label(badge_row, text="No camera",
                                    font=("Segoe UI", 9), fg="#888888",
                                    bg=self.bg_color)
        self._cam_status.pack(side=tk.LEFT, padx=(4, 0))

        # Video canvas — 320×240 to match robot QVGA
        self._cam_canvas = tk.Canvas(cam_outer, width=320, height=240,
                                     bg="#111111", highlightthickness=1,
                                     highlightbackground="#444444")
        self._cam_canvas.pack(pady=(4, 0))

        # Placeholder text when no frame yet
        self._cam_canvas.create_text(160, 120, text="Camera feed\n(say 'find the red lego')",
                                     fill="#555555", font=("Segoe UI", 9),
                                     justify=tk.CENTER, tags="placeholder")

        # Start refresh loop (~3fps; non-blocking via after())
        self.root.after(333, self._refresh_camera_frame)

    def _refresh_camera_frame(self):
        """Convert pre-decoded PIL image (from bg thread) to PhotoImage and display."""
        pil_img = self._pending_pil
        if pil_img is not None:
            self._pending_pil = None
            try:
                self._cam_photo = ImageTk.PhotoImage(pil_img)
                self._cam_canvas.delete("placeholder")
                self._cam_canvas.create_image(0, 0, anchor=tk.NW, image=self._cam_photo)
            except Exception:
                pass

        # Update status badge
        if self.app and getattr(self.app, "vision", None):
            status = self.app.vision.status()
            dot_color = self.accent_color if status.startswith("ACTIVE") else self.success_color
            self._cam_dot.config(fg=dot_color)
            self._cam_status.config(text=status, fg=dot_color)
        else:
            self._cam_dot.config(fg="#888888")
            self._cam_status.config(text="No camera", fg="#888888")

        self.root.after(100, self._refresh_camera_frame)  # ~10fps check

    def _on_vision_frame(self, jpeg: bytes):
        """Called from RobotVisionReceiver thread — decode JPEG, store PIL for main thread."""
        if not (_PIL_AVAILABLE and _CV2_GUI_AVAILABLE):
            return
        try:
            import cv2, numpy as np
            arr = np.frombuffer(jpeg, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._pending_pil = Image.fromarray(rgb).resize((320, 240), Image.NEAREST)
        except Exception:
            pass

    def create_status_panel(self, parent):
        """Consolidated status — connection, robot state, mic toggle."""
        ttk.Label(parent, text="Status", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, pady=(0, 6))

        grid = ttk.Frame(parent)
        grid.pack(anchor=tk.W)

        def _row(label, row, initial="—"):
            ttk.Label(grid, text=label, font=("Segoe UI", 9),
                      foreground="#888888").grid(row=row, column=0, sticky=tk.W, pady=2)
            val = ttk.Label(grid, text=initial, font=("Segoe UI", 9, "bold"))
            val.grid(row=row, column=1, sticky=tk.W, pady=2, padx=(8, 0))
            return val

        # Connection row has a coloured dot + text in one label
        ttk.Label(grid, text="Connection", font=("Segoe UI", 9),
                  foreground="#888888").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.connection_label = ttk.Label(grid, text="● Disconnected",
                                          font=("Segoe UI", 9, "bold"),
                                          foreground=self.error_color)
        self.connection_label.grid(row=0, column=1, sticky=tk.W, pady=2, padx=(8, 0))

        self.ip_label      = _row("IP",      1, self.robot_ip)
        self.face_label    = _row("Face",    2, "—")
        self.command_label = _row("Command", 3, "—")

        
    def start_backend(self):
        """Initialize the backend app in a separate thread"""
        def init_app():
            try:
                self.app = SesameCompanionApp(
                    self.robot_ip,
                    self.sesame_local,
                    self.gemini_api_key,
                    self.voice_enabled.get(),
                    self.tts_engine.get(),
                    self.wake_word,
                    self.wake_word_mode.get()
                )

                # Wire robot voice receiver → GUI chat
                def _on_robot_voice(user_text, result):
                    self.message_queue.put(("robot_voice_user", user_text))
                    response = (result.get("response") or "").strip()
                    command  = result.get("command")
                    if not response and command:
                        response = f"*{command}*"
                    if response:
                        self.message_queue.put(("robot_voice_sesame", response))

                self.app.robot_voice.on_interaction = _on_robot_voice
                self.app.start_robot_voice_receiver()

                # Wire camera frame callback → GUI display
                if getattr(self.app, "vision", None):
                    self.app.vision.on_frame = self._on_vision_frame

                # Wire passive reaction → conversation window
                _labels = {"wave": "Wave detected", "peekaboo": "Peek-a-boo!",
                           "face_close": "Face close-up", "found": "Found it!"}
                def _on_reaction(r):
                    self.message_queue.put(("robot", f"[Camera] {_labels.get(r, r)}"))
                self.app.on_passive_reaction = _on_reaction

                self.message_queue.put(("system", "[OK] Backend initialized"))
                self.message_queue.put(("system", "[OK] Robot voice receiver active (port 8889)"))
                if getattr(self.app, "vision", None):
                    self.message_queue.put(("system", "[OK] Vision receiver active (port 8891)"))
                self.check_connection()
            except Exception as e:
                self.message_queue.put(("error", f"Failed to initialize: {e}"))

        thread = threading.Thread(target=init_app, daemon=True)
        thread.start()
        
    def check_connection(self):
        """Check robot connection status"""
        if not self.app:
            return
            
        def check():
            try:
                status = self.app.robot.get_status()
                if "error" in status:
                    self.message_queue.put(("connection", False))
                else:
                    self.message_queue.put(("connection", True))
                    self.message_queue.put(("status", status))
            except Exception as e:
                self.message_queue.put(("connection", False))
        
        thread = threading.Thread(target=check, daemon=True)
        thread.start()
        
    def send_message(self):
        """Send a text message. Prefix with / to bypass LLM and send raw to robot."""
        message = self.input_entry.get().strip()
        if not message:
            return
        if not self.app:
            self.add_message("error", "Not ready yet — backend still initializing")
            return

        self.input_entry.delete(0, tk.END)

        if message.startswith("/"):
            raw_cmd = message[1:].strip()
            self.add_message("system", f"→ {raw_cmd}")
            def send_raw():
                try:
                    self.app.robot._tcp_send(raw_cmd)
                    self.message_queue.put(("success", f"[OK] sent: {raw_cmd}"))
                except Exception as e:
                    self.message_queue.put(("error", f"Send failed: {e}"))
            threading.Thread(target=send_raw, daemon=True).start()
            return

        self.add_message("user", message)

        def process():
            try:
                response, interpretation = self.app.process_input(message)
                self.message_queue.put(("sesame", interpretation.get("response", response)))

                if self.voice_enabled.get() and "response" in interpretation:
                    face = interpretation.get("face") if not interpretation.get("command") else None
                    self.app.voice.speak(
                        interpretation["response"],
                        async_mode=True,
                        face=face,
                        robot_controller=self.app.robot
                    )
            except Exception as e:
                self.message_queue.put(("error", f"Error: {e}"))

        thread = threading.Thread(target=process, daemon=True)
        thread.start()
        
    def toggle_listening(self):
        """Toggle voice listening. Click once to start, click again to cancel."""
        if not self.voice_enabled.get():
            self.add_message("system", "Voice mode is disabled. Enable it in settings.")
            return

        if self.wake_word_mode.get():
            self.add_message("system", f"Wake word mode is active. Say '{self.wake_word}' to trigger listening.")
            return

        if self.is_listening:
            self._listen_cancel.set()
            return

        if not self.app:
            self.add_message("error", "Not ready yet — backend still initializing")
            return

        self._listen_cancel = threading.Event()
        self.is_listening = True
        self.mic_button.config(bg=self.error_color)
        self.mic_button._label.config(bg=self.error_color, text="STOP")
        self.add_message("system", "Listening... (speak now, or click STOP)")

        def listen():
            try:
                text = self.app.voice.listen(cancel_event=self._listen_cancel)
                if self._listen_cancel.is_set():
                    self.message_queue.put(("system", "Listening cancelled"))
                elif text:
                    self.message_queue.put(("voice_input", text))
                else:
                    self.message_queue.put(("system", "No speech detected"))
            except Exception as e:
                self.message_queue.put(("error", f"Voice error: {e}"))
            finally:
                self.message_queue.put(("listening_done", None))

        thread = threading.Thread(target=listen, daemon=True)
        thread.start()
        
    def _dpad_press(self, direction):
        if not self.app:
            return
        def send():
            try:
                self.app.robot._tcp_send(direction)
            except Exception:
                pass
        threading.Thread(target=send, daemon=True).start()

    def _dpad_release(self):
        if not self.app:
            return
        def send():
            try:
                self.app.robot._tcp_send("stop")
            except Exception:
                pass
        threading.Thread(target=send, daemon=True).start()

    def open_motor_control(self):
        """Open a Toplevel with 8 servo sliders — mirrors captive portal motor panel."""
        if not self.app:
            self.add_message("error", "Not ready yet — backend still initializing")
            return

        win = tk.Toplevel(self.root)
        win.title("Motor Control")
        win.configure(bg=self.bg_color)
        win.geometry("360x480")
        win.resizable(False, False)

        ttk.Label(win, text="Manual Motor Control",
                  font=("Segoe UI", 12, "bold")).pack(pady=(14, 10))

        SERVO_NAMES = [
            "S0  R1 (right front hip)",
            "S1  R2 (right rear hip)",
            "S2  L1 (left front hip)",
            "S3  L2 (left rear hip)",
            "S4  R4 (right front knee)",
            "S5  R3 (right rear knee)",
            "S6  L3 (left front knee)",
            "S7  L4 (left rear knee)",
        ]

        frame = tk.Frame(win, bg=self.bg_color)
        frame.pack(fill=tk.BOTH, expand=True, padx=16)

        for i, name in enumerate(SERVO_NAMES):
            row = tk.Frame(frame, bg=self.bg_color)
            row.pack(fill=tk.X, pady=3)

            tk.Label(row, text=name, font=("Consolas", 8), bg=self.bg_color,
                     fg="#aaaaaa", anchor=tk.W).pack(side=tk.TOP, anchor=tk.W)

            ctrl = tk.Frame(row, bg=self.bg_color)
            ctrl.pack(fill=tk.X)

            val_lbl = tk.Label(ctrl, text="90°", font=("Segoe UI", 9, "bold"),
                               bg=self.bg_color, fg=self.accent_color, width=4)
            val_lbl.pack(side=tk.RIGHT)

            def _on_change(v, idx=i, lbl=val_lbl):
                angle = int(float(v))
                lbl.config(text=f"{angle}°")
                if self.app:
                    def send(a=angle, s=idx):
                        try:
                            self.app.robot._tcp_send(f"servo {s} {a}")
                        except Exception:
                            pass
                    threading.Thread(target=send, daemon=True).start()

            slider = tk.Scale(ctrl, from_=0, to=180, orient=tk.HORIZONTAL,
                              bg=self.secondary_bg, fg=self.text_color,
                              highlightthickness=0, troughcolor="#333333",
                              activebackground=self.accent_color,
                              showvalue=False, command=_on_change)
            slider.set(90)
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        tk.Frame(win, bg="#e63946", cursor="hand2").pack(fill=tk.X, padx=16, pady=12)
        close = self._make_btn(win, "Close", win.destroy,
                               bg="#e63946", font=("Segoe UI", 10, "bold"), pady=7)
        close.pack(fill=tk.X, padx=16, pady=(0, 14))

    def send_quick_command(self, command):
        """Send a quick command"""
        if not self.app:
            self.add_message("error", "Not ready yet — backend still initializing")
            return
        self.add_message("system", f"Executing: {command}")

        def execute():
            try:
                result = self.app.robot.send_command(command)
                if "error" in result:
                    self.message_queue.put(("error", f"Command failed: {result['error']}"))
                else:
                    self.message_queue.put(("success", f"[OK] {command} executed"))
                    self.refresh_status()
            except Exception as e:
                self.message_queue.put(("error", f"Error: {e}"))
        
        thread = threading.Thread(target=execute, daemon=True)
        thread.start()
        
    def refresh_status(self):
        """Refresh robot status"""
        self.check_connection()
        
    def toggle_voice_mode(self):
        """Toggle voice mode"""
        if self.app:
            self.app.voice_mode = self.voice_enabled.get()
            self.app.voice.voice_enabled = self.voice_enabled.get()
        status = "enabled" if self.voice_enabled.get() else "disabled"
        self.add_message("system", f"Voice mode {status}")
    
    def toggle_wake_word_mode(self):
        """Toggle wake word mode"""
        if self.app:
            self.app.wake_word_mode = self.wake_word_mode.get()
        status = "enabled" if self.wake_word_mode.get() else "disabled"
        self.add_message("system", f"Wake word mode {status}")
        
        if self.wake_word_mode.get():
            self.add_message("system", f"Say '{self.wake_word}' to activate listening")
            self.start_wake_word_listener()
        
    def change_tts_engine(self):
        """Change TTS engine"""
        if self.app:
            self.app.voice.tts_engine_type = self.tts_engine.get()
            self.app.tts_engine = self.tts_engine.get()
        self.add_message("system", f"TTS engine: {self.tts_engine.get()}")
    
    def start_wake_word_listener(self):
        """Start continuous wake word listening"""
        def listen_loop():
            while self.wake_word_mode.get() and self.voice_enabled.get():
                try:
                    if self.app.voice.listen_for_wake_word(timeout=10):
                        self.message_queue.put(("system", "Wake word detected! Listening for command..."))
                        text = self.app.voice.listen(timeout=10)
                        if text:
                            self.message_queue.put(("voice_input", text))
                        else:
                            self.message_queue.put(("system", "No command received"))
                except Exception as e:
                    self.message_queue.put(("error", f"Wake word error: {e}"))
                    break
        
        if self.wake_word_mode.get():
            thread = threading.Thread(target=listen_loop, daemon=True)
            thread.start()
        
    def add_message(self, sender, message):
        """Add a message to the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if sender == "user":
            self.chat_display.insert(tk.END, f"[{timestamp}] ", "system")
            self.chat_display.insert(tk.END, "You: ", "user")
            self.chat_display.insert(tk.END, f"{message}\n")
        elif sender == "sesame":
            self.chat_display.insert(tk.END, f"[{timestamp}] ", "system")
            self.chat_display.insert(tk.END, "Sesame: ", "sesame")
            self.chat_display.insert(tk.END, f"{message}\n")
        elif sender == "system":
            self.chat_display.insert(tk.END, f"[{timestamp}] ", "system")
            self.chat_display.insert(tk.END, f"{message}\n", "system")
        elif sender == "error":
            self.chat_display.insert(tk.END, f"[{timestamp}] ", "system")
            self.chat_display.insert(tk.END, f"[ERROR] {message}\n", "error")
        elif sender == "success":
            self.chat_display.insert(tk.END, f"[{timestamp}] ", "system")
            self.chat_display.insert(tk.END, f"{message}\n", "success")
        
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
    def process_queue(self):
        """Process messages from background threads"""
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()
                
                if msg_type == "robot_voice_user":
                    self.add_message("user", f"[ROBOT MIC] {data}")
                elif msg_type == "robot_voice_sesame":
                    self.add_message("sesame", data)
                elif msg_type in ["user", "sesame", "system", "error", "success"]:
                    self.add_message(msg_type, data)
                elif msg_type == "connection":
                    if data:
                        self.connection_label.config(text="● Connected",
                                                     foreground=self.success_color)
                    else:
                        self.connection_label.config(text="● Disconnected",
                                                     foreground=self.error_color)
                elif msg_type == "status":
                    self.face_label.config(text=data.get("currentFace", "—") or "—")
                    self.command_label.config(text=data.get("currentCommand", "—") or "—")
                    self.robot_status_label.config(text="")   # stub, no longer displayed
                elif msg_type == "voice_input":
                    self.add_message("user", f"[VOICE] {data}")
                    self.input_entry.insert(0, data)
                    self.send_message()
                elif msg_type == "listening_done":
                    self.is_listening = False
                    self.mic_button.config(bg=self.accent_color)
                    self.mic_button._label.config(bg=self.accent_color, text="MIC")
                    
        except queue.Empty:
            pass
        
        self.root.after(100, self.process_queue)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = SesameGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
