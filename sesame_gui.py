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
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[WARNING] python-dotenv not installed. .env file will be ignored.")

class SesameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sesame Robot Companion")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
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
        
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.create_quick_actions(left_frame)
        
        center_frame = ttk.Frame(content_frame)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.create_chat_area(center_frame)
        
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.create_settings_panel(right_frame)
        
    def create_status_bar(self, parent):
        """Create the top status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(status_frame, text="SESAME DESKTOP INTERFACE", font=("Segoe UI", 20, "bold"))
        title_label.pack(side=tk.LEFT)
        
        self.connection_label = ttk.Label(status_frame, text="[*] Disconnected", font=("Segoe UI", 10))
        self.connection_label.pack(side=tk.RIGHT, padx=10)
        
        self.robot_status_label = ttk.Label(status_frame, text="Status: Idle", font=("Segoe UI", 10))
        self.robot_status_label.pack(side=tk.RIGHT, padx=10)
        
    def create_quick_actions(self, parent):
        """Create quick action buttons panel"""
        label = ttk.Label(parent, text="Quick Actions", font=("Segoe UI", 12, "bold"))
        label.pack(pady=(0, 10))
        
        commands = [
            ("Wave", "wave"),
            ("Dance", "dance"),
            ("Walk", "walk"),
            ("Stand", "stand"),
            ("Rest", "rest"),
            ("Pushup", "pushup"),
            ("Crab Walk", "crab"),
            ("Box", "box"),
            ("Swim", "swim"),
            ("Bow", "bow"),
            ("Stop", "stop"),
        ]
        
        for text, command in commands:
            btn = self._make_btn(parent, text, lambda cmd=command: self.send_quick_command(cmd))
            btn.pack(fill=tk.X, pady=2)
            
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
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
        
    def create_settings_panel(self, parent):
        """Create settings and info panel"""
        ttk.Label(parent, text="Settings", font=("Segoe UI", 12, "bold")).pack(pady=(0, 10))
        
        voice_check = tk.Checkbutton(
            parent,
            text="Voice Mode",
            variable=self.voice_enabled,
            command=self.toggle_voice_mode,
            bg=self.bg_color,
            fg=self.text_color,
            selectcolor=self.secondary_bg,
            activebackground=self.bg_color,
            activeforeground=self.text_color,
            font=("Segoe UI", 10)
        )
        voice_check.pack(anchor=tk.W, pady=5)
        
        wake_check = tk.Checkbutton(
            parent,
            text=f"Wake Word Mode ({self.wake_word})",
            variable=self.wake_word_mode,
            command=self.toggle_wake_word_mode,
            bg=self.bg_color,
            fg=self.text_color,
            selectcolor=self.secondary_bg,
            activebackground=self.bg_color,
            activeforeground=self.text_color,
            font=("Segoe UI", 10)
        )
        wake_check.pack(anchor=tk.W, pady=5)
        
        ttk.Label(parent, text="TTS Engine:", font=("Segoe UI", 9)).pack(anchor=tk.W, pady=(10, 2))
        
        tts_frame = ttk.Frame(parent)
        tts_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(
            tts_frame,
            text="pyttsx3 (Fast)",
            variable=self.tts_engine,
            value="pyttsx3",
            command=self.change_tts_engine
        ).pack(anchor=tk.W)
        
        ttk.Radiobutton(
            tts_frame,
            text="Gemini (Natural)",
            variable=self.tts_engine,
            value="gemini",
            command=self.change_tts_engine
        ).pack(anchor=tk.W)
        
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        
        ttk.Label(parent, text="Robot Info", font=("Segoe UI", 12, "bold")).pack(pady=(0, 10))
        
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill=tk.X)
        
        ttk.Label(info_frame, text="IP:", font=("Segoe UI", 9)).grid(row=0, column=0, sticky=tk.W, pady=2)
        self.ip_label = ttk.Label(info_frame, text=self.robot_ip, font=("Segoe UI", 9, "bold"))
        self.ip_label.grid(row=0, column=1, sticky=tk.W, pady=2, padx=5)
        
        ttk.Label(info_frame, text="Face:", font=("Segoe UI", 9)).grid(row=1, column=0, sticky=tk.W, pady=2)
        self.face_label = ttk.Label(info_frame, text="unknown", font=("Segoe UI", 9, "bold"))
        self.face_label.grid(row=1, column=1, sticky=tk.W, pady=2, padx=5)
        
        ttk.Label(info_frame, text="Command:", font=("Segoe UI", 9)).grid(row=2, column=0, sticky=tk.W, pady=2)
        self.command_label = ttk.Label(info_frame, text="none", font=("Segoe UI", 9, "bold"))
        self.command_label.grid(row=2, column=1, sticky=tk.W, pady=2, padx=5)
        
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        
        ttk.Label(parent, text="Available Commands", font=("Segoe UI", 10, "bold")).pack(pady=(0, 5))
        
        commands_text = scrolledtext.ScrolledText(
            parent,
            height=8,
            wrap=tk.WORD,
            font=("Consolas", 8),
            bg=self.secondary_bg,
            fg=self.text_color,
            relief=tk.FLAT
        )
        commands_text.pack(fill=tk.BOTH)
        commands_text.insert(tk.END, ", ".join(AVAILABLE_COMMANDS))
        commands_text.config(state=tk.DISABLED)
        
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

                self.message_queue.put(("system", "[OK] Backend initialized"))
                self.message_queue.put(("system", "[OK] Robot voice receiver active (port 8889)"))
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
                        self.connection_label.config(text="[+] Connected", foreground=self.success_color)
                    else:
                        self.connection_label.config(text="[-] Disconnected", foreground=self.error_color)
                elif msg_type == "status":
                    self.face_label.config(text=data.get("currentFace", "unknown"))
                    self.command_label.config(text=data.get("currentCommand", "none") or "none")
                    self.robot_status_label.config(text=f"Status: {data.get('currentCommand', 'Idle')}")
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
