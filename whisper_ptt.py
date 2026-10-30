"""Whisper Push-to-Talk: hold a key, speak, let go, and your words are typed where the cursor is.

Speech is turned into text by Whisper (faster-whisper) on this PC's GPU, so audio never
leaves the machine and it works offline. Settings live in config.json next to this file.

    python whisper_ptt.py              run it (close the console window to quit)
    python whisper_ptt.py --test a.wav transcribe a file and print the text, then exit
"""
import ctypes
import json
import queue
import re
import sys
import threading
import time
import tkinter as tk
import wave
import winsound
from pathlib import Path

import numpy as np
import sounddevice as sd
import win32clipboard
import win32con

HERE = Path(__file__).resolve().parent
user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

RATE = 16000  # Whisper wants 16 kHz mono
KEYS = {
    "right_ctrl": (0xA3, "Right Ctrl"),
    "right_alt": (0xA5, "Right Alt"),
    "right_shift": (0xA1, "Right Shift"),
    "f8": (0x77, "F8"),
    "f9": (0x78, "F9"),
    "f10": (0x79, "F10"),
    "scroll_lock": (0x91, "Scroll Lock"),
    "pause": (0x13, "Pause"),
    "insert": (0x2D, "Insert"),
}
# Shift / Ctrl / Alt / Win never cancel a recording; any other key does (it was a shortcut).
MODIFIERS = {0x10, 0x11, 0x12, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0x5B, 0x5C}
# Whisper sometimes "hears" these in near-silence.
PHANTOMS = {"thank you", "thanks for watching", "thank you for watching", "you", "bye", "so"}
DEFAULTS = {
    "hotkey": "right_ctrl",
    "language": "en",
    "model": "large-v3",
    "device": "cuda",
    "compute_type": "int8_float16",
    "vocabulary": "",
    "replacements": {},
    "mic": None,
    "beeps": True,
    "restore_clipboard": True,
    "min_seconds": 0.35,
}


def load_config():
    cfg = dict(DEFAULTS)
    path = HERE / "config.json"
    if path.exists():
        cfg.update(json.loads(path.read_text(encoding="utf-8")))
    if cfg["hotkey"] not in KEYS:
        sys.exit(f"Unknown hotkey '{cfg['hotkey']}' in config.json. Use one of: {', '.join(KEYS)}")
    return cfg


# ---------------------------------------------------------------- keyboard + clipboard

def key_down(vk):
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def other_key_down(hotkey_vk):
    return any(key_down(vk) for vk in range(0x08, 0xFF) if vk != hotkey_vk and vk not in MODIFIERS)


def _open_clipboard():
    for _ in range(25):
        try:
            win32clipboard.OpenClipboard()
            return True
        except Exception:
            time.sleep(0.02)
    return False


def get_clipboard_text():
    if not _open_clipboard():
        return None
    try:
        if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
            return win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
        return None
    finally:
        win32clipboard.CloseClipboard()


def set_clipboard_text(text):
    if not _open_clipboard():
        raise RuntimeError("the clipboard is busy")
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text)
    finally:
        win32clipboard.CloseClipboard()


def type_text(text, restore_clipboard):
    """Paste text into whatever window has focus, then put the old clipboard back."""
    old = get_clipboard_text() if restore_clipboard else None
    set_clipboard_text(text)
    up = 0x0002
    user32.keybd_event(0x11, 0, 0, 0)   # Ctrl down
    user32.keybd_event(0x56, 0, 0, 0)   # V down
    user32.keybd_event(0x56, 0, up, 0)
    user32.keybd_event(0x11, 0, up, 0)
    if old is not None:
        time.sleep(0.4)  # let the target app read the clipboard first
        set_clipboard_text(old)


def beep(freq, enabled):
    if enabled:
        threading.Thread(target=winsound.Beep, args=(freq, 45), daemon=True).start()


# ---------------------------------------------------------------- audio

def to_16k(audio, rate):
    if rate == RATE or len(audio) == 0:
        return audio.astype(np.float32)
    n = int(len(audio) * RATE / rate)
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


class Recorder:
    def __init__(self, device):
        self.device = device
        self.stream = None
        self.chunks = []
        self.rate = RATE
        self.level = 0.0

    def mic_name(self):
        return sd.query_devices(self.device, "input")["name"]

    def start(self):
        self.chunks = []
        self.rate = int(sd.query_devices(self.device, "input")["default_samplerate"])
        self.stream = sd.InputStream(device=self.device, samplerate=self.rate, channels=1,
                                     dtype="float32", callback=self._on_audio)
        self.stream.start()

    def _on_audio(self, data, frames, t, status):
        self.chunks.append(data[:, 0].copy())
        self.level = float(np.sqrt(np.mean(data ** 2)))

    def stop(self):
        stream, self.stream = self.stream, None
        if stream:
            stream.stop()
            stream.close()
        self.level = 0.0
        audio = np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)
        return to_16k(audio, self.rate)


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    audio = np.frombuffer(raw, dtype).astype(np.float32)
    audio = (audio - 128) / 128 if width == 1 else audio / float(np.iinfo(dtype).max)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return to_16k(audio, rate)


# ---------------------------------------------------------------- whisper

def load_model(cfg):
    try:
        import torch  # noqa: F401  makes the CUDA/cuDNN DLLs that faster-whisper needs findable
    except ImportError:
        pass  # CPU-only setups don't need torch
    from faster_whisper import WhisperModel

    def make(device, compute):
        try:
            return WhisperModel(cfg["model"], device=device, compute_type=compute, local_files_only=True)
        except Exception as e:
            if "local" not in str(e).lower() and "find" not in str(e).lower():
                raise
            print(f"  Model '{cfg['model']}' is not on this PC yet, downloading it once...")
            return WhisperModel(cfg["model"], device=device, compute_type=compute)

    try:
        model = make(cfg["device"], cfg["compute_type"])
    except Exception as e:
        print(f"  GPU load failed ({e}); falling back to CPU, which is slower.")
        model = make("cpu", "int8")
    list(model.transcribe(np.zeros(RATE, np.float32), language="en")[0])  # warm-up
    return model


def transcribe(model, audio, cfg):
    language = None if cfg["language"] in (None, "", "auto") else cfg["language"]
    segments, _ = model.transcribe(
        audio, language=language, beam_size=5, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        initial_prompt=cfg["vocabulary"] or None,
        condition_on_previous_text=False, without_timestamps=True,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    if text.lower().strip(" .!?,") in PHANTOMS and float(np.sqrt(np.mean(audio ** 2))) < 0.02:
        return ""
    for wrong, right in cfg["replacements"].items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------- on-screen pill

class Overlay:
    """A small pill near the bottom of the screen. It never takes focus and clicks pass through it."""

    KEY = "#010203"  # painted pixels of this colour are see-through
    COLOURS = {"loading": "#fbbf24", "ready": "#34d399", "listening": "#f472b6",
               "working": "#60a5fa", "done": "#34d399", "quiet": "#6b7280", "error": "#f87171"}

    def __init__(self):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", self.KEY)
        self.root.attributes("-alpha", 0.0)
        s = self.root.winfo_fpixels("1i") / 96
        self.w, self.h = int(360 * s), int(44 * s)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{self.w}x{self.h}+{(sw - self.w) // 2}+{sh - self.h - int(90 * s)}")

        c = self.canvas = tk.Canvas(self.root, width=self.w, height=self.h, bg=self.KEY, highlightthickness=0)
        c.pack()
        r = self.h // 2
        c.create_polygon(self._pill(1, 1, self.w - 1, self.h - 1, r), smooth=True, fill="#15171d", outline="#2c313d")
        d = int(5 * s)
        self.dot = c.create_oval(r - d, r - d, r + d, r + d, fill="#6b7280", outline="")
        self.text = c.create_text(r + int(16 * s), r, anchor="w", fill="#e7e9ef", font=("Segoe UI", 10), text="")
        self.bar_x = self.w - int(96 * s)
        self.bars = [c.create_rectangle(0, 0, 0, 0, fill="#f472b6", outline="") for _ in range(10)]
        self.s = s

        fg = user32.GetForegroundWindow()
        self.root.update_idletasks()
        hwnd = user32.GetParent(self.root.winfo_id())
        style = user32.GetWindowLongW(hwnd, -20)  # GWL_EXSTYLE
        # NOACTIVATE | TOOLWINDOW (no taskbar button) | TRANSPARENT (click-through) | LAYERED
        user32.SetWindowLongW(hwnd, -20, style | 0x08000000 | 0x80 | 0x20 | 0x80000)
        self.root.deiconify()
        if fg:
            user32.SetForegroundWindow(fg)

        self.events = queue.Queue()
        self.hide_at = 0.0
        self.listening = False
        self.level = lambda: 0.0

    @staticmethod
    def _pill(x1, y1, x2, y2, r):
        return [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
                x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]

    def post(self, kind, text, hold=None):
        """Thread-safe: show a status. hold=None keeps it up until the next status."""
        self.events.put((kind, text, hold))

    def run(self):
        self._tick()
        self.root.mainloop()

    def _tick(self):
        while not self.events.empty():
            kind, text, hold = self.events.get()
            if kind == "hide":
                self.root.attributes("-alpha", 0.0)
                self.listening = False
                continue
            if len(text) > 46:
                text = text[:44] + "…"
            self.canvas.itemconfigure(self.text, text=text)
            self.canvas.itemconfigure(self.dot, fill=self.COLOURS.get(kind, "#6b7280"))
            self.listening = kind == "listening"
            self.root.attributes("-alpha", 0.95)
            self.hide_at = time.time() + hold if hold else 0.0
        if self.hide_at and time.time() > self.hide_at:
            self.root.attributes("-alpha", 0.0)
            self.hide_at = 0.0
        self._draw_level()
        self.root.after(33, self._tick)

    def _draw_level(self):
        s, h = self.s, self.h
        lvl = min(1.0, self.level() * 12) if self.listening else 0.0
        for i, bar in enumerate(self.bars):
            if not self.listening:
                self.canvas.coords(bar, 0, 0, 0, 0)
                continue
            wobble = 0.55 + 0.45 * np.sin(time.time() * 9 + i * 1.7)
            bh = max(2 * s, lvl * wobble * (h - 18 * s))
            x = self.bar_x + i * 7 * s
            self.canvas.coords(bar, x, h / 2 - bh / 2, x + 3 * s, h / 2 + bh / 2)


# ---------------------------------------------------------------- app

class App:
    def __init__(self, cfg):
        self.cfg = cfg
        self.vk, self.key_name = KEYS[cfg["hotkey"]]
        self.overlay = Overlay()
        self.recorder = Recorder(cfg["mic"])
        self.overlay.level = lambda: self.recorder.level
        self.jobs = queue.Queue()
        self.model = None
        threading.Thread(target=self._load, daemon=True).start()
        threading.Thread(target=self._watch_key, daemon=True).start()
        threading.Thread(target=self._work, daemon=True).start()

    def _load(self):
        self.overlay.post("loading", "Loading Whisper… (up to a minute)")
        t = time.time()
        try:
            self.model = load_model(self.cfg)
        except Exception as e:
            print(f"  Could not load Whisper: {e}")
            self.overlay.post("error", "Whisper failed to load, see console")
            return
        try:
            mic = self.recorder.mic_name()
        except Exception as e:
            mic = f"none found ({e})"
        print(f"  Ready in {time.time() - t:.0f}s. Mic: {mic}")
        print(f"  Hold {self.key_name}, speak, let go. Close this window to quit.\n")
        self.overlay.post("ready", f"Ready · hold {self.key_name} to talk", hold=3)

    def _watch_key(self):
        recording = blocked = shown = False
        started = 0.0
        while True:
            time.sleep(0.01)
            held = key_down(self.vk)
            if not held:
                if recording:
                    recording = False
                    audio = self.recorder.stop()
                    if time.time() - started >= self.cfg["min_seconds"] and shown:
                        beep(660, self.cfg["beeps"])
                        self.overlay.post("working", "Transcribing…")
                        self.jobs.put(audio)
                    else:
                        self.overlay.post("hide", "")
                blocked = shown = False
                continue
            if blocked:
                continue
            if not recording:
                if self.model is None:
                    self.overlay.post("loading", "Still loading Whisper…", hold=2)
                    blocked = True
                    continue
                try:
                    self.recorder.start()
                except Exception as e:
                    print(f"  Mic error: {e}")
                    self.overlay.post("error", "Mic error, see console", hold=3)
                    blocked = True
                    continue
                recording, started = True, time.time()
            elif other_key_down(self.vk):
                # The key was part of a shortcut (e.g. Ctrl+C), not dictation: drop it silently.
                self.recorder.stop()
                recording, blocked = False, True
                if shown:
                    self.overlay.post("hide", "")
            elif not shown and time.time() - started > 0.2:
                # Only announce after a short hold so quick shortcuts don't flash or beep.
                shown = True
                beep(880, self.cfg["beeps"])
                self.overlay.post("listening", "Listening…")

    def _work(self):
        while True:
            audio = self.jobs.get()
            t = time.time()
            try:
                text = transcribe(self.model, audio, self.cfg)
            except Exception as e:
                print(f"  Transcription failed: {e}")
                self.overlay.post("error", "Transcription failed, see console", hold=3)
                continue
            if not text:
                self.overlay.post("quiet", "Didn't catch anything", hold=1.5)
                continue
            try:
                type_text(text + " ", self.cfg["restore_clipboard"])
            except Exception as e:
                print(f"  Could not type the text ({e}). Text: {text}")
                self.overlay.post("error", "Couldn't type it, see console", hold=3)
                continue
            print(f"  [{len(audio) / RATE:.1f}s → {time.time() - t:.2f}s] {text}")
            self.overlay.post("done", text, hold=1.6)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_config()
    if len(sys.argv) == 3 and sys.argv[1] == "--test":
        t = time.time()
        model = load_model(cfg)
        print(f"loaded in {time.time() - t:.1f}s")
        audio = read_wav(sys.argv[2])
        t = time.time()
        text = transcribe(model, audio, cfg)
        print(f"[{len(audio) / RATE:.1f}s audio → {time.time() - t:.2f}s] {text}")
        return

    kernel32.CreateMutexW(None, False, "Local\\WhisperPushToTalk")
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        print("Whisper Push-to-Talk is already running.")
        time.sleep(3)
        return
    kernel32.SetConsoleTitleW("Whisper Push-to-Talk")
    print("Whisper Push-to-Talk (runs locally, nothing leaves this PC)")
    print(f"  Model: {cfg['model']} on {cfg['device'].upper()}, language: {cfg['language']}")
    App(cfg).overlay.run()


if __name__ == "__main__":
    main()
