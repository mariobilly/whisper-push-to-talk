# Whisper Push-to-Talk

Hold **Right Ctrl**, speak, let go, and your words are typed wherever the cursor is:
a Claude Code terminal, a browser, Notepad, anything. Local dictation for Windows.

- Speech-to-text is **Whisper large-v3** running on your GPU through
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Audio never leaves your PC and it works offline.
- About 1.6 GB of VRAM while it runs (`int8_float16`). A sentence typically takes about half a second.
- A small pill at the bottom of the screen shows Listening / Transcribing / the text.
  It never takes focus, and clicks pass through it.
- Pressing Right Ctrl as part of a shortcut (Right Ctrl+C and so on) is ignored. Only a hold
  longer than a fifth of a second with no other key starts dictation.
- One file, no account, no API key.

## Install

Windows 10 or 11 and Python 3.10 or newer.

```bash
git clone https://github.com/mariobilly/whisper-push-to-talk.git
cd whisper-push-to-talk
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU (NVIDIA). Skip this for CPU only
pip install -r requirements.txt
```

For CPU only, set `"device": "cpu"` and `"compute_type": "int8"` in `config.json`, and pick a smaller
`model` such as `small` or `base`.

## Run

```bash
python whisper_ptt.py
```

A console window logs each transcription. Close it to quit. Only one copy runs at a time.
The first run downloads the model once (about 3 GB for `large-v3`), to `~/.cache/huggingface/hub`.

`python whisper_ptt.py --test some.wav` transcribes a file and prints the text (handy for checks).

Tip: make a Windows shortcut that runs `python whisper_ptt.py` with *Run: Minimized*, and put it in
`shell:startup` to have it ready after every sign-in.

## Settings (`config.json`, restart after editing)

| Key | What it does |
|---|---|
| `hotkey` | `right_ctrl`, `right_alt`, `right_shift`, `f8`, `f9`, `f10`, `scroll_lock`, `pause`, `insert` |
| `language` | `en`, `ar`, `fr`, … or `auto` to detect (auto can misjudge very short clips) |
| `model` | `large-v3` (best), `medium`, `small`, `small.en`, `base` |
| `device` / `compute_type` | `cuda` + `int8_float16` (default), or `cpu` + `int8` to keep the GPU free |
| `vocabulary` | Words Whisper should expect (names, tools, jargon) |
| `replacements` | Fix words it keeps getting wrong, e.g. `"Cloud Code": "Claude Code"` |
| `mic` | `null` = the Windows default mic, or a device name/number |
| `beeps` | Short start and stop beeps |
| `restore_clipboard` | Put your old clipboard text back after typing |
| `min_seconds` | Recordings shorter than this are dropped |

## How it types

The text is put on the clipboard and pasted with Ctrl+V into the focused window, then your previous
clipboard text is restored. That works in terminals, browsers and editors alike.

## License

[MIT](LICENSE)
