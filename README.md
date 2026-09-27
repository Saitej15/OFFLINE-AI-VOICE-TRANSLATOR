# 🎤 Offline AI Voice Translator

A fully **offline**, **privacy-first** AI-powered voice translation system built with Python and Streamlit. Speak in one language, and hear the translation in another — all running locally on your machine with no internet required after the initial model download.

---

## ✨ Features

- 🎙️ **Voice Recording** — Record audio directly from your microphone
- 🧠 **Speech-to-Text (STT)** — Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (OpenAI Whisper) for accurate, multilingual transcription
- 🌐 **Neural Machine Translation** — Uses Meta's [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) model supporting 200+ languages
- 🔊 **Text-to-Speech (TTS)** — Powered by Meta's [MMS-TTS](https://huggingface.co/facebook/mms-tts-eng) for natural-sounding speech output
- 🔒 **Fully Offline** — After the first run, everything works without an internet connection
- 🖥️ **Simple Web UI** — Clean Streamlit interface, no coding required to use

---

## 🗂️ Project Structure

```
OFFLINE-AI-VOICE-TRANSLATOR/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── verify_setup.py         # Script to verify your environment setup
└── src/
    ├── audio_recorder.py   # Microphone audio capture (sounddevice)
    ├── stt_engine.py       # Speech-to-Text using faster-whisper
    ├── translator.py       # Translation using NLLB-200 (HuggingFace)
    └── tts_engine.py       # Text-to-Speech using MMS-TTS (VITS)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- A working microphone
- ~3–4 GB of disk space for AI models (downloaded on first run)

### 1. Clone the Repository

```bash
git clone https://github.com/Saitej15/OFFLINE-AI-VOICE-TRANSLATOR.git
cd OFFLINE-AI-VOICE-TRANSLATOR
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Your Setup

```bash
python verify_setup.py
```

### 5. Run the App

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`

> ⚠️ **First Run Note:** The app will automatically download AI models (~2–3 GB). This only happens once. After that, everything runs fully offline.

---

## 🌍 Supported Languages

| Language | STT | Translation | TTS |
|----------|-----|-------------|-----|
| English  | ✅  | ✅          | ✅  |
| French   | ✅  | ✅          | ✅  |
| Spanish  | ✅  | ✅          | ✅  |
| German   | ✅  | ✅          | ✅  |
| Hindi    | ✅  | ✅          | ✅  |

> Translation supports 200+ languages via NLLB-200. Additional TTS languages can be added by extending the model map in `src/tts_engine.py`.

---

## 🧩 How It Works

```
🎙️ Microphone Input
       ↓
🧠 faster-whisper  →  Transcribed Text + Detected Language
       ↓
🌐 NLLB-200        →  Translated Text
       ↓
🔊 MMS-TTS         →  Spoken Audio Output
```

1. **Record** your voice for a configurable duration (2–10 seconds)
2. **Whisper** transcribes it to text and auto-detects the source language
3. **NLLB-200** translates the text into your chosen target language
4. **MMS-TTS** synthesizes the translation into speech and plays it back

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|-----------|
| UI Framework | [Streamlit](https://streamlit.io/) |
| Speech-to-Text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (OpenAI Whisper) |
| Translation | [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) |
| Text-to-Speech | [facebook/mms-tts](https://huggingface.co/facebook/mms-tts-eng) (VITS) |
| Audio Capture | [sounddevice](https://python-sounddevice.readthedocs.io/) + [scipy](https://scipy.org/) |
| ML Framework | [PyTorch](https://pytorch.org/) + [HuggingFace Transformers](https://huggingface.co/transformers/) |

---

## 🛠️ Configuration

You can customize the following in `app.py`:

- **Whisper model size** — Change `model_size="base"` to `"small"`, `"medium"`, or `"large"` for better accuracy (at the cost of speed and memory)
- **Recording duration** — Adjustable via slider in the UI (2–10 seconds)
- **Languages** — Add more languages to the `LANGUAGES` dictionary in `app.py` and the model maps in `src/tts_engine.py`

---

## 📋 Requirements

```
streamlit
sounddevice
numpy
scipy
torch
torchaudio
transformers
faster-whisper
sentencepiece
sacremoses
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open an issue for bug reports or feature requests
- Submit a pull request with improvements
- Add support for more languages in the TTS engine

---

## 📄 License

This project is open-source. Feel free to use, modify, and distribute it.

---

## 👤 Author

**Saitej Reddy**  
📧 saitej1525se@gmail.com  
🔗 [GitHub](https://github.com/Saitej15)

---

> 💡 **Tip:** For better transcription accuracy, use a quiet environment and speak clearly. Upgrade to `model_size="small"` or `"medium"` in `app.py` for improved results.
