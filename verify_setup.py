import sys
import os

def check_modules():
    print("Checking imports...")
    try:
        import streamlit
        import sounddevice
        import faster_whisper
        import transformers
        import torch
        print("Imports successful.")
    except ImportError as e:
        print(f"Import failed: {e}")
        sys.exit(1)

def check_models():
    print("Checking Model Loading (this may download models)...")
    
    # 1. STT
    try:
        from src.stt_engine import STTEngine
        print("Loading STT Engine (Whisper)...")
        _ = STTEngine(model_size="tiny", device="cpu") # Use tiny for quick verify
        print("STT Engine loaded.")
    except Exception as e:
        print(f"STT Engine failed: {e}")

    # 2. Translation
    try:
        from src.translator import Translator
        print("Loading Translator (NLLB)...")
        # Use a smaller model for verification if possible, or just the default
        _ = Translator(model_name="facebook/nllb-200-distilled-600M") 
        print("Translator loaded.")
    except Exception as e:
        print(f"Translator failed: {e}")

    # 3. TTS
    try:
        from src.tts_engine import TTSEngine
        print("Loading TTS Engine (MMS)...")
        _ = TTSEngine(model_name="facebook/mms-tts-eng")
        print("TTS Engine loaded.")
    except Exception as e:
        print(f"TTS Engine failed: {e}")

if __name__ == "__main__":
    check_modules()
    check_models()
    print("Verification Complete.")
