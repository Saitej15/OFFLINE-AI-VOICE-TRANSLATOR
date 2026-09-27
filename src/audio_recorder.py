import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time
import os

def record_audio(filename="input.wav", duration=5, fs=44100):
    """
    Records audio from the microphone for a fixed duration.
    
    Args:
        filename (str): Path to save the WAV file.
        duration (int): Duration in seconds.
        fs (int): Sampling rate.
    """
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    
    wav.write(filename, fs, recording)
    return filename

def list_devices():
    """List available audio devices."""
    print(sd.query_devices())

if __name__ == "__main__":
    record_audio()
