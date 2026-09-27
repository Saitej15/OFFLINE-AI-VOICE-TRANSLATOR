from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile as wav
import numpy as np

class TTSEngine:
    def __init__(self, model_name="facebook/mms-tts-eng", device="cpu"):
        """
        Initializes the TTS model.
        We will default to facebook/mms-tts-eng for English.
        For multilang, we might need to swap models or use a multilang model like SpeechT5.
        For simplicity in this offline CPU demo, we might load specific small models per language 
        or use a multi-speaker one.
        
        Let's use MMS-TTS which has separate checkpoints for languages, 
        making it modular and small enough.
        """
        self.device = device
        self.current_model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model(model_name)

    def _load_model(self, model_name):
        print(f"Loading TTS model: {model_name}...")
        self.model = VitsModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.device == "cuda":
            self.model.to("cuda")
        print("TTS model loaded.")

    def synthesize(self, text, output_path="output.wav"):
        """
        Synthesizes text to audio.
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        if self.device == "cuda":
            inputs = inputs.to("cuda")

        with torch.no_grad():
            output = self.model(**inputs).waveform

        # Convert to numpy and save
        waveform = output.cpu().numpy().squeeze()
        
        # MMS usually is 16kHz ?? Check config.
        # VitsModel config usually has sampling_rate
        sample_rate = self.model.config.sampling_rate
        
        wav.write(output_path, sample_rate, waveform)
        return output_path

    def update_language(self, lang_code_iso):
        """
        Switch model based on language if using MMS-TTS separate checkpoints.
        lang_code_iso: e.g. 'eng', 'fra', 'deu', 'hin'.
        """
        # Map simple ISO codes to MMS model names
        # This is a broad assumption; in a real app we'd map carefully.
        model_map = {
            "eng": "facebook/mms-tts-eng",
            "fra": "facebook/mms-tts-fra",
            "deu": "facebook/mms-tts-deu",
            "spa": "facebook/mms-tts-spa",
            "hin": "facebook/mms-tts-hin",
            # Add others as needed
        }
        
        target_model = model_map.get(lang_code_iso)
        if target_model and target_model != self.current_model_name:
            self._load_model(target_model)
            self.current_model_name = target_model
        elif not target_model:
            print(f"Warning: No specific TTS model found for {lang_code_iso}, keeping current.")

if __name__ == "__main__":
    tts = TTSEngine()
    tts.synthesize("Hello world, this is a test.", "test_tts.wav")
