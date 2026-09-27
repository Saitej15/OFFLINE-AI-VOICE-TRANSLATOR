from faster_whisper import WhisperModel
import os

class STTEngine:
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        """
        Initializes the Whisper model.
        
        Args:
            model_size (str): Size of the model (tiny, base, small, medium, large).
            device (str): Device to run on (cpu or cuda).
            compute_type (str): Quantization type (int8, float16, float32).
        """
        print(f"Loading Whisper model: {model_size} on {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Whisper model loaded.")

    def transcribe(self, audio_path):
        """
        Transcribes audio file to text.
        
        Args:
            audio_path (str): Path to the input wav file.
            
        Returns:
            dict: {'text': str, 'language': str}
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        
        detected_language = info.language
        print(f"Detected language: {detected_language} with probability {info.language_probability}")
        
        text = ""
        for segment in segments:
            text += segment.text + " "
            
        return {
            "text": text.strip(),
            "language": detected_language
        }

if __name__ == "__main__":
    # Test
    engine = STTEngine()
    # result = engine.transcribe("input.wav")
    # print(result)
