from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

class Translator:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M", device="cpu"):
        """
        Initializes the Translation model (NLLB).
        
        Args:
            model_name (str): HuggingFace model name.
            device (str): 'cpu' or 'cuda'.
        """
        print(f"Loading Translation model: {model_name} on {device}...")
        self.device = 0 if device == "cuda" and torch.cuda.is_available() else -1
        
        # Load model and tokenizer
        # We use the pipeline for easier inference
        self.translator = pipeline("translation", model=model_name, device=self.device)
        print("Translation model loaded.")

    def translate(self, text, src_lang, tgt_lang):
        """
        Translates text from source to target language.
        
        Args:
            text (str): Text to translate.
            src_lang (str): Source language code (NLLB format, e.g., 'eng_Latn').
            tgt_lang (str): Target language code (NLLB format, e.g., 'fra_Latn').
            
        Returns:
            str: Translated text.
        """
        # NLLB requires forcing the source and target language codes
        # Note: The pipeline handle for NLLB might require specific usage or direct model calls
        # if the pipeline wrapping doesn't support src_lang param easily.
        # Let's use the underlying model/tokenizer for precise control if needed, 
        # but pipeline usually handles it with src_lang/tgt_lang args if supported.
        
        # For NLLB, standard transformers pipeline usage:
        output = self.translator(text, src_lang=src_lang, tgt_lang=tgt_lang, max_length=400)
        translated_text = output[0]['translation_text']
        return translated_text

# Language mapping helper (Simple subset for demo)
LANG_CODES = {
    "English": "eng_Latn",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "German": "deu_Latn",
    "Hindi": "hin_Deva",
    "Telugu": "tel_Telu",
    "Chinese": "zho_Hans",
    "Japanese": "jpn_Jpan",
    "Russian": "rus_Cyrl"
}

if __name__ == "__main__":
    t = Translator()
    res = t.translate("Hello, how are you?", src_lang="eng_Latn", tgt_lang="fra_Latn")
    print(res)
