import streamlit as st
import os
import time
from src.audio_recorder import record_audio
from src.stt_engine import STTEngine
from src.translator import Translator
from src.tts_engine import TTSEngine

# Page Config
st.set_page_config(page_title="Offline AI Voice Translator", page_icon="🎤", layout="centered")

# Custom CSS for styling
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 50px;
    }
    .success-box {
        padding: 10px;
        background-color: #d4edda;
        color: #155724;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .text-output {
        padding: 15px;
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎤 Offline AI Voice Translator")
st.markdown("Fully offline, privacy-first voice translation system.")

# --- Models Initialization (Cached) ---
@st.cache_resource
def load_stt():
    return STTEngine(model_size="base", device="cpu") # Use 'small' or 'medium' for better accuracy

@st.cache_resource
def load_translator():
    # Load default NLLB.
    return Translator()

@st.cache_resource
def load_tts():
    # Load default English TTS
    return TTSEngine()

# Load models with a spinner
with st.spinner("Loading AI Models... (First run might take time to download)"):
    stt_engine = load_stt()
    translator_engine = load_translator()
    tts_engine = load_tts()

# --- Language Mapping ---
# Map: Display Name -> {NLLB Code, MMS Code, Whisper Code (implicit)}
LANGUAGES = {
    "English": {"nllb": "eng_Latn", "mms": "eng"},
    "French": {"nllb": "fra_Latn", "mms": "fra"},
    "Spanish": {"nllb": "spa_Latn", "mms": "spa"},
    "German": {"nllb": "deu_Latn", "mms": "deu"},
    "Hindi": {"nllb": "hin_Deva", "mms": "hin"},
    # Add more as needed
}

# --- UI Layout ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Input Settings")
    duration = st.slider("Step 1: Recording Duration (sec)", 2, 10, 5)

with col2:
    st.subheader("Target Language")
    target_lang_name = st.selectbox("Step 2: Translate to", list(LANGUAGES.keys()))

# --- Step 3: Record ---
st.divider()
st.subheader("Step 3: Record & Process")

if st.button("🔴 Start Recording & Translate", type="primary"):
    # 1. Record
    status_text = st.empty()
    status_text.info(f"Recording for {duration} seconds...")
    progress_bar = st.progress(0)
    
    audio_file = "input.wav"
    record_audio(audio_file, duration=duration)
    
    # 2. Transcribe
    status_text.info("Transcribing audio...")
    progress_bar.progress(30)
    
    try:
        stt_result = stt_engine.transcribe(audio_file)
        source_text = stt_result['text']
        detected_lang = stt_result['language']
        
        st.markdown("**Example Transcription:**")
        st.markdown(f"<div class='text-output'>{source_text}</div>", unsafe_allow_html=True)
        st.caption(f"Detected Language: {detected_lang}")
        
    except Exception as e:
        st.error(f"Error in transcription: {e}")
        st.stop()
        
    # 3. Translate
    status_text.info(f"Translating to {target_lang_name}...")
    progress_bar.progress(60)
    
    # Determine Source NLLB Code
    # Whisper returns 2-letter codes usually (en, fr, es). We need to map to NLLB.
    # Simple heuristic map for demo:
    whisper_to_nllb = {
        "en": "eng_Latn", "fr": "fra_Latn", "es": "spa_Latn", "de": "deu_Latn", "hi": "hin_Deva"
    }
    src_lang_code = whisper_to_nllb.get(detected_lang, "eng_Latn") # Default to Eng if unknown
    tgt_lang_code = LANGUAGES[target_lang_name]["nllb"]
    
    try:
        translated_text = translator_engine.translate(source_text, src_lang=src_lang_code, tgt_lang=tgt_lang_code)
        
        st.markdown(f"**Translation ({target_lang_name}):**")
        st.markdown(f"<div class='text-output' style='background-color:#e8f0fe;'>{translated_text}</div>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error in translation: {e}")
        st.stop()

    # 4. Speak (TTS)
    status_text.info("Generating Audio...")
    progress_bar.progress(80)
    
    # Update TTS model to target language if valid
    mms_lang = LANGUAGES[target_lang_name].get("mms")
    if mms_lang:
        # Note: In a real efficient app, we wouldn't reload model every click if same.
        # But our TTSEngine handles lazy checking.
        tts_engine.update_language(mms_lang)
    
    try:
        output_audio_path = "output.wav"
        tts_engine.synthesize(translated_text, output_audio_path)
        
        st.audio(output_audio_path, format="audio/wav", autoplay=True)
        status_text.success("Done!")
        progress_bar.progress(100)
        
    except Exception as e:
        st.error(f"Error in TTS: {e}")

st.divider()
st.info("Note: First run will download models (~2-3 GB). Subsequent runs are fully offline.")
