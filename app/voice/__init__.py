"""
IASIS AI — Production-Grade Voice System

Modules:
    stt/         Speech-to-Text (faster-whisper primary, openai-whisper fallback)
    tts/         Text-to-Speech (Kokoro primary, Piper fallback, edge-tts last resort)
    audio_utils/ Audio validation, conversion, silence detection
    schemas.py   Pydantic request/response models
"""
