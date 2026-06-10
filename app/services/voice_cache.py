"""
Voice Cache — Pre-synthesized audio for common medical questions (EN + BN).

Instant audio (0ms TTS) for ~40 common follow-up questions.
In-memory dict + disk persistence (voice_cache/*.mp3).
Warmup runs in background thread at startup.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path('voice_cache')

_COMMON_EN: list[str] = [
    "How long have you had the cough?",
    "Do you have chest pain?",
    "Is the cough dry or producing phlegm?",
    "Do you have a fever?",
    "Are you short of breath?",
    "When did your symptoms start?",
    "How severe is the pain on a scale of 1 to 10?",
    "Do you have blood in your cough?",
    "Do you have night sweats?",
    "Have you lost weight recently?",
    "Are you currently taking any medications?",
    "Do you smoke?",
    "Do you have diabetes?",
    "Is the pain sharp or dull?",
    "Does anything make it better or worse?",
    "Where exactly is the pain?",
    "Do you have nausea or vomiting?",
    "Have you been in contact with anyone sick?",
    "Do you have any allergies?",
    "Tell me more about your symptoms.",
    "How long have you had this pain?",
    "Is it getting worse?",
    "Do you have a headache?",
    "Are you feeling dizzy?",
    "Do you have any swelling?",
    "I understand. Could you describe the pain more?",
    "Thank you. One more question — do you have fatigue?",
]

_COMMON_BN: list[str] = [
    "কতদিন ধরে কাশি হচ্ছে?",
    "আপনার কি বুকে ব্যথা আছে?",
    "কাশি কি শুকনো নাকি কফ বের হচ্ছে?",
    "আপনার কি জ্বর আছে?",
    "শ্বাস নিতে কষ্ট হচ্ছে?",
    "লক্ষণগুলো কখন শুরু হয়েছে?",
    "ব্যথা কতটা তীব্র, ১ থেকে ১০ এর মধ্যে?",
    "কাশিতে কি রক্ত আছে?",
    "রাতে কি ঘাম হয়?",
    "সম্প্রতি কি ওজন কমেছে?",
    "বর্তমানে কি কোনো ওষুধ খাচ্ছেন?",
    "আপনি কি ধূমপান করেন?",
    "আপনার কি ডায়াবেটিস আছে?",
    "ব্যথাটা কি তীক্ষ্ণ নাকি মৃদু?",
    "ব্যথা কোথায়?",
    "কি করলে ব্যথা বাড়ে বা কমে?",
    "বমি বমি ভাব বা বমি হচ্ছে?",
    "কোনো অসুস্থ মানুষের সাথে যোগাযোগ হয়েছে?",
    "আপনার কি কোনো অ্যালার্জি আছে?",
    "আপনার লক্ষণ সম্পর্কে আরও বলুন।",
    "কতদিন ধরে ব্যথা হচ্ছে?",
    "ব্যথা কি বাড়ছে?",
    "মাথাব্যথা আছে?",
    "মাথা ঘুরছে?",
    "কোনো ফোলাভাব আছে?",
    "বুঝতে পেরেছি। ব্যথার বিবরণ আরও দিন।",
]


class VoiceCache:
    """Thread-safe audio cache for pre-synthesized medical questions."""

    def __init__(self):
        self._cache: dict[str, bytes] = {}
        self._lock = threading.RLock()
        self._warmed = False
        CACHE_DIR.mkdir(exist_ok=True)

    def _key(self, text: str, language: str) -> str:
        return hashlib.md5(f'{language}:{text.strip()}'.encode()).hexdigest()

    def get(self, text: str, language: str) -> Optional[bytes]:
        """Return cached audio bytes or None on cache miss."""
        key = self._key(text, language)

        with self._lock:
            if key in self._cache:
                return self._cache[key]

        path = CACHE_DIR / f'{key}.mp3'
        if path.exists():
            try:
                data = path.read_bytes()
                with self._lock:
                    self._cache[key] = data
                return data
            except OSError:
                pass

        return None

    def put(self, text: str, language: str, audio_bytes: bytes) -> None:
        """Store audio in memory and on disk."""
        key = self._key(text, language)
        with self._lock:
            self._cache[key] = audio_bytes
        try:
            (CACHE_DIR / f'{key}.mp3').write_bytes(audio_bytes)
        except OSError as exc:
            logger.warning('VoiceCache disk write failed: %s', exc)

    def warm_up(self) -> None:
        """Pre-synthesize common questions in a background thread. Non-blocking."""
        if self._warmed:
            return
        self._warmed = True
        t = threading.Thread(target=self._warmup_worker, daemon=True, name='VoiceCacheWarm')
        t.start()

    def _warmup_worker(self) -> None:
        import asyncio
        try:
            from app.voice.tts.engine import tts_engine
        except Exception as exc:
            logger.warning('VoiceCache: cannot import TTS engine — %s', exc)
            return

        if not tts_engine.is_ready:
            tts_engine.load()
        if not tts_engine.is_ready:
            logger.warning('VoiceCache: TTS not ready — skipping warmup')
            return

        voices = {'en': 'af_heart', 'bn': 'bn-BD-NabanitaNeural'}
        batches: list[tuple[list[str], str]] = [
            (_COMMON_EN, 'en'),
            (_COMMON_BN, 'bn'),
        ]
        synthesized = skipped = 0

        for questions, lang in batches:
            voice_id = voices[lang]
            for text in questions:
                key = self._key(text, lang)
                if (CACHE_DIR / f'{key}.mp3').exists():
                    skipped += 1
                    continue
                try:
                    loop = asyncio.new_event_loop()
                    audio_bytes, _ = loop.run_until_complete(
                        tts_engine.synthesize_async(text, voice_id, 1.0, 'mp3')
                    )
                    loop.close()
                    self.put(text, lang, audio_bytes)
                    synthesized += 1
                except Exception as exc:
                    logger.debug('VoiceCache skip "%s": %s', text[:30], exc)

        logger.info(
            'VoiceCache warmup done — %d synthesized, %d already cached',
            synthesized, skipped,
        )

    def stats(self) -> dict:
        with self._lock:
            n_mem = len(self._cache)
        n_disk = sum(1 for _ in CACHE_DIR.glob('*.mp3'))
        return {'in_memory': n_mem, 'on_disk': n_disk, 'warmed': self._warmed}


voice_cache = VoiceCache()
