/**
 * LanguageContext — Single source of truth for language, voice, and speech settings.
 *
 * Persists all preferences in localStorage so they survive page refreshes.
 * Every frontend module and every API call reads from this context.
 */
import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import type { ReactNode } from 'react';
import { translate } from '../i18n';
import type { SupportedLanguage } from '../i18n';

// ── Storage keys ────────────────────────────────────────────────────────────
const LS_LANGUAGE = 'iasis_language';
const LS_VOICE_GENDER = 'iasis_voice_gender';
const LS_SPEECH_RATE = 'iasis_speech_rate';

// ── Types ───────────────────────────────────────────────────────────────────

export type VoiceGender = 'female' | 'male';
export type SpeechRate = 'slow' | 'normal' | 'fast';

const SPEED_MAP: Record<SpeechRate, number> = {
  slow: 0.8,
  normal: 1.0,
  fast: 1.3,
};

/** Resolve the correct TTS voice ID from language + gender. */
const VOICE_MAP: Record<SupportedLanguage, Record<VoiceGender, string>> = {
  en: {
    female: 'en-US-AriaNeural',
    male: 'en-US-GuyNeural',
  },
  bn: {
    female: 'bn-BD-NabanitaNeural',
    male: 'bn-BD-PradeepNeural',
  },
};

interface LanguageContextProps {
  /** Current language — 'en' or 'bn' */
  language: SupportedLanguage;
  /** Update language (persists to localStorage) */
  setLanguage: (lang: SupportedLanguage) => void;
  /** Translation helper — t('chat.placeholder') */
  t: (key: string) => string;

  /** Voice gender preference */
  voiceGender: VoiceGender;
  setVoiceGender: (g: VoiceGender) => void;

  /** Speech rate preference */
  speechRate: SpeechRate;
  setSpeechRate: (r: SpeechRate) => void;

  /** Resolved numeric speed for API calls (0.8 | 1.0 | 1.3) */
  speed: number;

  /** Resolved TTS voice ID for API calls */
  voiceId: string;
}

const LanguageContext = createContext<LanguageContextProps | undefined>(undefined);

// ── Helper: safe localStorage read ──────────────────────────────────────────

function readLS<T extends string>(key: string, fallback: T, valid: T[]): T {
  try {
    const val = localStorage.getItem(key) as T | null;
    if (val && valid.includes(val)) return val;
  } catch { /* SSR or private browsing */ }
  return fallback;
}

// ── Provider ────────────────────────────────────────────────────────────────

export const LanguageProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [language, setLanguageState] = useState<SupportedLanguage>(
    () => readLS<SupportedLanguage>(LS_LANGUAGE, 'en', ['en', 'bn'])
  );
  const [voiceGender, setVoiceGenderState] = useState<VoiceGender>(
    () => readLS<VoiceGender>(LS_VOICE_GENDER, 'female', ['female', 'male'])
  );
  const [speechRate, setSpeechRateState] = useState<SpeechRate>(
    () => readLS<SpeechRate>(LS_SPEECH_RATE, 'normal', ['slow', 'normal', 'fast'])
  );

  // Persist on change
  const setLanguage = useCallback((lang: SupportedLanguage) => {
    setLanguageState(lang);
    try { localStorage.setItem(LS_LANGUAGE, lang); } catch { /* ignore */ }
  }, []);

  const setVoiceGender = useCallback((g: VoiceGender) => {
    setVoiceGenderState(g);
    try { localStorage.setItem(LS_VOICE_GENDER, g); } catch { /* ignore */ }
  }, []);

  const setSpeechRate = useCallback((r: SpeechRate) => {
    setSpeechRateState(r);
    try { localStorage.setItem(LS_SPEECH_RATE, r); } catch { /* ignore */ }
  }, []);

  // Set <html lang="..."> attribute for accessibility
  useEffect(() => {
    document.documentElement.lang = language;
  }, [language]);

  const t = useCallback(
    (key: string) => translate(language, key),
    [language]
  );

  const speed = SPEED_MAP[speechRate];
  const voiceId = VOICE_MAP[language][voiceGender];

  return (
    <LanguageContext.Provider
      value={{
        language,
        setLanguage,
        t,
        voiceGender,
        setVoiceGender,
        speechRate,
        setSpeechRate,
        speed,
        voiceId,
      }}
    >
      {children}
    </LanguageContext.Provider>
  );
};

// ── Hook ────────────────────────────────────────────────────────────────────

export const useLanguage = (): LanguageContextProps => {
  const ctx = useContext(LanguageContext);
  if (!ctx) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return ctx;
};
