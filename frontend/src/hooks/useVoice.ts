/**
 * useVoice — Custom React hook for voice interactions.
 *
 * Features:
 *   - Microphone recording with MediaRecorder
 *   - Push-to-talk and click-to-toggle recording
 *   - Auto-stop on silence detection (Web Audio API AnalyserNode)
 *   - Real-time volume/waveform data for visualization
 *   - Audio playback with controls
 *   - Full voice chat pipeline (audio → STT → AI → TTS → playback)
 *   - Explicit language control (no auto-detection)
 *   - Error handling for missing mic, permission denied, etc.
 */
import { useState, useRef, useCallback, useEffect } from 'react';
import { apiService } from '../services/api';
import type { VoiceChatResponse } from '../types/api';

export type VoiceState = 'idle' | 'recording' | 'processing' | 'playing';

interface UseVoiceOptions {
    conversationId: string;
    /** Explicit language for STT/TTS — 'en' or 'bn'. No auto-detection. */
    language: string;
    /** Silence threshold (0–255). Lower = more sensitive. Default 15. */
    silenceThreshold?: number;
    /** Silence duration (ms) before auto-stop. Default 2500. */
    silenceDuration?: number;
    /** Patient demographics for voice chat */
    age?: number | null;
    gender?: string | null;
    /** TTS voice preference (resolved from LanguageContext) */
    voiceId?: string;
    /** TTS speed (resolved from LanguageContext) */
    speed?: number;
    /** Callback when voice chat completes */
    onVoiceChatComplete?: (response: VoiceChatResponse) => void;
    /** Callback when an error occurs */
    onError?: (error: string) => void;
}

interface UseVoiceReturn {
    state: VoiceState;
    /** Start recording */
    startRecording: () => Promise<void>;
    /** Stop recording and process */
    stopRecording: () => void;
    /** Toggle recording on/off */
    toggleRecording: () => Promise<void>;
    /** Play TTS audio from base64 string */
    playAudio: (base64Audio: string, format?: string) => void;
    /** Play TTS audio from text */
    speakText: (text: string) => Promise<void>;
    /** Stop any playing audio */
    stopAudio: () => void;
    /** Current volume level (0–1) for waveform visualization */
    volume: number;
    /** Frequency data array for waveform (128 values, 0–255) */
    frequencyData: Uint8Array | null;
    /** Recording duration in seconds */
    recordingDuration: number;
    /** Whether the browser supports voice features */
    isSupported: boolean;
    /** Last error message */
    error: string | null;
}

export function useVoice(options: UseVoiceOptions): UseVoiceReturn {
    const {
        conversationId,
        language,
        silenceThreshold = 15,
        silenceDuration = 2500,
        age,
        gender,
        voiceId,
        speed = 1.0,
        onVoiceChatComplete,
        onError,
    } = options;

    const [state, setState] = useState<VoiceState>('idle');
    const [volume, setVolume] = useState(0);
    const [frequencyData, setFrequencyData] = useState<Uint8Array | null>(null);
    const [recordingDuration, setRecordingDuration] = useState(0);
    const [error, setError] = useState<string | null>(null);

    // Refs for media objects
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const streamRef = useRef<MediaStream | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const animationFrameRef = useRef<number | null>(null);
    const silenceTimerRef = useRef<number | null>(null);
    const durationTimerRef = useRef<number | null>(null);
    const audioPlayerRef = useRef<HTMLAudioElement | null>(null);

    const isSupported = typeof navigator !== 'undefined' &&
        !!navigator.mediaDevices?.getUserMedia &&
        typeof MediaRecorder !== 'undefined';

    // ── Cleanup ──────────────────────────────────────────────────────────────

    const cleanup = useCallback(() => {
        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }
        if (silenceTimerRef.current) {
            clearTimeout(silenceTimerRef.current);
            silenceTimerRef.current = null;
        }
        if (durationTimerRef.current) {
            clearInterval(durationTimerRef.current);
            durationTimerRef.current = null;
        }
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(t => t.stop());
            streamRef.current = null;
        }
        if (audioContextRef.current?.state !== 'closed') {
            audioContextRef.current?.close().catch(() => {});
            audioContextRef.current = null;
        }
        analyserRef.current = null;
        setVolume(0);
        setFrequencyData(null);
        setRecordingDuration(0);
    }, []);

    useEffect(() => {
        return () => {
            cleanup();
            // Directly stop audio on unmount — avoids stale stopAudio closure
            if (audioPlayerRef.current) {
                audioPlayerRef.current.pause();
                audioPlayerRef.current = null;
            }
        };
    }, [cleanup]);

    // ── Volume monitoring with AnalyserNode ──────────────────────────────────

    const startVolumeMonitoring = useCallback((stream: MediaStream) => {
        try {
            const audioContext = new AudioContext();
            const analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            analyser.smoothingTimeConstant = 0.8;

            const source = audioContext.createMediaStreamSource(stream);
            source.connect(analyser);

            audioContextRef.current = audioContext;
            analyserRef.current = analyser;

            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            let silenceStart: number | null = null;

            const tick = () => {
                if (!analyserRef.current) return;

                analyserRef.current.getByteFrequencyData(dataArray);

                // Compute average volume (0–255 → 0–1)
                let sum = 0;
                for (let i = 0; i < dataArray.length; i++) {
                    sum += dataArray[i];
                }
                const avg = sum / dataArray.length;
                setVolume(avg / 255);
                setFrequencyData(new Uint8Array(dataArray));

                // Silence detection
                if (avg < silenceThreshold) {
                    if (silenceStart === null) {
                        silenceStart = Date.now();
                    } else if (Date.now() - silenceStart > silenceDuration) {
                        // Auto-stop on prolonged silence
                        console.log('[useVoice] Auto-stopping: silence detected');
                        stopRecording();
                        return;
                    }
                } else {
                    silenceStart = null;
                }

                animationFrameRef.current = requestAnimationFrame(tick);
            };

            animationFrameRef.current = requestAnimationFrame(tick);
        } catch (err) {
            console.warn('[useVoice] Volume monitoring unavailable:', err);
        }
    }, [silenceThreshold, silenceDuration]);

    // ── Start recording ──────────────────────────────────────────────────────

    const startRecording = useCallback(async () => {
        if (!isSupported) {
            const msg = 'Voice recording is not supported in this browser.';
            setError(msg);
            onError?.(msg);
            return;
        }

        setError(null);

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 16000,
                },
            });

            streamRef.current = stream;
            audioChunksRef.current = [];

            // Determine best supported MIME type
            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : MediaRecorder.isTypeSupported('audio/webm')
                    ? 'audio/webm'
                    : '';

            const mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
            mediaRecorderRef.current = mediaRecorder;

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) audioChunksRef.current.push(e.data);
            };

            mediaRecorder.onstop = async () => {
                // Capture mimeType before cleanup nulls the stream
                const recordedMimeType = mediaRecorder.mimeType || 'audio/webm';
                cleanup();
                const audioBlob = new Blob(audioChunksRef.current, { type: recordedMimeType });

                if (audioBlob.size < 100) {
                    setState('idle');
                    return;
                }

                // Process voice chat pipeline — language is explicitly passed
                setState('processing');
                try {
                    const response = await apiService.voiceChat(audioBlob, conversationId, {
                        age: age ?? undefined,
                        gender: gender ?? undefined,
                        voiceId,
                        speed,
                        language,
                    });

                    onVoiceChatComplete?.(response);

                    // Auto-play TTS response if available
                    if (response.audio_base64) {
                        playAudio(response.audio_base64, response.audio_format);
                    } else {
                        setState('idle');
                    }
                } catch (err) {
                    const msg = err instanceof Error ? err.message : 'Voice chat failed';
                    console.error('[useVoice] Voice chat error:', err);
                    setError(msg);
                    onError?.(msg);
                    setState('idle');
                }
            };

            // Start recording with 250ms timeslice for chunked data
            mediaRecorder.start(250);
            setState('recording');

            // Start volume monitoring for visualization + silence detection
            startVolumeMonitoring(stream);

            // Start duration timer
            setRecordingDuration(0);
            durationTimerRef.current = window.setInterval(() => {
                setRecordingDuration(prev => prev + 1);
            }, 1000);

        } catch (err: any) {
            let msg = 'Microphone access was denied.';
            if (err?.name === 'NotFoundError') {
                msg = 'No microphone found. Please connect a microphone.';
            } else if (err?.name === 'NotAllowedError') {
                msg = 'Microphone permission denied. Please allow access in browser settings.';
            }
            console.error('[useVoice] Mic error:', err);
            setError(msg);
            onError?.(msg);
            setState('idle');
        }
    }, [isSupported, conversationId, language, age, gender, voiceId, speed, onVoiceChatComplete, onError, cleanup, startVolumeMonitoring]);

    // ── Stop recording ───────────────────────────────────────────────────────

    const stopRecording = useCallback(() => {
        if (mediaRecorderRef.current?.state === 'recording') {
            mediaRecorderRef.current.stop();
        }
    }, []);

    // ── Toggle ───────────────────────────────────────────────────────────────

    const toggleRecording = useCallback(async () => {
        if (state === 'recording') {
            stopRecording();
        } else if (state === 'idle') {
            await startRecording();
        }
    }, [state, startRecording, stopRecording]);

    // ── Audio playback ───────────────────────────────────────────────────────

    const playAudio = useCallback((base64Audio: string, format: string = 'mp3') => {
        // Stop any existing audio directly — avoids stale stopAudio closure
        if (audioPlayerRef.current) {
            audioPlayerRef.current.pause();
            audioPlayerRef.current.currentTime = 0;
            audioPlayerRef.current = null;
        }

        try {
            const binaryStr = atob(base64Audio);
            const bytes = new Uint8Array(binaryStr.length);
            for (let i = 0; i < binaryStr.length; i++) {
                bytes[i] = binaryStr.charCodeAt(i);
            }

            const mimeType = format === 'wav' ? 'audio/wav' : 'audio/mpeg';
            const blob = new Blob([bytes], { type: mimeType });
            const url = URL.createObjectURL(blob);

            const audio = new Audio(url);
            audioPlayerRef.current = audio;

            audio.onplay = () => setState('playing');
            audio.onended = () => {
                setState('idle');
                URL.revokeObjectURL(url);
            };
            audio.onerror = () => {
                console.error('[useVoice] Audio playback error');
                setState('idle');
                URL.revokeObjectURL(url);
            };

            audio.play().catch(err => {
                console.warn('[useVoice] Auto-play blocked:', err);
                setState('idle');
                URL.revokeObjectURL(url);
            });
        } catch (err) {
            console.error('[useVoice] Playback error:', err);
            setState('idle');
        }
    }, []);

    const speakText = useCallback(async (text: string) => {
        if (!text.trim()) return;

        setState('processing');
        try {
            const audioBuffer = await apiService.textToSpeech(text, voiceId, speed, 'mp3', language);
            // Chunked encoding — avoids stack overflow on large buffers
            const bytes = new Uint8Array(audioBuffer);
            let binary = '';
            const chunkSize = 8192;
            for (let i = 0; i < bytes.length; i += chunkSize) {
                binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
            }
            playAudio(btoa(binary), 'mp3');
        } catch (err) {
            console.error('[useVoice] TTS error:', err);
            setError('Text-to-speech failed');
            setState('idle');
        }
    }, [voiceId, speed, language, playAudio]);

    const stopAudio = useCallback(() => {
        if (audioPlayerRef.current) {
            audioPlayerRef.current.pause();
            audioPlayerRef.current.currentTime = 0;
            audioPlayerRef.current = null;
        }
        if (state === 'playing') {
            setState('idle');
        }
    }, [state]);

    return {
        state,
        startRecording,
        stopRecording,
        toggleRecording,
        playAudio,
        speakText,
        stopAudio,
        volume,
        frequencyData,
        recordingDuration,
        isSupported,
        error,
    };
}
