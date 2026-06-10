/**
 * useVoiceMode — Streaming voice mode hook for IASIS AI.
 *
 * Uses Server-Sent Events for streaming audio delivery.
 * Uses Web Audio API AudioContext for gapless sentence-by-sentence playback.
 * First audio chunk plays ~300ms after LLM response (vs 2-3s in non-streaming mode).
 * Supports barge-in: interrupt ongoing speech by calling interrupt().
 */
import { useState, useRef, useCallback, useEffect } from 'react';
import { apiService } from '../services/api';
import type { VoiceStreamEvent } from '../types/api';

export type VoiceModeState = 'idle' | 'recording' | 'processing' | 'playing' | 'error';

export interface LatencyMetrics {
    stt: number;
    llm: number;
    tts: number;
    total: number;
}

interface UseVoiceModeOptions {
    conversationId: string;
    language: string;
    voiceId?: string;
    speed?: number;
    age?: number | null;
    gender?: string | null;
    silenceThreshold?: number;
    silenceDuration?: number;
    onTranscript?: (text: string) => void;
    onClinical?: (event: VoiceStreamEvent) => void;
    onError?: (message: string) => void;
}

interface UseVoiceModeReturn {
    state: VoiceModeState;
    volume: number;
    frequencyData: Uint8Array | null;
    transcript: string;
    voiceResponse: string;
    startRecording: () => Promise<void>;
    stopRecording: () => void;
    toggleRecording: () => Promise<void>;
    interrupt: () => void;
    isSupported: boolean;
    error: string | null;
    latencyMetrics: LatencyMetrics | null;
}

export function useVoiceMode(options: UseVoiceModeOptions): UseVoiceModeReturn {
    const {
        conversationId,
        language,
        voiceId,
        speed = 1.0,
        age,
        gender,
        silenceThreshold = 15,
        silenceDuration = 2500,
        onTranscript,
        onClinical,
        onError,
    } = options;

    const [state, setState] = useState<VoiceModeState>('idle');
    const [volume, setVolume] = useState(0);
    const [frequencyData, setFrequencyData] = useState<Uint8Array | null>(null);
    const [transcript, setTranscript] = useState('');
    const [voiceResponse, setVoiceResponse] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [latencyMetrics, setLatencyMetrics] = useState<LatencyMetrics | null>(null);

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const streamRef = useRef<MediaStream | null>(null);
    const micCtxRef = useRef<AudioContext | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const animFrameRef = useRef<number | null>(null);
    const abortRef = useRef<AbortController | null>(null);

    // Web Audio API context for gapless streaming playback
    const playCtxRef = useRef<AudioContext | null>(null);
    const nextPlayTimeRef = useRef<number>(0);
    const activeSourcesRef = useRef<AudioBufferSourceNode[]>([]);
    const interruptedRef = useRef(false);
    const audioChunkCountRef = useRef(0);

    const isSupported =
        typeof navigator !== 'undefined' &&
        !!navigator.mediaDevices?.getUserMedia &&
        typeof MediaRecorder !== 'undefined' &&
        typeof AudioContext !== 'undefined';

    // ── Cleanup mic resources ─────────────────────────────────────────────────

    const cleanupMic = useCallback(() => {
        if (animFrameRef.current) {
            cancelAnimationFrame(animFrameRef.current);
            animFrameRef.current = null;
        }
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(t => t.stop());
            streamRef.current = null;
        }
        if (micCtxRef.current?.state !== 'closed') {
            micCtxRef.current?.close().catch(() => {});
            micCtxRef.current = null;
        }
        analyserRef.current = null;
        setVolume(0);
        setFrequencyData(null);
    }, []);

    // ── Barge-in: stop all ongoing playback ──────────────────────────────────

    const interrupt = useCallback(() => {
        interruptedRef.current = true;
        abortRef.current?.abort();

        for (const src of activeSourcesRef.current) {
            try { src.stop(0); } catch { /* already stopped */ }
        }
        activeSourcesRef.current = [];
        nextPlayTimeRef.current = 0;
        audioChunkCountRef.current = 0;

        setState(s => (s === 'playing' || s === 'processing' ? 'idle' : s));
    }, []);

    // ── Play one audio chunk (gapless scheduling via Web Audio API) ───────────

    const playAudioChunk = useCallback(async (base64Audio: string): Promise<void> => {
        if (interruptedRef.current) return;

        if (!playCtxRef.current || playCtxRef.current.state === 'closed') {
            playCtxRef.current = new AudioContext();
            nextPlayTimeRef.current = 0;
        }
        const ctx = playCtxRef.current;

        try {
            const binary = atob(base64Audio);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

            const audioBuffer = await ctx.decodeAudioData(bytes.buffer.slice(0));
            if (interruptedRef.current) return;

            const source = ctx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(ctx.destination);
            activeSourcesRef.current.push(source);

            const now = ctx.currentTime;
            if (nextPlayTimeRef.current < now) nextPlayTimeRef.current = now;

            source.start(nextPlayTimeRef.current);
            nextPlayTimeRef.current += audioBuffer.duration;

            source.onended = () => {
                const idx = activeSourcesRef.current.indexOf(source);
                if (idx !== -1) activeSourcesRef.current.splice(idx, 1);
                if (activeSourcesRef.current.length === 0 && !interruptedRef.current) {
                    setState('idle');
                }
            };
        } catch (err) {
            console.warn('[useVoiceMode] Chunk decode error:', err);
        }
    }, []);

    // ── Volume monitoring + silence detection ─────────────────────────────────

    const startVolumeMonitoring = useCallback(
        (stream: MediaStream, onSilence: () => void) => {
            try {
                const ctx = new AudioContext();
                const analyser = ctx.createAnalyser();
                analyser.fftSize = 256;
                analyser.smoothingTimeConstant = 0.8;
                ctx.createMediaStreamSource(stream).connect(analyser);
                micCtxRef.current = ctx;
                analyserRef.current = analyser;

                const data = new Uint8Array(analyser.frequencyBinCount);
                let silenceStart: number | null = null;

                const tick = () => {
                    if (!analyserRef.current) return;
                    analyserRef.current.getByteFrequencyData(data);
                    const avg = data.reduce((s, v) => s + v, 0) / data.length;
                    setVolume(avg / 255);
                    setFrequencyData(new Uint8Array(data));

                    if (avg < silenceThreshold) {
                        if (silenceStart === null) silenceStart = Date.now();
                        else if (Date.now() - silenceStart > silenceDuration) {
                            onSilence();
                            return;
                        }
                    } else {
                        silenceStart = null;
                    }
                    animFrameRef.current = requestAnimationFrame(tick);
                };
                animFrameRef.current = requestAnimationFrame(tick);
            } catch (err) {
                console.warn('[useVoiceMode] Volume monitoring unavailable:', err);
            }
        },
        [silenceThreshold, silenceDuration],
    );

    // ── Process audio through the streaming pipeline ──────────────────────────

    const processStream = useCallback(
        async (blob: Blob) => {
            setState('processing');
            interruptedRef.current = false;
            audioChunkCountRef.current = 0;

            const ctrl = apiService.voiceChatStream(
                blob,
                conversationId,
                {
                    age: age ?? undefined,
                    gender: gender ?? undefined,
                    voiceId,
                    speed,
                    language,
                },
                async (event) => {
                    switch (event.type) {
                        case 'transcript':
                            setTranscript(event.text ?? '');
                            onTranscript?.(event.text ?? '');
                            break;

                        case 'audio_chunk':
                            if (event.audio_b64 && !interruptedRef.current) {
                                setState('playing');
                                audioChunkCountRef.current++;
                                await playAudioChunk(event.audio_b64);
                            }
                            break;

                        case 'clinical':
                            setVoiceResponse(event.voice_response ?? '');
                            onClinical?.(event);
                            break;

                        case 'done':
                            setLatencyMetrics({
                                stt: event.stt_time ?? 0,
                                llm: event.llm_time ?? 0,
                                tts: event.tts_time ?? 0,
                                total: event.total_time ?? 0,
                            });
                            if (audioChunkCountRef.current === 0) setState('idle');
                            break;

                        case 'error':
                            throw new Error(event.message ?? 'Stream error');
                    }
                },
                (err) => {
                    const msg = err.message ?? 'Voice processing failed';
                    setError(msg);
                    onError?.(msg);
                    setState('error');
                },
            );

            abortRef.current = ctrl;
        },
        [conversationId, language, voiceId, speed, age, gender,
            onTranscript, onClinical, onError, playAudioChunk],
    );

    // ── Start recording ───────────────────────────────────────────────────────

    const startRecording = useCallback(async () => {
        if (!isSupported) {
            const msg = 'Voice recording is not supported in this browser.';
            setError(msg);
            onError?.(msg);
            return;
        }

        setError(null);
        setTranscript('');
        setVoiceResponse('');

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

            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : 'audio/webm';
            const recorder = new MediaRecorder(stream, { mimeType });
            mediaRecorderRef.current = recorder;

            recorder.ondataavailable = e => {
                if (e.data.size > 0) audioChunksRef.current.push(e.data);
            };

            recorder.onstop = async () => {
                const mime = recorder.mimeType || 'audio/webm';
                cleanupMic();
                const blob = new Blob(audioChunksRef.current, { type: mime });
                if (blob.size < 100) { setState('idle'); return; }
                await processStream(blob);
            };

            recorder.start(250);
            setState('recording');
            startVolumeMonitoring(stream, stopRecording);
        } catch (err: any) {
            const msg =
                err?.name === 'NotFoundError'
                    ? 'No microphone found.'
                    : 'Microphone access denied.';
            setError(msg);
            onError?.(msg);
            setState('error');
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isSupported, onError, cleanupMic, startVolumeMonitoring, processStream]);

    // ── Stop recording ────────────────────────────────────────────────────────

    const stopRecording = useCallback(() => {
        if (mediaRecorderRef.current?.state === 'recording') {
            mediaRecorderRef.current.stop();
        }
    }, []);

    // ── Toggle ────────────────────────────────────────────────────────────────

    const toggleRecording = useCallback(async () => {
        if (state === 'recording') {
            stopRecording();
        } else if (state === 'playing' || state === 'processing') {
            interrupt();
        } else if (state === 'idle' || state === 'error') {
            await startRecording();
        }
    }, [state, startRecording, stopRecording, interrupt]);

    // ── Cleanup on unmount ────────────────────────────────────────────────────

    useEffect(() => {
        return () => {
            cleanupMic();
            abortRef.current?.abort();
            for (const src of activeSourcesRef.current) {
                try { src.stop(0); } catch { /* ignore */ }
            }
            playCtxRef.current?.close().catch(() => {});
        };
    }, [cleanupMic]);

    return {
        state,
        volume,
        frequencyData,
        transcript,
        voiceResponse,
        startRecording,
        stopRecording,
        toggleRecording,
        interrupt,
        isSupported,
        error,
        latencyMetrics,
    };
}
