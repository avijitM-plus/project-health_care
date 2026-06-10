/**
 * VoiceModePanel — Full-screen Voice Mode overlay for IASIS AI.
 *
 * Shows:
 *  - Animated waveform (real-time mic level or playback indicator)
 *  - State indicator (listening / thinking / speaking)
 *  - Live transcript (what you said)
 *  - AI voice response (short spoken version)
 *  - Tap-to-speak / interrupt button
 *  - Latency metrics in development
 */
import React, { useEffect, useRef } from 'react';
import { Mic, MicOff, Brain, Volume2, Square, X } from 'lucide-react';
import { useVoiceMode, type VoiceModeState } from '../hooks/useVoiceMode';
import type { VoiceStreamEvent } from '../types/api';
import './VoiceModePanel.css';

interface VoiceModePanelProps {
    conversationId: string;
    language: string;
    voiceId?: string;
    speed?: number;
    age?: number | null;
    gender?: string | null;
    onClose: () => void;
    onTranscript?: (text: string) => void;
    onClinical?: (event: VoiceStreamEvent) => void;
}

const STATE_LABELS: Record<VoiceModeState, string> = {
    idle: 'Tap to speak',
    recording: 'Listening...',
    processing: 'Thinking...',
    playing: 'Speaking...',
    error: 'Something went wrong',
};

const STATE_LABELS_BN: Record<VoiceModeState, string> = {
    idle: 'বলতে ট্যাপ করুন',
    recording: 'শুনছি...',
    processing: 'ভাবছি...',
    playing: 'বলছি...',
    error: 'সমস্যা হয়েছে',
};

export const VoiceModePanel: React.FC<VoiceModePanelProps> = ({
    conversationId,
    language,
    voiceId,
    speed,
    age,
    gender,
    onClose,
    onTranscript,
    onClinical,
}) => {
    const isBn = language === 'bn';
    const labels = isBn ? STATE_LABELS_BN : STATE_LABELS;

    const {
        state,
        volume,
        frequencyData,
        transcript,
        voiceResponse,
        toggleRecording,
        interrupt,
        isSupported,
        error,
        latencyMetrics,
    } = useVoiceMode({
        conversationId,
        language,
        voiceId,
        speed,
        age,
        gender,
        onTranscript,
        onClinical,
    });

    // Waveform canvas
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animRef = useRef<number | null>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const draw = () => {
            const { width, height } = canvas;
            ctx.clearRect(0, 0, width, height);

            if (state === 'idle' || state === 'error') {
                // Flat line for idle
                ctx.beginPath();
                ctx.moveTo(0, height / 2);
                ctx.lineTo(width, height / 2);
                ctx.strokeStyle = 'rgba(99,102,241,0.3)';
                ctx.lineWidth = 2;
                ctx.stroke();
            } else if (frequencyData && (state === 'recording')) {
                // Bar waveform from mic
                const bars = Math.min(frequencyData.length, 48);
                const barWidth = width / bars;
                for (let i = 0; i < bars; i++) {
                    const v = frequencyData[i] / 255;
                    const barH = Math.max(4, v * height * 0.85);
                    const x = i * barWidth;
                    const y = (height - barH) / 2;
                    const alpha = 0.4 + v * 0.6;
                    ctx.fillStyle = `rgba(99,102,241,${alpha})`;
                    ctx.beginPath();
                    ctx.roundRect(x + 1, y, barWidth - 2, barH, 3);
                    ctx.fill();
                }
            } else {
                // Animated sine wave for processing/playing
                const t = Date.now() / 400;
                const amp = state === 'playing' ? 18 + volume * 20 : 8;
                ctx.beginPath();
                for (let x = 0; x <= width; x++) {
                    const y = height / 2 + amp * Math.sin((x / width) * 4 * Math.PI + t);
                    if (x === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.strokeStyle = state === 'playing' ? '#818cf8' : '#6366f1';
                ctx.lineWidth = 2.5;
                ctx.stroke();
            }

            animRef.current = requestAnimationFrame(draw);
        };

        animRef.current = requestAnimationFrame(draw);
        return () => {
            if (animRef.current) cancelAnimationFrame(animRef.current);
        };
    }, [state, frequencyData, volume]);

    const handleMainButton = () => {
        toggleRecording();
    };

    const StateIcon = () => {
        if (state === 'recording') return <Mic size={44} />;
        if (state === 'processing') return <Brain size={44} />;
        if (state === 'playing') return <Volume2 size={44} />;
        return <Mic size={44} />;
    };

    const isDev = import.meta.env.DEV;

    return (
        <div className="vmp-overlay" role="dialog" aria-modal="true" aria-label="Voice Mode">
            <div className="vmp-panel">
                {/* Header */}
                <div className="vmp-header">
                    <div className="vmp-title">
                        <Mic size={18} />
                        <span>{isBn ? 'ভয়েস মোড' : 'Voice Mode'}</span>
                    </div>
                    <button className="vmp-close-btn" onClick={onClose} aria-label="Close voice mode">
                        <X size={20} />
                    </button>
                </div>

                {/* State icon */}
                <div className={`vmp-state-icon vmp-state-${state}`}>
                    <StateIcon />
                </div>

                {/* Waveform canvas */}
                <div className="vmp-waveform-container">
                    <canvas
                        ref={canvasRef}
                        className="vmp-waveform"
                        width={320}
                        height={80}
                    />
                </div>

                {/* Status label */}
                <p className={`vmp-status-label vmp-status-${state}`}>
                    {error ? error : labels[state]}
                </p>

                {/* Transcript */}
                {transcript && (
                    <div className="vmp-transcript">
                        <span className="vmp-role-label">{isBn ? 'আপনি:' : 'You:'}</span>
                        <span className="vmp-text">{transcript}</span>
                    </div>
                )}

                {/* AI voice response */}
                {voiceResponse && (
                    <div className="vmp-response">
                        <span className="vmp-role-label">IASIS:</span>
                        <span className="vmp-text">{voiceResponse}</span>
                    </div>
                )}

                {/* Controls */}
                <div className="vmp-controls">
                    {(state === 'playing' || state === 'processing') ? (
                        <button
                            className="vmp-interrupt-btn"
                            onClick={interrupt}
                            aria-label="Interrupt"
                        >
                            <Square size={20} />
                            <span>{isBn ? 'থামুন' : 'Interrupt'}</span>
                        </button>
                    ) : (
                        <button
                            className={`vmp-main-btn ${state === 'recording' ? 'vmp-recording' : ''}`}
                            onClick={handleMainButton}
                            disabled={!isSupported}
                            aria-label={state === 'recording' ? 'Stop recording' : 'Start recording'}
                        >
                            {state === 'recording' ? <MicOff size={28} /> : <Mic size={28} />}
                        </button>
                    )}
                </div>

                {/* Latency metrics (dev only) */}
                {isDev && latencyMetrics && (
                    <div className="vmp-metrics">
                        STT {latencyMetrics.stt}s &nbsp;|&nbsp;
                        LLM {latencyMetrics.llm}s &nbsp;|&nbsp;
                        TTS {latencyMetrics.tts}s &nbsp;|&nbsp;
                        Total {latencyMetrics.total}s
                    </div>
                )}
            </div>
        </div>
    );
};
